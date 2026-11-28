"""Oracle write diagnostic for LiveMem 71a6a8b. All outputs stay under --run.

Preparation never loads model weights. Each worker performs independent,
batch-size-one, cache-free forwards; vocabulary logits exist only at labels.
"""
from __future__ import annotations

import argparse
import bisect
import collections
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import zipfile

ARMS = ("write_all", "oracle", "random", "write_nothing", "oracle_keep_decay")


def json_default(value):
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def save(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=json_default) + "\n")
    os.replace(tmp, path)


def publish_arrays(path, **arrays):
    """Publish a complete ZIP stream without seeking on the persistent mount."""
    import numpy as np
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    payload = buffer.getvalue()
    tmp = path.with_name(path.name+'.tmp')
    tmp.write_bytes(payload)
    os.replace(tmp,path)
    actual = path.read_bytes()
    assert actual == payload, f'array publication mismatch: {path}'
    with zipfile.ZipFile(io.BytesIO(actual)) as archive:
        assert archive.testzip() is None
    return hashlib.sha256(actual).hexdigest()


def matched_sessions(lengths, evidence, seed):
    """Randomized subset-sum: closest whole-session non-evidence token budget."""
    budget = sum(lengths[i] for i in evidence)
    order = [i for i in range(len(lengths)) if i not in evidence]
    random.Random(seed).shuffle(order)
    states, reachable = [], 1
    for i in order:
        states.append(reachable)
        reachable |= reachable << lengths[i]
    below = (reachable & ((1 << (budget + 1)) - 1)).bit_length() - 1
    above_bits = reachable >> budget
    above = budget + ((above_bits & -above_bits).bit_length() - 1) if above_bits else None
    target = below if above is None or budget - below <= above - budget else above
    total = target
    selected = []
    for i, before in reversed(list(zip(order, states))):
        if not ((before >> target) & 1):
            selected.append(i)
            target -= lengths[i]
    assert target == 0
    assert set(selected).isdisjoint(evidence)
    return sorted(selected), budget, total


def matched_random_spans(sessions, evidence, seed):
    """Match whole sessions first, then crop at most one session for exact size."""
    lengths = [e-s for s,e in sessions]
    selected, budget, whole_total = matched_sessions(lengths, evidence, seed)
    spans = {i:tuple(sessions[i]) for i in selected}
    residual = budget-whole_total
    rng = random.Random(seed+1)
    partial = None
    if residual > 0:
        candidates = [i for i in range(len(sessions))
                      if i not in evidence and i not in selected and lengths[i]>=residual]
        if not candidates:
            raise ValueError('insufficient non-evidence tokens for a matched control')
        i = rng.choice(candidates)
        s,e = sessions[i]
        start = s+rng.randrange(e-s-residual+1)
        spans[i] = (start,start+residual)
        partial = i
    elif residual < 0:
        candidates = [i for i in selected if lengths[i]>=-residual]
        i = rng.choice(candidates)
        s,e = sessions[i]
        kept = e-s+residual
        start = s+rng.randrange(e-s-kept+1)
        spans[i] = (start,start+kept)
        partial = i
    assert sum(e-s for s,e in spans.values()) == budget
    return sorted(spans.values()), sorted(spans), budget, whole_total, partial


def schedule(length, sink_end, question_start, chunk_size, window):
    """Same FIFO eviction as train/sft/eviction.py, with a system sink."""
    import numpy as np
    spans = [(0, sink_end)]
    for lo, hi in [(sink_end, question_start), (question_start, length)]:
        spans.extend((s, min(s + chunk_size, hi)) for s in range(lo, hi, chunk_size))
    chunk = np.zeros(length, dtype=np.int32)
    evict = np.full(length, len(spans), dtype=np.int32)
    live, tokens = collections.deque(), sink_end
    for i, (s, e) in enumerate(spans):
        chunk[s:e] = i
        if i == 0:
            continue
        live.append(i)
        tokens += e - s
        while len(live) > 1 and tokens > window:
            j = live.popleft()
            a, b = spans[j]
            evict[a:b] = i
            tokens -= b - a
    return spans, chunk, evict


def make_mask(meta, arm, length):
    import numpy as np
    if arm == "write_all":
        return None
    mask = np.ones(length, dtype=np.float32)
    for s, e in meta["session_spans"]:
        mask[s:e] = 0
    selected = meta["random_sessions"] if arm == "random" else meta["evidence"]
    if arm == "write_nothing":
        selected = []
    for i in selected:
        s, e = meta["session_spans"][i]
        mask[s:e] = 1
    if arm == "random" and "random_spans" in meta:
        for s,e in meta["session_spans"]:
            mask[s:e] = 0
        for s,e in meta["random_spans"]:
            mask[s:e] = 1
    assert mask[:meta["sink_end"]].all()
    assert mask[meta["question_start"]:].all()
    return mask


def truncation_indices(meta, arr, window):
    """Keep ChatML system/user prefix, visible history and all QA tokens."""
    import numpy as np
    prefix_end = meta['session_spans'][0][0]
    question_start = meta['question_start']
    visible = np.flatnonzero(arr['evict_step'] > meta['question_chunk'])
    history = visible[(visible >= prefix_end) & (visible < question_start)]
    qa = np.arange(question_start,len(arr['input_ids']))
    budget = window-prefix_end-len(qa)
    if budget < 0:
        raise ValueError('system/user prefix and QA exceed the truncation window')
    history = history[-budget:] if budget else history[:0]
    indices = np.concatenate([np.arange(prefix_end),history,qa])
    assert len(indices) <= window
    assert np.array_equal(indices[:prefix_end],np.arange(prefix_end))
    return indices


def prepare(args):
    import numpy as np
    from transformers import AutoTokenizer
    sys.path.insert(0, str(args.repo / "tools/data_process"))
    import format as fmt
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    rows = json.loads(args.data.read_text())
    (args.run/'data/arrays_v3').mkdir(exist_ok=True)
    eligible, exclusions = [], []
    for ri, row in enumerate(rows):
        qid = str(row["question_id"])
        reason = None
        evidence = [i for i, sid in enumerate(row["haystack_session_ids"])
                    if sid in set(row.get("answer_session_ids") or [])]
        if qid.endswith("_abs"):
            reason = "abstention"
        elif not evidence:
            reason = "no_evidence"
        if reason:
            exclusions.append({"id": qid, "reason": reason})
            continue
        dates = row.get("haystack_dates") or [None] * len(row["haystack_sessions"])
        docs = [(f"[Date: {d}]\n" if d else "") + "\n".join(
            f"{t.get('role', '')}: {t.get('content', '')}" for t in turns)
            for turns, d in zip(row["haystack_sessions"], dates)]
        question = fmt.user_single("longmemeval_s", row["question"], reason=False)
        gold = fmt.gold_single(str(row["answer"]))
        parts = [fmt.system_segment(), "<|im_start|>user\n"]
        parts += [doc + "\n\n" for doc in docs]
        parts += [question + "<|im_end|>\n<|im_start|>assistant\n", gold + "<|im_end|>\n"]
        full = "".join(parts)
        canonical = tokenizer.apply_chat_template([
            {"role": "system", "content": fmt.SYSTEM_PROMPT},
            {"role": "user", "content": "\n\n".join(docs) + "\n\n" + question},
            {"role": "assistant", "content": gold}], tokenize=False, enable_thinking=False)
        if full != canonical:
            raise AssertionError(f"chat template mismatch: {qid}")
        piece_ids = [tokenizer(p, add_special_tokens=False).input_ids for p in parts]
        tokenized = tokenizer(full, add_special_tokens=False, return_offsets_mapping=True)
        ids = np.asarray(tokenized.input_ids, dtype=np.int32)
        if tokenized.input_ids != [i for part in piece_ids for i in part]:
            raise AssertionError(f"segmented tokenization differs from whole prompt: {qid}")
        offsets = np.cumsum([0] + [len(p) for p in piece_ids]).tolist()
        sessions = list(zip(offsets[2:2+len(docs)], offsets[3:3+len(docs)]))
        question_start = offsets[-3]
        sink_end = offsets[1]
        spans, chunk, evict = schedule(len(ids), sink_end, question_start, args.chunk, args.window)
        question_chunk = int(chunk[question_start])
        fully_evicted = all(bool(np.all(evict[sessions[i][0]:sessions[i][1]] <= question_chunk))
                            for i in evidence)
        if not fully_evicted:
            exclusions.append({"id": qid, "reason": "evidence_not_fully_evicted"})
            continue
        if len(ids) > 262144:
            exclusions.append({"id": qid, "reason": "beyond_checkpoint_positional_horizon"})
            continue
        seed = int.from_bytes(hashlib.sha256(f"{args.seed}:{qid}".encode()).digest()[:8], "little")
        try:
            random_spans, selected, budget, whole_total, partial = matched_random_spans(sessions,evidence,seed)
        except ValueError:
            exclusions.append({"id":qid,"reason":"insufficient_non_evidence_tokens_for_matched_control"})
            continue
        answer_char_start = sum(map(len, parts[:-1])) + len("```text\nAnswer: ")
        answer_char_end = answer_char_start + len(str(row["answer"]))
        labels = np.full(len(ids), -100, dtype=np.int32)
        for i, (s, e) in enumerate(tokenized.offset_mapping):
            if e > answer_char_start and s < answer_char_end:
                labels[i] = ids[i]
        assert np.any(labels[1:] != -100), qid
        meta = {"id": qid, "row_index": ri, "question_type": row["question_type"],
                "length": len(ids), "sink_end": sink_end, "question_start": question_start,
                "question_chunk": question_chunk, "session_spans": sessions,
                "evidence": evidence, "random_sessions": selected,
                "random_spans":random_spans,"random_partial_session":partial,
                "random_whole_session_tokens_before_adjustment":whole_total,
                "oracle_history_tokens": budget, "random_history_tokens": budget,
                "answer_tokens": int((labels != -100).sum()), "chunk_spans": spans,
                "piecewise_tokens_equal": True, "chat_template_equal": True,
                "all_evidence_evicted": fully_evicted,
                "input_ids_sha256": hashlib.sha256(ids.tobytes()).hexdigest()}
        fn = f"arrays_v3/{ri:04d}.npz"
        meta['array_sha256'] = publish_arrays(args.run / "data" / fn,
            input_ids=ids, labels=labels, chunk_id=chunk, evict_step=evict)
        meta["array_file"] = fn
        for arm in ARMS:
            make_mask(meta, arm, len(ids))
        eligible.append(meta)
        if len(eligible) % 25 == 0:
            print(f"prepared {len(eligible)} eligible; visited {ri+1}/{len(rows)}", flush=True)
    # Seeded interleaving gives broad type coverage without loss-based selection.
    by_type = collections.defaultdict(list)
    for e in eligible:
        by_type[e["question_type"]].append(e)
    rng = random.Random(args.seed)
    for group in by_type.values():
        rng.shuffle(group)
    ordered = []
    while any(by_type.values()):
        for typ in sorted(by_type):
            if by_type[typ]:
                ordered.append(by_type[typ].pop())
    save(args.run / "data/eligible.json", ordered)
    save(args.run / "data/exclusions.json", exclusions)
    save(args.run / "data/preparation.json", {
        "rows": len(rows), "eligible": len(ordered), "excluded": len(exclusions),
        "by_type": dict(collections.Counter(e["question_type"] for e in ordered)),
        "exclusion_reasons": dict(collections.Counter(e["reason"] for e in exclusions)),
        "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "window": args.window, "chunk": args.chunk, "seed": args.seed,
        "label_scope": "raw gold answer tokens within official SFT gold_single wrapper; no wrapper/EOS loss",
        "qa_boundary": "question begins a new chunk; system is its own sink chunk",
        "random_policy": "randomized whole non-evidence sessions, cropping at most one session to exactly match oracle token count"})
    print(f"Preparation complete: {len(ordered)} eligible / {len(rows)}", flush=True)


def run(args):
    import numpy as np
    import torch
    from transformers.modeling_utils import PreTrainedModel
    sys.path.insert(0, str(args.repo))
    from models import MemoryQwen3ForCausalLM
    from train.sft.eviction import per_token_eviction
    outpath = args.run / f"raw/worker_{args.rank}.jsonl"
    done = set()
    results_dir = args.run / "results"
    results_dir.mkdir(exist_ok=True)
    for path in results_dir.glob("*.json"):
        record = json.loads(path.read_text())
        done.add((record["id"], record["arm"]))
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    torch.cuda.set_device(args.device)
    device = torch.device(f"cuda:{args.device}")
    # The official subclass override assumes a model, whereas output_loading_info
    # returns a tuple. Bind the same HF loader directly and apply its freeze hook.
    model, loading = PreTrainedModel.from_pretrained.__func__(MemoryQwen3ForCausalLM,
        args.checkpoint, dtype=torch.bfloat16, attn_implementation="flex_attention",
        local_files_only=True, output_loading_info=True)
    model._apply_freeze_main_path()
    if loading.get("missing_keys") or loading.get("unexpected_keys") or loading.get("mismatched_keys"):
        raise RuntimeError(f"checkpoint loading mismatch: {loading}")
    model = model.eval().to(device)
    modules = [m for m in model.modules() if m.__class__.__name__ == "MemoryGatedDeltaNet2"]
    assert len(modules) == 36
    runtime_file=f'control_worker_{args.rank}_runtime.json' if args.controls_only else f'worker_{args.rank}_runtime.json'
    save(args.run / runtime_file, {
        "torch": torch.__version__, "gpu": torch.cuda.get_device_name(device),
        "checkpoint": str(args.checkpoint), "memory_layers": len(modules),
        "parameters": sum(p.numel() for p in model.parameters()), "loading_info": loading})
    while args.wait_for_data and not (args.run / "data/preparation.json").is_file():
        time.sleep(2)
    selected = json.loads((args.run / "data/eligible.json").read_text())[:args.limit or None]
    selected = selected[args.rank::args.world]

    def loss(arr, mask, keep_decay=False, indices=None, reset_positions=False):
        for m in modules:
            m.skip_keeps_decay = keep_decay
        idx = np.arange(len(arr["input_ids"])) if indices is None else indices
        kwargs = {k: torch.tensor(arr[k][idx].astype(np.int64), device=device)[None]
                  for k in ["input_ids", "labels", "chunk_id", "evict_step"]}
        if indices is not None:
            kwargs.pop("chunk_id")
            kwargs.pop("evict_step")
            kwargs["position_ids"] = torch.tensor(
                np.arange(len(idx)) if reset_positions else idx, device=device)[None]
        if mask is not None:
            kwargs["write_mask"] = torch.tensor(mask[idx], device=device)[None]
        torch.cuda.reset_peak_memory_stats(device)
        start = time.monotonic()
        with torch.inference_mode():
            result = model(**kwargs, use_cache=False)
        torch.cuda.synchronize(device)
        value = float(result.loss)
        assert result.logits is None, "full-vocabulary sequence logits were materialized"
        assert math.isfinite(value)
        return value, time.monotonic()-start, torch.cuda.max_memory_allocated(device)

    with outpath.open("a", buffering=1) as out:
        for n, meta in enumerate(selected):
            payload = (args.run / "data" / meta["array_file"]).read_bytes()
            assert hashlib.sha256(payload).hexdigest() == meta['array_sha256']
            with np.load(io.BytesIO(payload)) as packed:
                arr = {key:packed[key] for key in packed.files}
            c, e = per_token_eviction(meta["chunk_spans"], 10**9, args.window, 1)
            assert np.array_equal(c.numpy(), arr["chunk_id"])
            assert np.array_equal(e.numpy(), arr["evict_step"])
            if args.controls_only:
                controls_dir=args.run/'controls'
                controls_dir.mkdir(exist_ok=True)
                control_path=controls_dir/f"{meta['row_index']:04d}.json"
                if not control_path.exists():
                    primary=json.loads((results_dir/f"{meta['row_index']:04d}_write_nothing.json").read_text())
                    idx=truncation_indices(meta,arr,args.window)
                    none=make_mask(meta,'write_nothing',meta['length'])
                    trunc_all,seconds_all,peak_all=loss(arr,None,indices=idx,reset_positions=True)
                    trunc_none,seconds_none,peak_none=loss(arr,none,indices=idx,reset_positions=True)
                    save(control_path,{'id':meta['id'],'question_type':meta['question_type'],
                        'full_write_nothing_loss':primary['loss'],'truncation_loss':trunc_all,
                        'truncation_no_history_write_loss':trunc_none,
                        'truncation_signed_diff':primary['loss']-trunc_all,
                        'matched_mask_signed_diff':primary['loss']-trunc_none,
                        'user_prefix_preserved':True,'sequence_length':len(idx),
                        'window':args.window,'answer_tokens':meta['answer_tokens'],
                        'seconds':[seconds_all,seconds_none],
                        'peak_allocated_bytes':[peak_all,peak_none]})
                save(args.run/f'control_progress_{args.rank}.json',{
                    'completed_questions':n+1,'assigned_questions':len(selected),'status':'running'})
                continue
            values = {}
            for arm in ARMS:
                if (meta["id"], arm) in done:
                    continue
                value, seconds, peak = loss(arr, make_mask(meta, arm, meta["length"]), arm == "oracle_keep_decay")
                record = {"id": meta["id"], "question_type": meta["question_type"], "arm": arm,
                          "loss": value, "answer_tokens": meta["answer_tokens"],
                          "seconds": seconds, "peak_allocated_bytes": peak,
                          "oracle_history_tokens": meta["oracle_history_tokens"],
                          "random_history_tokens": meta["random_history_tokens"]}
                save(results_dir / f"{meta['row_index']:04d}_{arm}.json", record)
                out.write(json.dumps(record)+"\n")
                out.flush()
                values[arm] = value
                save(args.run / f"progress_{args.rank}.json", {"completed_questions": n,
                    "assigned_questions": len(selected), "last": record, "status": "running"})
                print(json.dumps(record), flush=True)
            if n < args.validate:
                checks = {}
                baseline = values.get("write_all")
                if baseline is None:
                    baseline = loss(arr, None)[0]
                ones = loss(arr, np.ones(meta["length"], dtype=np.float32))[0]
                repeat = loss(arr, None)[0]
                checks.update(all_ones_loss=ones, baseline=baseline,
                              ones_abs_diff=abs(ones-baseline), repeat_abs_diff=abs(repeat-baseline))
                checks["ones_pass"] = checks["ones_abs_diff"] <= 1e-5
                checks["state_reset_pass"] = checks["repeat_abs_diff"] <= 1e-5
                live_at_question = truncation_indices(meta,arr,args.window)
                none = make_mask(meta, "write_nothing", meta["length"])
                empty = values.get("write_nothing")
                if empty is None:
                    empty = loss(arr, none)[0]
                trunc_all = loss(arr, None, indices=live_at_question, reset_positions=True)[0]
                trunc_none = loss(arr, none, indices=live_at_question, reset_positions=True)[0]
                checks.update(write_nothing_loss=empty, truncation_loss=trunc_all,
                              truncation_no_history_write_loss=trunc_none,
                              truncation_abs_diff=abs(empty-trunc_all),
                              truncation_matched_mask_abs_diff=abs(empty-trunc_none))
                checks["truncation_tolerance_nats"] = .05
                checks["truncation_pass"] = abs(empty-trunc_all) <= .05
                save(args.run / f"validation_{args.rank}_{n}.json", {"id":meta["id"], **checks})
                if not checks["ones_pass"] or not checks["state_reset_pass"]:
                    raise AssertionError(f"mask/state validation failed: {checks}")
            save(args.run / f"progress_{args.rank}.json", {"completed_questions": n+1,
                "assigned_questions": len(selected), "last_id": meta["id"], "status":"running"})
    progress_name=f'control_progress_{args.rank}.json' if args.controls_only else f'progress_{args.rank}.json'
    save(args.run / progress_name, {"completed_questions":len(selected),
        "assigned_questions":len(selected), "status":"complete"})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["prepare", "run"])
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data", type=Path)
    p.add_argument("--window", type=int, default=32768)
    p.add_argument("--chunk", type=int, default=1024)
    p.add_argument("--seed", type=int, default=20261128)
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--world", type=int, default=1)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--validate", type=int, default=1)
    p.add_argument("--wait-for-data", action="store_true")
    p.add_argument("--controls-only", action="store_true")
    args = p.parse_args()
    for name in ["run", "repo", "checkpoint"]:
        setattr(args, name, getattr(args, name).resolve())
    for sub in ["raw", "data"]:
        (args.run / sub).mkdir(parents=True, exist_ok=True)
    (prepare if args.mode == "prepare" else run)(args)


if __name__ == "__main__":
    main()
