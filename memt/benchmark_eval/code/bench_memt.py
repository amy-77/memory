"""MemT-1.0 memory benchmarks (report sec. 6.1 and appendix D.3) for the 4B MemT-style model.

For every benchmark an adapter (adapters.py) builds, per history, the two parts of the FULL-CONTEXT input of the
report's appendix D.3.1:  history part  +  query part.

  mode memory    reset the memory, write the history part in time order (2048-token chunks; attention sees only
                 the chunk being written, as in training), then answer from the query part alone, worded for memory
                 ("... stored in your memory", report D.3.1). All questions of a history share the written memory:
                 answering only reads it.
  mode question  empty memory, the query part alone                                    (lower bound)
  mode wrong     write the history part of ANOTHER history of the same benchmark, then answer (content vs generic)
  mode full      base Qwen3-4B without memory: history part + query part in one prompt, the history cut to its most
                 recent tokens so that the prompt fits --full-tokens (the report keeps the most recent suffix, at 256K)

Decoding as report D.3.2: temperature 0 unless the benchmark says otherwise, thinking off, benchmark-specific
maximum of new tokens. In memory modes the answer is decoded with a KV cache inside the current 2048-token segment;
when a segment is full it is written to memory and decoding continues in a new segment (the model's streaming rule).
"""
import argparse
import json
import os
import sys
import time
import zlib
from pathlib import Path

import torch

import adapters

CHUNK = 2048


# ------------------------------------------------------------------------------------------------ memory model
class MemoryModel:
    """The 4B model with matrix memory, driven chunk by chunk (lact_qwen.streaming)."""

    def __init__(self, model, device):
        from lact_qwen.streaming import begin_stream, end_stream, forward_chunk
        self.model, self.device = model, device
        self.begin, self.end, self.forward_chunk = begin_stream, end_stream, forward_chunk
        self.active = False

    def reset(self):
        """Empty memory."""
        if self.active:
            self.end(self.model)
        self.begin(self.model, CHUNK)
        self.active = True

    def write(self, ids, offset=0):
        """Write tokens into memory, CHUNK tokens per forward, in order. Returns the position after the last token."""
        for s in range(0, len(ids), CHUNK):
            chunk = ids[s:s + CHUNK]
            self.forward_chunk(self.model, torch.tensor([chunk], device=self.device), offset + s, update=True)
        return offset + len(ids)

    def _step(self, ids, pos, past):
        """Forward `ids` at positions pos.. that only READ the memory, with the KV cache of the current segment."""
        ctx = self.model.lact_ctx
        ctx.position_offset = pos
        ctx.set_segments([(0, len(ids))], [False])
        out = self.model(input_ids=torch.tensor([ids], device=self.device),
                         position_ids=torch.arange(pos, pos + len(ids), device=self.device)[None],
                         past_key_values=past, use_cache=True, logits_to_keep=1)
        return out.logits[0, -1].float(), out.past_key_values

    def generate(self, prompt_ids, offset, max_new, stop_ids, temperature=0.0, seed=0, stop_text=(), tok=None):
        """Answer from memory. The prompt and the answer live in segments of at most CHUNK tokens: a prompt longer than
        CHUNK has its leading full chunks written first; a full segment is written before decoding goes on."""
        lead = (len(prompt_ids) - 1) // CHUNK * CHUNK
        seg_start = self.write(prompt_ids[:lead], offset) if lead else offset
        seg = list(prompt_ids[lead:])
        logits, past = self._step(seg, seg_start, None)
        gen = torch.Generator(device="cpu").manual_seed(seed)
        out = []
        for _ in range(max_new):
            if temperature > 0:
                nxt = int(torch.multinomial(torch.softmax(logits.cpu() / temperature, -1), 1, generator=gen))
            else:
                nxt = int(logits.argmax())
            if nxt in stop_ids:
                break
            out.append(nxt)
            if len(out) == max_new:
                break
            if stop_text and (len(out) < 600 or len(out) % 16 == 0):
                text = adapters.strip_lead(tok.decode(out)) if len(out) < 600 else tok.decode(out[-40:])
                if any(s in text for s in stop_text):  # leading blank lines do not end the answer
                    break
            if len(seg) == CHUNK:                       # segment full: write it, start an empty one
                seg_start = self.write(seg, seg_start)
                seg, past = [], None
            seg.append(nxt)
            logits, past = self._step([nxt], seg_start + len(seg) - 1, past)
        return out


def generate_uncached(mm, prompt_ids, offset, max_new, stop_ids):
    """Reference decoder for the check: no KV cache, the whole segment is re-run for every token (prompt <= CHUNK)."""
    ids, out = list(prompt_ids), []
    for _ in range(max_new):
        nxt = int(mm.forward_chunk(mm.model, torch.tensor([ids], device=mm.device), offset, update=False)[0, -1].argmax())
        if nxt in stop_ids:
            break
        out.append(nxt)
        ids.append(nxt)
    return out


# ------------------------------------------------------------------------------------------------ main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bench", required=True, choices=list(adapters.BENCHMARKS))
    p.add_argument("--data", required=True, help="benchmark data directory or file (see adapters.py)")
    p.add_argument("--mode", required=True, choices=["memory", "question", "wrong", "full"])
    p.add_argument("--framework", default=os.environ.get("MEMT_FRAMEWORK", "../framework"))
    p.add_argument("--full-tokens", type=int, default=32768, help="mode full: maximum prompt length")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--nshards", type=int, default=1)
    p.add_argument("--format", default="plain", choices=["plain", "chat"], help="prompt format, see adapters.py")
    p.add_argument("--limit", type=int, default=0, help="only the first N questions of this shard (smoke test)")
    p.add_argument("--check-cache", type=int, default=0, help="also decode the first N questions without KV cache")
    p.add_argument("--out", required=True)
    args, rest = p.parse_known_args()
    sys.path.insert(0, args.framework)
    from lact_qwen.loading import add_model_args, autocast, load_model
    mp = argparse.ArgumentParser()
    add_model_args(mp)
    margs = mp.parse_args(rest)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(margs.model)
    model, cfg = load_model(margs)
    if args.mode == "full":
        assert cfg is None and not margs.window, "mode full: pass --memory none --window 0"
    else:
        assert cfg is not None and cfg.chunk_size == CHUNK
    device = next(model.parameters()).device
    stop_ids = {i for i in (tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>"),
                            tok.convert_tokens_to_ids("<|endoftext|>")) if isinstance(i, int) and i >= 0}
    mm = MemoryModel(model, device) if cfg is not None else None

    items = adapters.BENCHMARKS[args.bench](args.data, tok)
    mine = [i for i in range(len(items)) if i % args.nshards == args.shard]      # whole histories per shard
    jobs = [(i, j) for i in mine for j in range(len(items[i]["questions"]))]
    if args.limit:
        jobs = jobs[:args.limit]
    print(f"{args.bench} {args.mode}: {len(items)} histories, {sum(len(x['questions']) for x in items)} questions; "
          f"shard {args.shard}/{args.nshards}: {len(mine)} histories, {len(jobs)} questions", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():                                   # resume
        done = {json.loads(l)["qid"] for l in open(out_path)}
    fout = open(out_path, "a")
    t0, written_for, hist_cache = time.time(), None, {}

    def history_ids(i):
        if i not in hist_cache:
            hist_cache.clear()
            hist_cache[i] = tok(adapters.history_text(tok, items[i], args.format), add_special_tokens=False).input_ids
        return hist_cache[i]

    with torch.no_grad(), autocast(margs):
        for n, (i, j) in enumerate(jobs):
            item, q = items[i], items[i]["questions"][j]
            if q["qid"] in done:
                continue
            tq = time.time()
            if args.mode == "full":
                hist = history_ids(i)
                probe = adapters.render(tok, q, "", args.format)
                room = max(0, args.full_tokens - len(tok(probe, add_special_tokens=False).input_ids))
                kept = hist[-room:] if room else []
                prompt = tok(adapters.render(tok, q, tok.decode(kept), args.format), add_special_tokens=False).input_ids
                ids = torch.tensor([prompt], device=device)
                kw = dict(do_sample=q["temperature"] > 0, temperature=q["temperature"] or None, top_p=None, top_k=None)
                if args.format == "plain" and q.get("stop"):   # HF rejects an empty stop list
                    kw.update(stop_strings=q["stop"], tokenizer=tok)
                g = model.generate(ids, max_new_tokens=q["max_new"], use_cache=True, pad_token_id=tok.eos_token_id, **kw)
                gen = [t for t in g[0, len(prompt):].tolist()]
                gen = gen[:next((k for k, t in enumerate(gen) if t in stop_ids), len(gen))]
                written, hist_len = 0, len(kept)
            else:
                source = (i + 1) % len(items) if args.mode == "wrong" else i
                if args.mode == "question":
                    mm.reset()
                    written_for, written = None, 0
                elif written_for != source:                 # write once per history, reuse it for its questions
                    mm.reset()
                    mm.write(history_ids(source))
                    written_for = source
                written = 0 if args.mode == "question" else len(history_ids(source))
                prompt = tok(adapters.render(tok, q, None, args.format), add_special_tokens=False).input_ids
                stops = q.get("stop", []) if args.format == "plain" else []
                gen = mm.generate(prompt, written, q["max_new"], stop_ids, q["temperature"],
                                  seed=zlib.crc32(q["qid"].encode()), stop_text=stops, tok=tok)
                if n < args.check_cache and args.mode != "full":
                    ref = generate_uncached(mm, prompt, written, min(q["max_new"], 64), stop_ids)
                    first = next((k for k, (a, b) in enumerate(zip(ref, gen)) if a != b), None)
                    print(f"CHECK cached vs uncached for {q['qid']}: {len(ref)} tokens, first difference at "
                          f"{first if first is not None else 'none'}"
                          + (f" (uncached {tok.decode(ref[first:first + 8])!r} / cached {tok.decode(gen[first:first + 8])!r})"
                             if first is not None else ""), flush=True)
                hist_len = written
            raw = tok.decode(gen, skip_special_tokens=True)
            pred = adapters.strip_lead(raw) if args.format == "plain" else raw
            if args.format == "plain":                        # the answer ends at the first stop string
                for s in q.get("stop", []):
                    pred = pred.split(s)[0]
            pred = pred.strip()
            rec = {"bench": args.bench, "mode": args.mode, "item": item["id"], "qid": q["qid"], "category": q["category"],
                   "question": q["question"], "gold": q["gold"], "pred": pred, "history_tokens": len(history_ids(i)),
                   "used_history_tokens": hist_len, "prompt_tokens": len(prompt), "new_tokens": len(gen),
                   "hit_max": len(gen) >= q["max_new"], "format": args.format, "raw": raw[:4000],
                   "stop": q.get("stop", []) if args.format == "plain" else [], "seconds": round(time.time() - tq, 2), **q.get("meta", {})}
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            if n % 10 == 0:
                print(f"{n + 1}/{len(jobs)} {time.time() - t0:.0f}s  Q: {q['question'][:60]!r}  gold: {str(q['gold'])[:40]!r}  "
                      f"pred: {pred[:60]!r}", flush=True)
    if mm is not None and mm.active:
        mm.end(model)
    print("DONE", len(jobs), f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
