"""LoCoMo answer log-prob for the 4B MemT checkpoint (frozen Qwen3-4B + matrix memory, chunk-local attention).

Same token sequence as the 0.6B LaCT runs (sw_protect_20261122/lact_qwen/dialog.py build_query_batch):
history = concat over sessions of "\n\n[Turn i]\n" + session text; then question; then " " + answer.
One question at a time. The sequence (history + question + answer) is cut into chunks of 2048 counted from
the END, so the last chunk holds the question, the answer and the ~2K tokens of history right before them;
attention only sees the current chunk (as in training), older chunks only reach the model through memory.

modes
  full      base Qwen3-4B, no memory, full attention over the whole sequence          (row A)
  last      base Qwen3-4B, no memory, only the last chunk                               (row B)
  nowrite   memory attached, no chunk written; only the last chunk                      (row D)
  write     every chunk written, read-then-write, exactly the eval_streaming 'write'     (row C)
  evidence  only chunks overlapping an annotated evidence session are written           (row F, uses labels)
  random    same number of chunks as 'evidence', chosen at random among earlier chunks  (row G)
  wrong     memory filled with ANOTHER conversation: as many chunks as 'write', taken from the end of the
            history of the next LoCoMo conversation; then the true last chunk                    (row W)
            (if W helps as much as C, the gain is generic -- format / style of LoCoMo -- not this conversation's content)
  qk        same number of chunks as 'evidence', chosen by the question's attention     (row S)
            (SnapKV-style, question known before writing): the frozen Qwen3-4B runs once with full attention
            over history + question; for every layer and head, each question token's query is dotted with all
            history keys (after RoPE), softmax over positions, summed over layers, heads and question tokens;
            the first 4 tokens (attention sink) are excluded; a chunk's score is the sum over its tokens.
Chunk boundaries are counted back from (history + question + 64 reserved tokens), so scoring the gold answer
and greedy generation (--gen) use the same chunks; the last chunk only reads memory.
"""
import argparse, glob, json, math, random, sys, time, zlib
from pathlib import Path

import torch

R = "/data/qyl/lact-insight/overleaf_6af80380/memory_research_20261119/frameworks/memt"
sys.path.insert(0, R)
from lact_qwen.loading import add_model_args, load_model, autocast  # noqa: E402
from lact_qwen.streaming import begin_stream, forward_chunk, end_stream  # noqa: E402

TURN_HEADER = "\n\n[Turn {i}]\n"

# --- question -> history attention mass, captured inside the scoring forward (mode qk only)
from transformers.models.qwen3 import modeling_qwen3 as _mq  # noqa: E402
_orig_rope = _mq.apply_rotary_pos_emb
CAP = {"on": False, "nq": 0, "mass": None, "layers": 0}


def _rope_capture(q, k, cos, sin, *a, **kw):
    q2, k2 = _orig_rope(q, k, cos, sin, *a, **kw)
    if CAP["on"]:
        nq, T = CAP["nq"], k2.shape[2]
        qq = q2[:, :, T - nq:].float()                                   # [1, H, nq, d]
        kk = k2.float().repeat_interleave(qq.shape[1] // k2.shape[1], dim=1)  # GQA -> [1, H, T, d]
        sc = (qq @ kk.transpose(-1, -2)) / math.sqrt(qq.shape[-1])       # [1, H, nq, T]
        pos = torch.arange(T, device=sc.device)
        sc = sc.masked_fill(pos[None, :] > (T - nq + torch.arange(nq, device=sc.device))[:, None], float("-inf"))
        m = sc.softmax(-1).sum(dim=(0, 1, 2))
        CAP["mass"] = m if CAP["mass"] is None else CAP["mass"] + m
        CAP["layers"] += 1
    return q2, k2


_mq.apply_rotary_pos_emb = _rope_capture


def encode(dlg, tok):
    ids, spans = [], []
    for i, body in enumerate(dlg["turns"]):
        t = tok(TURN_HEADER.format(i=i + 1) + body, add_special_tokens=False).input_ids
        spans.append((len(ids), len(ids) + len(t)))
        ids.extend(t)
    return ids, spans


def qa_ids(q, tok):
    q_ids = tok(q["question"], add_special_tokens=False).input_ids
    a_ids = tok(" " + q["answer"], add_special_tokens=False).input_ids
    lead = 0
    while lead < len(a_ids) - 1 and not tok.decode(a_ids[lead:lead + 1]).strip():
        lead += 1
    return q_ids + a_ids[:lead], a_ids[lead:]


def score(logits, seq, a0, a1, offset):
    """logits[:, j] predicts token offset+j+1."""
    lp = torch.log_softmax(logits[0, a0 - 1 - offset:a1 - 1 - offset].float(), dim=-1)
    tgt = torch.tensor(seq[a0:a1], device=lp.device)
    return bool((lp.argmax(-1) == tgt).all()), float(lp.gather(-1, tgt[:, None]).sum())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--mode", required=True, choices=["full", "last", "nowrite", "write", "evidence", "random", "qk", "wrong", "scored"])
    p.add_argument("--chunk", type=int, default=2048)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--nshards", type=int, default=1)
    p.add_argument("--limit", type=int, default=0, help="only the first N rows (smoke test)")
    p.add_argument("--gen", type=int, default=24, help="greedy-decode up to N tokens after the question; 0 = off")
    p.add_argument("--reserve", type=int, default=64, help="tokens kept free at the end of the last chunk")
    p.add_argument("--scores", default=None, help="mode scored: glob of score_strategies.py outputs")
    p.add_argument("--strategy", default=None, help="mode scored: bm25 | att_max[L9-17] | rrf | random")
    p.add_argument("--k", type=int, default=2, help="mode scored: chunks written per question (fixed, label-free)")
    p.add_argument("--replay-k", type=int, default=0,
                   help="Robins-style rehearsal with the question's own memory queries: probes per head; 0 = off")
    p.add_argument("--replay-alpha", type=float, default=1.0, help="rehearsal strength (1 = restore the read-out)")
    p.add_argument("--replay-gate", default=None,
                   help="rehearse a write only if the incoming chunk scores lower (this scorer, needs --scores) than the "
                        "best chunk already written, i.e. never resist a chunk more relevant than the memory content")
    p.add_argument("--out", required=True)
    add_model_args(p)
    args = p.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    model, cfg = load_model(args)
    if args.mode in ("full", "last"):
        assert cfg is None, "full/last are base-model rows: pass --memory none"
    else:
        assert cfg is not None and cfg.chunk_size == args.chunk
    dev = next(model.parameters()).device
    R_ = None
    if args.replay_k:
        import replay_patch as R_
        R_.install(); R_.REPLAY["k"], R_.REPLAY["alpha"] = args.replay_k, args.replay_alpha
    scorer = None
    if args.mode == "qk":
        from lact_qwen.loading import load_teacher   # frozen base model, full attention, no memory
        scorer = load_teacher(args)
    C = args.chunk
    stop_ids = {i for i in (tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>"),
                            tok.convert_tokens_to_ids("<|endoftext|>")) if isinstance(i, int) and i >= 0}

    SC = {}
    if args.mode == "scored" or args.replay_gate:
        for pat in args.scores.split(","):            # first pattern = base scores; later ones add score keys
            for f in glob.glob(pat):
                for r_ in json.load(open(f)):
                    k_ = (r_["row"], r_["query_index"])
                    if k_ in SC:
                        SC[k_]["scores"].update(r_.get("scores", {}))
                    else:
                        SC[k_] = r_

    def layer_avg(r_, key, layers):
        tot = [0.0] * r_["n_chunks"]
        for l in layers:
            row = r_["scores"][key][l]; z = sum(row) or 1.0
            for j, x in enumerate(row):
                tot[j] += x / z
        return tot

    def strategy_scores(r_, name, ri, qi):
        if name == "bm25":
            return r_["scores"]["bm25"]
        if name == "att_max[L9-17]":
            return layer_avg(r_, "att_max", range(9, 18))
        if name == "att_c_max[L9-17]":           # question content words only (template and stop words removed)
            return layer_avg(r_, "att_c_max", range(9, 18))
        if name == "rrf":                 # reciprocal-rank fusion of bm25 and att_max[L9-17]
            out = [0.0] * r_["n_chunks"]
            for sc in (r_["scores"]["bm25"], layer_avg(r_, "att_max", range(9, 18))):
                for i, j in enumerate(sorted(range(len(sc)), key=lambda j: -sc[j])):
                    out[j] += 1 / (60 + i)
            return out
        if name == "random":
            rng = random.Random(zlib.crc32(f"{ri}#{qi}#fixedk".encode()))
            return [rng.random() for _ in range(r_["n_chunks"])]
        raise ValueError(name)

    rows = [json.loads(l) for l in open(args.data)]
    if args.limit:
        rows = rows[:args.limit]
    conv = lambda r: r["id"].split("#")[0]
    convs = list(dict.fromkeys(conv(r) for r in rows))
    donor_hist = {}
    if args.mode == "wrong":
        first = {}
        for r in rows:
            first.setdefault(conv(r), r)
        for i, c in enumerate(convs):
            donor_hist[c] = encode(first[convs[(i + 1) % len(convs)]], tok)[0]
    recs, t0, n_written, n_chunks, n_ev, n_hit = [], time.time(), 0, 0, 0, 0
    n_gate = [0, 0]   # writes onto a non-empty memory: rehearsed, total
    for ri, dlg in enumerate(rows):
        if ri % args.nshards != args.shard:
            continue
        hist, spans = encode(dlg, tok)
        for qi, q in enumerate(dlg["queries"]):
            qq, aa = qa_ids(q, tok)
            seq = hist + qq + aa
            P = len(hist) + len(qq)                    # prompt = history + question
            L, a0, a1 = len(seq), P, len(seq)
            R = max(args.reserve, len(aa), args.gen)   # a few gold answers are longer than 64 tokens
            # chunks of C counted back from P + R: the last chunk holds ~C - R tokens of history + question, and
            # the answer (scored) or the generated tokens fit in the R tokens left; earlier chunks are history only
            bounds = [(max(0, e - C), e) for e in range(P + R, 0, -C)][::-1]
            earlier = list(range(len(bounds) - 1))
            fs = bounds[-1][0]
            ev = q.get("evidence_turns") or [q["fact_turn"]]
            evs = [spans[t] for t in ev]
            evid = {j for j in earlier if any(s_ < bounds[j][1] and e_ > bounds[j][0] for s_, e_ in evs)}
            with torch.no_grad(), autocast(args):
                if args.mode == "full":
                    keep = L - a0 + 1
                    lg = model(input_ids=torch.tensor([seq], device=dev), logits_to_keep=keep).logits
                    ex, lpv = score(lg, seq, a0, a1, L - keep)
                    step = None
                    if args.gen:
                        g = model.generate(torch.tensor([seq[:P]], device=dev), max_new_tokens=args.gen, do_sample=False,
                                           temperature=None, top_p=None, top_k=None, use_cache=True,
                                           pad_token_id=tok.eos_token_id)
                        gen_ids = g[0, P:].tolist()
                else:
                    if args.mode == "last":
                        step = lambda ids: model(input_ids=torch.tensor([ids], device=dev),
                                                 position_ids=torch.arange(fs, fs + len(ids), device=dev)[None]).logits
                    else:
                        if args.mode == "nowrite":
                            flags = set()
                        elif args.mode in ("write", "wrong"):
                            flags = set(earlier)
                        elif args.mode == "evidence":
                            flags = set(evid)
                        elif args.mode == "scored":
                            r_ = SC[(ri, qi)]
                            assert r_["n_chunks"] == len(earlier) and r_["evidence"] == sorted(evid), (ri, qi)
                            sc = strategy_scores(r_, args.strategy, ri, qi) if earlier else []
                            flags = set(sorted(earlier, key=lambda j: -sc[j])[:args.k])
                        elif args.mode == "qk":
                            CAP.update(on=True, nq=len(qq), mass=None, layers=0)
                            scorer(input_ids=torch.tensor([hist + qq], device=dev), logits_to_keep=1)
                            CAP["on"] = False
                            assert CAP["layers"] == scorer.config.num_hidden_layers, CAP["layers"]
                            mass = CAP["mass"][:len(hist)].clone(); mass[:4] = 0
                            csc = {j: float(mass[bounds[j][0]:bounds[j][1]].sum()) for j in earlier}
                            flags = set(sorted(earlier, key=lambda j: -csc[j])[:len(evid)])
                        else:
                            rng = random.Random(zlib.crc32(f"{dlg['id']}#{qi}".encode()))
                            flags = set(rng.sample(earlier, len(evid)))
                        n_written += len(flags); n_chunks += len(earlier)
                        if args.mode in ("evidence", "random", "qk", "scored"):
                            n_ev += len(evid); n_hit += len(evid & flags)
                        if R_ is not None:
                            # probe queries = the question's own memory queries, taken from a forward of the chunk
                            # that holds the question with an empty memory (the question is known in advance)
                            begin_stream(model, C)
                            R_.REPLAY["q"], R_.REPLAY["captured"] = None, {}
                            R_.REPLAY["capture"] = (len(hist) - fs, P - fs)
                            forward_chunk(model, torch.tensor([seq[fs:P]], device=dev), fs, update=False)
                            R_.REPLAY["capture"] = None
                            R_.REPLAY["q"] = dict(R_.REPLAY["captured"])
                            end_stream(model)
                        begin_stream(model, C)
                        if args.mode == "wrong":
                            dh = donor_hist[conv(dlg)]
                            db = [(max(0, e - C), e) for e in range(len(dh), 0, -C)][::-1][-len(earlier):] if earlier else []
                            for s_, e_ in db:
                                forward_chunk(model, torch.tensor([dh[s_:e_]], device=dev), s_, update=True)
                            n_written += len(db) - len(flags)
                            flags = set()
                        gsc, best = None, None
                        if R_ is not None and args.replay_gate:
                            r_ = SC[(ri, qi)]
                            assert r_["n_chunks"] == len(earlier) and r_["evidence"] == sorted(evid), (ri, qi)
                            gsc = strategy_scores(r_, args.replay_gate, ri, qi) if earlier else []
                        for j in earlier:
                            if j in flags:
                                s_, e_ = bounds[j]
                                if gsc is not None:
                                    on = best is not None and gsc[j] < best
                                    R_.REPLAY["k"] = args.replay_k if on else 0
                                    if best is not None:
                                        n_gate[0] += on; n_gate[1] += 1
                                    best = gsc[j] if best is None else max(best, gsc[j])
                                forward_chunk(model, torch.tensor([seq[s_:e_]], device=dev), s_, update=True)
                        if R_ is not None:
                            R_.REPLAY["q"] = None; R_.REPLAY["k"] = args.replay_k
                        # the last chunk only reads: every token of a chunk reads the state left by the earlier
                        # chunks (memory_branch._read_matrix), so update=False gives the same output as in training
                        step = lambda ids: forward_chunk(model, torch.tensor([ids], device=dev), fs, update=False)
                    lg = step(seq[fs:a1])
                    ex, lpv = score(lg, seq, a0, a1, fs)
                    if args.gen:   # greedy, no KV cache: re-run the last chunk for every new token
                        cur, gen_ids = list(seq[fs:P]), []
                        for _ in range(args.gen):
                            nxt = int(step(cur)[0, -1].argmax())
                            gen_ids.append(nxt); cur.append(nxt)
                            if nxt in stop_ids or "\n" in tok.decode([nxt]):
                                break
                    if args.mode != "last":
                        end_stream(model)
            pred = None
            if args.gen:
                pred = tok.decode([t for t in gen_ids if t not in stop_ids]).split("\n")[0].strip()
            ft = q["fact_turn"]
            recs.append({"row": ri, "query_index": qi, "dialogue": dlg["id"], "category": q["category"],
                         "distance_tokens": len(hist) - spans[ft][1], "fact_turn": ft, "num_turns": len(spans),
                         "exact": ex, "logprob": lpv, "seq_len": L, "question": q["question"].strip(),
                         "answer": q["answer"], "pred": pred,
                         "evidence_chunks": len(evid), "evidence_chunks_written": None if args.mode in ("full", "last") else len(evid & flags) if args.mode in ("evidence", "random", "qk", "scored") else None})
        if len(recs) and ri % 20 == args.shard:
            print(f"row {ri+1}/{len(rows)} questions {len(recs)} {time.time()-t0:.0f}s", flush=True)
    summ = {"written_chunks": n_written, "earlier_chunks": n_chunks, "evidence_chunks": n_ev, "evidence_chunks_written": n_hit}
    if R_ is not None:
        st = R_.REPLAY["stats"]
        summ["replay"] = {"layer_writes": st["writes"], "layer_writes_with_rehearsal": st["replayed"],
                          "mean_relative_restore_error": st["restore_err"] / max(1, st["restore_n"])}
        if args.replay_gate:
            summ["replay"]["gate_rehearsed_writes"], summ["replay"]["gate_nonfirst_writes"] = n_gate
        print("replay", summ["replay"], flush=True)
    print("chunks written", n_written, "of", n_chunks, "earlier chunks; evidence chunks written", n_hit, "of", n_ev, flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"args": vars(args), "summary": summ, "records": recs}, open(args.out, "w"))
    print("DONE", len(recs), f"{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
