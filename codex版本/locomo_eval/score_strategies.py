"""Score every earlier 2048-token chunk of every LoCoMo question with label-free write strategies (no generation).

Same chunks as eval_locomo_memt.py (cut back from history + question + 64 reserved tokens; the last chunk is seen
by attention and is never a candidate). Per question we save, for every candidate chunk:

  question-agnostic (could run while streaming, before the question exists)
    recency      chunk index (later = higher)
    surprise     mean next-token loss of the memory model reading the chunk (chunk-local attention, empty memory)
  question-aware (the question is known before writing, SnapKV setting)
    bm25         Okapi BM25 of the question words against the chunk text (k1 1.5, b 0.75, chunks of this history)
    att_*        question -> history attention of the frozen Qwen3-4B with full attention, per layer:
                 softmax over all positions of each question token's query against all keys (after RoPE), summed
                 over heads and question tokens; first 4 tokens excluded. Pooled per chunk as
                 sum / max / top32 (mean of the 32 largest) / sumnh (sum without session-header tokens)
    mqk_*        the memory's own addressing: question memory queries (W~_Q, L2) . chunk memory keys (W~_K, L2),
                 per layer; max over the chunk's tokens, then mean over question tokens and heads (mqk_max), or
                 mean of each question token's 8 best keys in the chunk (mqk_top8)
plus the evidence chunks (labels, only for evaluation) and question metadata.
"""
import argparse, json, math, re, sys, time
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
from einops import rearrange

R = "/data/qyl/lact-insight/overleaf_6af80380/memory_research_20261119/frameworks/memt"
sys.path.insert(0, R)
from lact_qwen.loading import add_model_args, load_model, load_teacher, autocast  # noqa: E402
from lact_qwen.streaming import begin_stream, forward_chunk, end_stream  # noqa: E402
from lact_qwen import memory_branch as MB  # noqa: E402
from lact_qwen.fast_weight import l2_norm  # noqa: E402
from transformers.models.qwen3 import modeling_qwen3 as mq  # noqa: E402

TURN_HEADER = "\n\n[Turn {i}]\n"
STOP = set("a an the and or of to in on at for with is are was were be been do does did what when where who whom which "
           "why how that this these those it its by from as about into than then there their his her they them he she "
           "you your i my me we our use date conversation answer question approximate".split())

# ---- per-layer question -> history attention (full attention scorer)
_orig_rope = mq.apply_rotary_pos_emb
CAP = {"on": False, "nq": 0, "mass": []}


def _rope(q, k, cos, sin, *a, **kw):
    q2, k2 = _orig_rope(q, k, cos, sin, *a, **kw)
    if CAP["on"]:
        nq, T = CAP["nq"], k2.shape[2]
        qq = q2[:, :, T - nq:].float()
        kk = k2.float().repeat_interleave(qq.shape[1] // k2.shape[1], dim=1)
        sc = (qq @ kk.transpose(-1, -2)) / math.sqrt(qq.shape[-1])
        pos = torch.arange(T, device=sc.device)
        sc = sc.masked_fill(pos[None, :] > (T - nq + torch.arange(nq, device=sc.device))[:, None], float("-inf"))
        CAP["mass"].append(sc.softmax(-1).sum(dim=(0, 1, 2)))
    return q2, k2


mq.apply_rotary_pos_emb = _rope

# ---- memory addressing: capture question queries, then score chunk keys against them
_orig_read = MB.LaCTMemoryBranch._read_matrix
MEM = {"mode": None, "span": None, "q": {}, "out": {}}


def _read(self, x):
    h, d = self.cfg.num_heads, self.cfg.head_dim
    if MEM["mode"] == "q":
        s, e = MEM["span"]
        q = F.linear(x[:, s:e], self.qkv.weight[: h * d])
        MEM["q"][self.layer_idx] = l2_norm(rearrange(q.float(), "b l (h d) -> (b h) l d", h=h, d=d))
    elif MEM["mode"] == "k":
        k = F.linear(x, self.qkv.weight[h * d: 2 * h * d])
        k = l2_norm(rearrange(k.float(), "b l (h d) -> (b h) l d", h=h, d=d))
        sim = MEM["q"][self.layer_idx] @ k.transpose(1, 2)           # [h, nq, T]
        top = sim.topk(min(8, sim.shape[-1]), dim=-1).values
        MEM["out"][self.layer_idx] = (float(sim.max(-1).values.mean()), float(top.mean()))
    return _orig_read(self, x)


MB.LaCTMemoryBranch._read_matrix = _read


def encode(dlg, tok):
    ids, spans, heads = [], [], []
    for i, body in enumerate(dlg["turns"]):
        t = tok(TURN_HEADER.format(i=i + 1) + body, add_special_tokens=False).input_ids
        first_line = body.split("\n", 1)[0]
        nh = len(tok(TURN_HEADER.format(i=i + 1) + first_line, add_special_tokens=False).input_ids)
        spans.append((len(ids), len(ids) + len(t))); heads.append((len(ids), len(ids) + min(nh, len(t))))
        ids.extend(t)
    return ids, spans, heads


def qa_ids(q, tok):
    q_ids = tok(q["question"], add_special_tokens=False).input_ids
    a_ids = tok(" " + q["answer"], add_special_tokens=False).input_ids
    lead = 0
    while lead < len(a_ids) - 1 and not tok.decode(a_ids[lead:lead + 1]).strip():
        lead += 1
    return q_ids + a_ids[:lead], a_ids[lead:]


def words(s):
    return [w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in STOP]


def bm25(query, docs, k1=1.5, b=0.75):
    dw = [words(d) for d in docs]
    avg = sum(map(len, dw)) / max(1, len(dw))
    df = Counter(w for d in dw for w in set(d))
    N = len(dw)
    out = []
    for d in dw:
        tf = Counter(d); s = 0.0
        for w in set(words(query)):
            if w in tf:
                idf = math.log(1 + (N - df[w] + 0.5) / (df[w] + 0.5))
                s += idf * tf[w] * (k1 + 1) / (tf[w] + k1 * (1 - b + b * len(d) / avg))
        out.append(s)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--chunk", type=int, default=2048)
    p.add_argument("--reserve", type=int, default=64)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--nshards", type=int, default=1)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--out", required=True)
    add_model_args(p)
    args = p.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    model, cfg = load_model(args)                 # memory model (for surprise and memory addressing)
    scorer = load_teacher(args)                   # frozen base, full attention (for question attention)
    dev = next(model.parameters()).device
    C = args.chunk
    rows = [json.loads(l) for l in open(args.data)]
    if args.limit:
        rows = rows[:args.limit]
    out, t0 = [], time.time()
    for ri, dlg in enumerate(rows):
        if ri % args.nshards != args.shard:
            continue
        hist, spans, heads = encode(dlg, tok)
        for qi, q in enumerate(dlg["queries"]):
            qq, aa = qa_ids(q, tok)
            P = len(hist) + len(qq)
            bounds = [(max(0, e - C), e) for e in range(P + args.reserve, 0, -C)][::-1]
            earlier = list(range(len(bounds) - 1))
            fs = bounds[-1][0]
            ev = q.get("evidence_turns") or [q["fact_turn"]]
            evid = sorted(j for j in earlier if any(spans[t][0] < bounds[j][1] and spans[t][1] > bounds[j][0] for t in ev))
            rec = {"row": ri, "query_index": qi, "dialogue": dlg["id"], "conv": dlg["id"].split("#")[0],
                   "category": q["category"], "distance_tokens": len(hist) - spans[q["fact_turn"]][1],
                   "n_chunks": len(earlier), "evidence": evid, "scores": {}}
            if not earlier:
                out.append(rec); continue
            S = rec["scores"]
            S["recency"] = [float(j) for j in earlier]
            texts = [tok.decode(hist[bounds[j][0]:min(bounds[j][1], len(hist))]) for j in earlier]
            S["bm25"] = bm25(q["question"], texts)
            hmask = torch.zeros(len(hist), dtype=torch.bool)
            for a_, b_ in heads:
                hmask[a_:b_] = True
            with torch.no_grad(), autocast(args):
                # question -> history attention, per layer
                CAP.update(on=True, nq=len(qq), mass=[])
                scorer(input_ids=torch.tensor([hist + qq], device=dev), logits_to_keep=1)
                CAP["on"] = False
                M = torch.stack(CAP["mass"])[:, :len(hist)].float().cpu()      # [L, T]
                M[:, :4] = 0
                for name in ("sum", "max", "top32", "sumnh"):
                    S["att_" + name] = []
                for l in range(M.shape[0]):
                    row = {"sum": [], "max": [], "top32": [], "sumnh": []}
                    for j in earlier:
                        s_, e_ = bounds[j]
                        seg = M[l, s_:e_]
                        row["sum"].append(float(seg.sum())); row["max"].append(float(seg.max()))
                        row["top32"].append(float(seg.topk(min(32, len(seg))).values.mean()))
                        row["sumnh"].append(float(seg[~hmask[s_:e_]].sum()))
                    for name in row:
                        S["att_" + name].append(row[name])
                # memory addressing + surprise, chunk-local attention, empty memory (nothing is written)
                begin_stream(model, C)
                MEM.update(mode="q", span=(len(hist) - fs, P - fs), q={})
                forward_chunk(model, torch.tensor([hist[fs:] + qq], device=dev), fs, update=False)
                S["mqk_max"] = [[] for _ in range(cfg.layer_indices(36).__len__())]
                S["mqk_top8"] = [[] for _ in range(len(S["mqk_max"]))]
                S["surprise"] = []
                MEM["mode"] = "k"
                for j in earlier:
                    s_, e_ = bounds[j]
                    MEM["out"] = {}
                    lg = forward_chunk(model, torch.tensor([hist[s_:e_]], device=dev), s_, update=False)
                    tgt = torch.tensor(hist[s_ + 1:e_], device=dev)
                    S["surprise"].append(float(F.cross_entropy(lg[0, :len(tgt)].float(), tgt)))
                    for li, l in enumerate(sorted(MEM["out"])):
                        S["mqk_max"][li].append(MEM["out"][l][0]); S["mqk_top8"][li].append(MEM["out"][l][1])
                MEM["mode"] = None
                end_stream(model)
            out.append(rec)
        if ri % 10 == args.shard:
            print(f"row {ri+1}/{len(rows)} questions {len(out)} {time.time()-t0:.0f}s", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"))
    print("DONE", len(out), f"{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
