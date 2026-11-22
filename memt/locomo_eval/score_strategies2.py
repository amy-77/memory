"""Improved question-aware chunk scores (second pass; same chunks and questions as score_strategies.py).

Content queries: only the tokens of the question's own words are used as queries -- the LoCoMo template
("Question:", "Use DATE of CONVERSATION to answer with an approximate date.", "Answer:") and stop words are dropped,
the same words BM25 ignores. Example: "When did Caroline go to the LGBTQ support group?" -> Caroline, go, LGBTQ,
support, group.

Saved per question and per layer (36 layers), for every candidate chunk:
  att_c_{sum,max,top32,sumnh}  question -> history attention of the frozen Qwen3-4B (full attention), summed over
                               heads and CONTENT question tokens only; pooled per chunk as before
  mqk_c_top{K}z     memory q.k, content queries: for every (head, query) the mean similarity of its K best keys
                    in the chunk (K = 1, 4, 16, 64; each query picks its own keys), z-scored ACROSS the question's
                    chunks, then summed over heads and queries -- a query adds to a chunk only where it matches
                    clearly better than in the other chunks
  mqk_c_top{K}mean  the same K-best means, simply averaged over heads and queries (no comparison across chunks)
  mqk_all_top16z    as mqk_c_top16z but with all question tokens (template included), to separate the two changes
  mqk_c_glob100     SnapKV-like vote: every (head, content query) takes its 100 best keys over the WHOLE history
                    (all candidate chunks together); a chunk's score is how many of those keys it holds, summed
                    over heads and queries
"""
import argparse, json, math, re, sys, time
from pathlib import Path

import torch
import torch.nn.functional as F
from einops import rearrange

sys.path.insert(0, "/data/qyl/memt_locomo_20261122")
import score_strategies as S0   # noqa: E402  (encode, qa_ids, STOP, model loading helpers, rope hook installed)
from lact_qwen.loading import add_model_args, load_model, load_teacher, autocast  # noqa: E402
from lact_qwen.streaming import begin_stream, forward_chunk, end_stream  # noqa: E402
from lact_qwen import memory_branch as MB  # noqa: E402
from lact_qwen.fast_weight import l2_norm  # noqa: E402
from transformers.models.qwen3 import modeling_qwen3 as mq  # noqa: E402

# ---- attention restricted to content query rows
CAP = {"on": False, "nq": 0, "sel": None, "mass": []}


def _rope(q, k, cos, sin, *a, **kw):
    q2, k2 = S0._orig_rope(q, k, cos, sin, *a, **kw)
    if CAP["on"]:
        nq, T = CAP["nq"], k2.shape[2]
        rows = torch.tensor(CAP["sel"], device=q2.device)
        qq = q2[:, :, T - nq:][:, :, rows].float()
        kk = k2.float().repeat_interleave(qq.shape[1] // k2.shape[1], dim=1)
        sc = (qq @ kk.transpose(-1, -2)) / math.sqrt(qq.shape[-1])
        pos = torch.arange(T, device=sc.device)
        sc = sc.masked_fill(pos[None, :] > (T - nq + rows)[:, None], float("-inf"))
        CAP["mass"].append(sc.softmax(-1).sum(dim=(0, 1, 2)))
    return q2, k2


mq.apply_rotary_pos_emb = _rope

# ---- memory q.k: capture question queries, then per chunk the best (and top-3) match of every (head, query)
KS = (1, 4, 16, 64, 100)
MEM = {"mode": None, "span": None, "q": {}, "topk": {}, "tv": {}}


def _read(self, x):
    h, d = self.cfg.num_heads, self.cfg.head_dim
    if MEM["mode"] == "q":
        s, e = MEM["span"]
        q = F.linear(x[:, s:e], self.qkv.weight[: h * d])
        MEM["q"][self.layer_idx] = l2_norm(rearrange(q.float(), "b l (h d) -> (b h) l d", h=h, d=d))
    elif MEM["mode"] == "k":
        k = F.linear(x, self.qkv.weight[h * d: 2 * h * d])
        k = l2_norm(rearrange(k.float(), "b l (h d) -> (b h) l d", h=h, d=d))
        sim = MEM["q"][self.layer_idx] @ k.transpose(1, 2)              # [h, nq, T]
        tv = sim.topk(min(max(KS), sim.shape[-1]), -1).values                 # [h, nq, 100], sorted
        MEM["topk"].setdefault(self.layer_idx, []).append(
            torch.stack([tv[..., :K].mean(-1) for K in KS]).cpu())              # [len(KS), h, nq]
        MEM["tv"].setdefault(self.layer_idx, []).append(tv.cpu())             # for the global top-100 vote
    return S0._orig_read(self, x)


MB.LaCTMemoryBranch._read_matrix = _read


def content_positions(qtext, tok, n_qq):
    """Indices (within the question tokens) of tokens that belong to a content word of the question body."""
    enc = tok(qtext, add_special_tokens=False, return_offsets_mapping=True)
    m = re.search(r"Question:\s*(.*?)\s*(Use DATE of CONVERSATION.*?)?\s*\nAnswer:\s*$", qtext, re.S)
    b0 = m.start(1)
    words = [(b0 + w.start(), b0 + w.end()) for w in re.finditer(r"[A-Za-z0-9']+", m.group(1))
             if w.group(0).lower().strip("'") not in S0.STOP and len(w.group(0)) > 1]
    sel = [i for i, (a, b) in enumerate(enc["offset_mapping"][:n_qq]) if any(a < we and b > ws for ws, we in words)]
    return sel, [qtext[a:b] for a, b in words]


def zvote(M):
    """M [chunks, h, nq] -> per chunk: sum over (h, nq) of the z-score across chunks."""
    if M.shape[0] < 2:
        return [0.0] * M.shape[0]
    z = (M - M.mean(0, keepdim=True)) / (M.std(0, keepdim=True) + 1e-6)
    return z.sum(dim=(1, 2)).tolist()


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
    model, cfg = load_model(args)
    scorer = load_teacher(args)
    dev = next(model.parameters()).device
    C = args.chunk
    rows = [json.loads(l) for l in open(args.data)]
    if args.limit:
        rows = rows[:args.limit]
    out, t0 = [], time.time()
    for ri, dlg in enumerate(rows):
        if ri % args.nshards != args.shard:
            continue
        hist, spans, heads = S0.encode(dlg, tok)
        for qi, q in enumerate(dlg["queries"]):
            qq, aa = S0.qa_ids(q, tok)
            P = len(hist) + len(qq)
            bounds = [(max(0, e - C), e) for e in range(P + args.reserve, 0, -C)][::-1]
            earlier = list(range(len(bounds) - 1))
            fs = bounds[-1][0]
            sel, cwords = content_positions(q["question"], tok, len(qq))
            rec = {"row": ri, "query_index": qi, "n_chunks": len(earlier), "content_words": cwords,
                   "n_content_tokens": len(sel), "scores": {}}
            if not earlier or not sel:
                out.append(rec); continue
            Sd = rec["scores"]
            hmask = torch.zeros(len(hist), dtype=torch.bool)
            for a_, b_ in heads:
                hmask[a_:b_] = True
            with torch.no_grad(), autocast(args):
                CAP.update(on=True, nq=len(qq), sel=sel, mass=[])
                scorer(input_ids=torch.tensor([hist + qq], device=dev), logits_to_keep=1)
                CAP["on"] = False
                M = torch.stack(CAP["mass"])[:, :len(hist)].float().cpu(); M[:, :4] = 0
                for name in ("sum", "max", "top32", "sumnh"):
                    Sd["att_c_" + name] = []
                for l in range(M.shape[0]):
                    row = {"sum": [], "max": [], "top32": [], "sumnh": []}
                    for j in earlier:
                        s_, e_ = bounds[j]; seg = M[l, s_:e_]
                        row["sum"].append(float(seg.sum())); row["max"].append(float(seg.max()))
                        row["top32"].append(float(seg.topk(min(32, len(seg))).values.mean()))
                        row["sumnh"].append(float(seg[~hmask[s_:e_]].sum()))
                    for name in row:
                        Sd["att_c_" + name].append(row[name])
                begin_stream(model, C)
                MEM.update(mode="q", span=(len(hist) - fs, P - fs), q={}, topk={}, tv={})
                forward_chunk(model, torch.tensor([hist[fs:] + qq], device=dev), fs, update=False)
                MEM["mode"] = "k"
                for j in earlier:
                    s_, e_ = bounds[j]
                    forward_chunk(model, torch.tensor([hist[s_:e_]], device=dev), s_, update=False)
                MEM["mode"] = None
                end_stream(model)
            for l in sorted(MEM["topk"]):
                T_ = torch.stack(MEM["topk"][l])                                   # [chunks, len(KS), h, nq_all]
                for i, K in enumerate(KS):
                    Sd.setdefault(f"mqk_c_top{K}z", []).append(zvote(T_[:, i][:, :, sel]))
                    Sd.setdefault(f"mqk_c_top{K}mean", []).append(T_[:, i][:, :, sel].mean(dim=(1, 2)).tolist())
                Sd.setdefault("mqk_all_top16z", []).append(zvote(T_[:, KS.index(16)]))
                # global top-100 per (head, content query): the union of the per-chunk top-100 holds it
                tvs = MEM["tv"][l]
                vals = torch.cat([t[:, sel] for t in tvs], dim=-1)                  # [h, nq_c, sum of per-chunk k]
                cid = torch.cat([torch.full((t.shape[-1],), j) for j, t in enumerate(tvs)])
                top = vals.topk(min(100, vals.shape[-1]), -1).indices               # [h, nq_c, 100]
                Sd.setdefault("mqk_c_glob100", []).append(
                    torch.bincount(cid[top].flatten(), minlength=len(tvs)).float().tolist())
            out.append(rec)
        if ri % 10 == args.shard:
            print(f"row {ri+1}/{len(rows)} questions {len(out)} {time.time()-t0:.0f}s", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"))
    print("DONE", len(out), f"{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
