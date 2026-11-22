"""Which aggregation of per-token attention best ranks the gold sessions? (LongMemEval subset, labels used only to evaluate)"""
import hashlib, json, os, sys
import numpy as np
W = "/data0/qyl/livemem_repro_20261121"
sys.path.insert(0, os.getcwd()); sys.path.insert(0, "tools/data_process")
from eval.materialize import materialize_requests
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(W + "/models/Qwen3-4B-Instruct-2507")
idx = {e["id"]: e for e in json.load(open(W + "/masks/attn_10_0/_index.json"))["index"]}
bm = {e["id"]: e for e in json.load(open(W + "/masks/bm25sub_10_0/_index.json"))["index"]}
aggs = {
    "sum": lambda a: a.sum(), "mean": lambda a: a.mean(), "max": lambda a: a.max(),
    "top8": lambda a: np.sort(a)[-8:].sum(), "top32": lambda a: np.sort(a)[-32:].sum(), "top128": lambda a: np.sort(a)[-128:].sum(),
    "sum/sqrt(len)": lambda a: a.sum() / np.sqrt(len(a)),
}
hit = {k: {5: 0, 10: 0} for k in aggs}; hit["bm25"] = {5: 0, 10: 0}; hit["attn_top32 + bm25 (rank sum)"] = {5: 0, 10: 0}
gold_n = 0; lens_g, lens_o = [], []
import math
from collections import Counter
for req in materialize_requests("longmemeval_s_sub", 10, None):
    g = set(idx[req.id]["gold"]);
    if not g: continue
    a = np.load(f"{W}/attn/tok/{hashlib.sha1(req.id.encode()).hexdigest()}.npy").astype(np.float32)
    dt = [tok(d, add_special_tokens=False).input_ids for d in req.memory_docs]
    L = [len(d) for d in dt]; off = np.cumsum([0] + L); assert off[-1] == len(a)
    gold_n += len(g)
    for i, l in enumerate(L): (lens_g if i in g else lens_o).append(l)
    ranks = {}
    for k, f in aggs.items():
        sc = [float(f(a[off[i]:off[i + 1]])) if L[i] else 0.0 for i in range(len(L))]
        order = sorted(range(len(L)), key=lambda i: -sc[i]); ranks[k] = {d: r for r, d in enumerate(order)}
        for n in (5, 10): hit[k][n] += len(g & set(order[:n]))
    q = set(tok(req.questions[0], add_special_tokens=False).input_ids); n_ = len(dt)
    df = Counter(t for d in dt for t in set(d)); avg = sum(L) / n_
    idf = {t: math.log(1 + (n_ - df[t] + 0.5) / (df[t] + 0.5)) for t in q}
    sc = []
    for d in dt:
        c = Counter(d); sc.append(sum(idf[t] * c[t] * 2.5 / (c[t] + 1.5 * (0.25 + 0.75 * len(d) / avg)) for t in q if t in c))
    order = sorted(range(n_), key=lambda i: -sc[i]); rb = {d: r for r, d in enumerate(order)}
    for n in (5, 10): hit["bm25"][n] += len(g & set(order[:n]))
    order = sorted(range(n_), key=lambda i: rb[i] + ranks["top32"][i])
    for n in (5, 10): hit["attn_top32 + bm25 (rank sum)"][n] += len(g & set(order[:n]))
print(f"gold sessions {gold_n}; mean tokens gold {np.mean(lens_g):.0f} others {np.mean(lens_o):.0f}")
for k, v in hit.items(): print(f"{k:32s} recall@5 {v[5] / gold_n:.3f}  recall@10 {v[10] / gold_n:.3f}")
