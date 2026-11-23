"""Per run, on the beyond-window questions: how often at least one answer chunk was written (hit), and accuracy on
hit / miss questions. Then, for pairs of runs, accuracy on the questions where both runs hit."""
import json, sys
runs = sys.argv[1:]
D = {}
for n in runs:
    recs = json.load(open(f"out/full/{n}.judged.json"))["records"]
    D[n] = {(r["row"], r["query_index"]): r for r in recs if r["distance_tokens"] >= 2048}
    for r in D[n].values():   # older runs (C, W, A, B) did not store the count: C writes every chunk, so all hit
        if r.get("evidence_chunks_written") is None:
            r["evidence_chunks_written"] = r["evidence_chunks"] if n.startswith("C_") else 0
keys = set.intersection(*(set(d) for d in D.values()))
print(f"questions {len(keys)}")
print(f"{'run':12s} {'hit':>6s} {'acc|hit':>12s} {'acc|miss':>12s} {'total':>6s}")
for n in runs:
    d = D[n]; h = [k for k in keys if d[k]["evidence_chunks_written"] > 0]; m = [k for k in keys if k not in set(h)]
    ah = sum(d[k]["judge"] for k in h); am = sum(d[k]["judge"] for k in m)
    print(f"{n:12s} {len(h):6d} {ah:5d}/{len(h):<4d}{ah/max(len(h),1):.2f} {am:5d}/{len(m):<4d}{am/max(len(m),1):.2f} {ah+am:6d}")
for a, b in [("K2_attc", "K2_bm25"), ("K1_bm25", "K2_bm25"), ("K2_bm25", "F_evidence"), ("K1_bm25", "F_evidence")]:
    if a in D and b in D:
        both = [k for k in keys if D[a][k]["evidence_chunks_written"] > 0 and D[b][k]["evidence_chunks_written"] > 0]
        print(f"both hit {a} vs {b}: n={len(both)}  {sum(D[a][k]['judge'] for k in both)} vs {sum(D[b][k]['judge'] for k in both)}")
