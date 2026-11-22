"""Judged accuracy on the questions that every run in <dir> has finished (evidence beyond 2048), plus paired counts."""
import json, glob, os, sys
D = sys.argv[1]
runs = {os.path.basename(f)[:-len(".judged.json")]: {(r["row"], r["query_index"]): r for r in json.load(open(f))["records"]}
        for f in sorted(glob.glob(D + "/*.judged.json"))}
common = set.intersection(*(set(v) for v in runs.values()))
ks = sorted(k for k in common if next(iter(runs.values()))[k]["distance_tokens"] >= 2048)
print(f"questions finished by all runs: {len(common)}; evidence beyond 2048: {len(ks)}")
cats = ["single_hop", "multi_hop", "temporal", "open_domain"]
print(f"{'run':12s} {'correct/total':>14s} " + " ".join(f"{c:>12s}" for c in cats))
for n, R in runs.items():
    line = f"{n:12s} {sum(R[k]['judge'] for k in ks):>6d}/{len(ks):<7d} "
    for c in cats:
        kc = [k for k in ks if R[k]["category"] == c]
        line += f"{sum(R[k]['judge'] for k in kc):>5d}/{len(kc):<6d} "
    print(line)
names = list(runs)
print("\npaired: questions only X gets right / only Y gets right")
for x, y in [("C_write", "B_last"), ("C_write", "W_wrong"), ("F_evidence", "C_write"), ("F_evidence", "G_random"),
             ("S_qk", "G_random"), ("S_qk", "F_evidence"), ("A_full", "C_write"),
             ("C_write_R16", "C_write"), ("S_qk_R16", "S_qk"), ("F_evidence_R16", "F_evidence"),
             ("K2_bm25", "K2_random"), ("K2_bm25", "C_write"), ("K2_bm25", "S_qk"), ("K2_bm25", "F_evidence"),
             ("K1_bm25", "K2_bm25"), ("K3_bm25", "K2_bm25"), ("K2_rrf", "K2_bm25"), ("K2_attmax", "K2_random"),
             ("K2_random", "C_write"), ("K2_attc", "K2_random"), ("K2_attc", "K2_bm25"), ("K2_attc", "S_qk"),
             ("K2_attc", "C_write")]:
    if x in runs and y in runs:
        a = sum(runs[x][k]["judge"] and not runs[y][k]["judge"] for k in ks)
        b = sum(runs[y][k]["judge"] and not runs[x][k]["judge"] for k in ks)
        print(f"  {x:15s} vs {y:11s}: only first right {a:4d}, only second right {b:4d}")
