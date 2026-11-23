"""Replay vs no replay, split by how far the answer lies before the question (number of 2048 chunks between the
last answer session and the question), on the beyond-window questions. Also an exact two-sided sign test."""
import json, math, sys
def load(n):
    return {(r["row"], r["query_index"]): r for r in json.load(open(f"out/full/{n}.judged.json"))["records"]
            if r["distance_tokens"] >= 2048}
def sign_p(a, b):
    n, k = a + b, min(a, b)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
for x, y in [("C_write_R16", "C_write"), ("C_write_R16a05", "C_write"), ("K2_attc_R16", "K2_attc"), ("F_evidence_R16", "F_evidence")]:
    X, Y = load(x), load(y); ks = sorted(set(X) & set(Y))
    a = sum(X[k]["judge"] and not Y[k]["judge"] for k in ks); b = sum(Y[k]["judge"] and not X[k]["judge"] for k in ks)
    print(f"{x} vs {y}: {sum(X[k]['judge'] for k in ks)} vs {sum(Y[k]['judge'] for k in ks)}  only-first {a} only-second {b}  sign-test p={sign_p(a, b):.3f}")
    for lo, hi in [(2048, 6144), (6144, 12288), (12288, 10**9)]:
        kk = [k for k in ks if lo <= X[k]["distance_tokens"] < hi]
        print(f"   distance {lo//1024}K-{'' if hi > 10**8 else hi//1024}{'K' if hi < 10**8 else 'K+'}: n={len(kk):4d}  {sum(X[k]['judge'] for k in kk):4d} vs {sum(Y[k]['judge'] for k in kk):4d}")
