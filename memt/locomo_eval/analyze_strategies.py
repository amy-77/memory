"""Evidence-chunk recall of label-free write strategies at a FIXED budget of K chunks per question.

usage: python analyze_strategies.py 'out/strat/*.json'
Questions: evidence beyond the 2048 window and at least one evidence chunk among the candidates.
Split: the first 3 LoCoMo conversations = dev (choose layers / pooling there), the other 7 = test (report).
recall@K = (evidence chunks among the K chosen) / (all evidence chunks);  hit@K = share of questions with >= 1.
Per-layer scores are turned into a distribution over the question's chunks per layer before averaging layers.
"""
import glob, json, random, sys
from collections import defaultdict

recs = []
for f in sorted(glob.glob(sys.argv[1])):
    recs += json.load(open(f))
convs = list(dict.fromkeys(r["conv"] for r in sorted(recs, key=lambda r: (r["row"], r["query_index"]))))
dev_convs = set(convs[:3])
recs = [r for r in recs if r["distance_tokens"] >= 2048 and r["evidence"] and r["n_chunks"] > 0]
print(f"questions: {len(recs)} (dev {sum(r['conv'] in dev_convs for r in recs)}, test {sum(r['conv'] not in dev_convs for r in recs)}); "
      f"mean candidate chunks {sum(r['n_chunks'] for r in recs)/len(recs):.1f}, mean evidence chunks {sum(len(r['evidence']) for r in recs)/len(recs):.2f}")

L = len(recs[0]["scores"]["att_sum"])
GROUPS = {"all": range(L), "L0-8": range(0, 9), "L9-17": range(9, 18), "L18-26": range(18, 27), "L27-35": range(27, L)}


def layer_avg(r, key, layers):
    tot = [0.0] * r["n_chunks"]
    for l in layers:
        row = r["scores"][key][l]
        z = sum(row) or 1.0
        lo = min(row)
        if key.startswith("mqk"):          # similarities can be negative: shift to >= 0 before normalising
            row = [x - lo for x in row]; z = sum(row) or 1.0
        for j, x in enumerate(row):
            tot[j] += x / z
    return tot


def strategies():
    S = {"random": None, "recency": lambda r: r["scores"]["recency"], "surprise": lambda r: r["scores"]["surprise"],
         "bm25": lambda r: r["scores"]["bm25"]}
    for key in ("att_sum", "att_max", "att_top32", "att_sumnh", "mqk_max", "mqk_top8"):
        for g, layers in GROUPS.items():
            S[f"{key}[{g}]"] = (lambda key, layers: (lambda r: layer_avg(r, key, layers)))(key, layers)
    return S


def rank(scores):
    return sorted(range(len(scores)), key=lambda j: -scores[j])


def evaluate(fn, rs, K, seed=0):
    hit = cov = tot = 0
    rng = random.Random(seed)
    for r in rs:
        n = r["n_chunks"]
        order = rng.sample(range(n), n) if fn is None else rank(fn(r))
        top = set(order[:K]); ev = set(r["evidence"])
        cov += len(top & ev); tot += len(ev); hit += bool(top & ev)
    return cov / tot, hit / len(rs)


S = strategies()
S["rrf(bm25,att_sumnh[all])"] = lambda r: [1 / (60 + a) + 1 / (60 + b) for a, b in zip(
    [rank(r["scores"]["bm25"]).index(j) for j in range(r["n_chunks"])],
    [rank(layer_avg(r, "att_sumnh", range(L))).index(j) for j in range(r["n_chunks"])])]
dev = [r for r in recs if r["conv"] in dev_convs]
test = [r for r in recs if r["conv"] not in dev_convs]
print(f"\n{'strategy':28s}" + "".join(f"  dev R@{k}  test R@{k}  test hit@{k}" for k in (1, 2, 3)))
res = {}
for name, fn in S.items():
    line = f"{name:28s}"
    for K in (1, 2, 3):
        if fn is None:
            d = [evaluate(None, dev, K, s) for s in range(20)]; t = [evaluate(None, test, K, s) for s in range(20)]
            dr = sum(x[0] for x in d) / 20; tr = sum(x[0] for x in t) / 20; th = sum(x[1] for x in t) / 20
        else:
            dr = evaluate(fn, dev, K)[0]; tr, th = evaluate(fn, test, K)
        res[(name, K)] = (dr, tr, th)
        line += f"  {dr:7.3f}  {tr:8.3f}  {th:10.3f}"
    print(line)
print("\nbest on dev (R@2) per family, with its test numbers:")
for fam in ("att_sum", "att_max", "att_top32", "att_sumnh", "mqk_max", "mqk_top8"):
    cands = [n for n in S if n.startswith(fam + "[")]
    best = max(cands, key=lambda n: res[(n, 2)][0])
    print(f"  {best:28s} dev R@2 {res[(best, 2)][0]:.3f}  test R@1/2/3 "
          + " / ".join(f"{res[(best, k)][1]:.3f}" for k in (1, 2, 3)))
