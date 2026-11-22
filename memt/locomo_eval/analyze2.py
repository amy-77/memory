"""Recall of chunk-selection strategies at a fixed budget K, merging the first scoring pass (out/strat) with the
improved question-aware scores (out/strat2). Same protocol as analyze_strategies.py: questions with evidence beyond
the window, dev = first 3 conversations (choose the layer group there), test = the other 7 (report).
usage: python analyze2.py 'out/strat/s*.json' 'out/strat2/s*.json'"""
import glob, json, random, sys

base = {}
for f in glob.glob(sys.argv[1]):
    for r in json.load(open(f)):
        base[(r["row"], r["query_index"])] = r
for f in glob.glob(sys.argv[2]):
    for r in json.load(open(f)):
        k = (r["row"], r["query_index"])
        if k in base:
            base[k]["scores"].update(r.get("scores", {}))
            base[k]["content_words"] = r.get("content_words")
recs = sorted(base.values(), key=lambda r: (r["row"], r["query_index"]))
convs = list(dict.fromkeys(r["conv"] for r in recs))
dev_convs = set(convs[:3])
recs = [r for r in recs if r["distance_tokens"] >= 2048 and r["evidence"] and r["n_chunks"] > 0
        and "mqk_c_top16z" in r["scores"]]
dev = [r for r in recs if r["conv"] in dev_convs]
test = [r for r in recs if r["conv"] not in dev_convs]
print(f"questions {len(recs)} (dev {len(dev)}, test {len(test)})")
L = len(recs[0]["scores"]["att_sum"])
GROUPS = {"all": range(L), "L0-8": range(0, 9), "L9-17": range(9, 18), "L18-26": range(18, 27), "L27-35": range(27, L)}


def layer_avg(r, key, layers):
    tot = [0.0] * r["n_chunks"]
    for l in layers:
        row = r["scores"][key][l]
        lo = min(row); row = [x - lo for x in row]; z = sum(row) or 1.0   # shift to >= 0, then a distribution
        for j, x in enumerate(row):
            tot[j] += x / z
    return tot


def rank(sc):
    return sorted(range(len(sc)), key=lambda j: -sc[j])


def evaluate(fn, rs, K, seed=None):
    cov = tot = hit = 0
    rng = random.Random(seed)
    for r in rs:
        order = rng.sample(range(r["n_chunks"]), r["n_chunks"]) if fn is None else rank(fn(r))
        top = set(order[:K]); ev = set(r["evidence"])
        cov += len(top & ev); tot += len(ev); hit += bool(top & ev)
    return cov / tot, hit / len(rs)


fams = ["att_sum", "att_max", "att_c_sum", "att_c_max", "att_c_top32", "att_c_sumnh", "mqk_max", "mqk_top8",
        "mqk_c_top1mean", "mqk_c_top16mean", "mqk_c_top100mean",
        "mqk_c_top1z", "mqk_c_top4z", "mqk_c_top16z", "mqk_c_top64z", "mqk_c_top100z", "mqk_all_top16z", "mqk_c_glob100"]
print(f"\n{'strategy (layer group chosen on dev by R@2)':44s} {'dev R@2':>8s} {'test R@1':>9s} {'test R@2':>9s} {'test R@3':>9s} {'test hit@2':>11s}")
rows = [("random", None), ("bm25", lambda r: r["scores"]["bm25"])]
for fam in fams:
    if fam not in recs[0]["scores"]:
        continue
    best = max(GROUPS, key=lambda g: evaluate((lambda r, fam=fam, g=g: layer_avg(r, fam, GROUPS[g])), dev, 2)[0])
    rows.append((f"{fam}[{best}]", (lambda r, fam=fam, g=best: layer_avg(r, fam, GROUPS[g]))))
for name, fn in rows:
    if fn is None:
        d = sum(evaluate(None, dev, 2, s)[0] for s in range(20)) / 20
        t = [[evaluate(None, test, K, s) for s in range(20)] for K in (1, 2, 3)]
        t = [(sum(x[0] for x in tt) / 20, sum(x[1] for x in tt) / 20) for tt in t]
    else:
        d = evaluate(fn, dev, 2)[0]; t = [evaluate(fn, test, K) for K in (1, 2, 3)]
    print(f"{name:44s} {d:8.3f} {t[0][0]:9.3f} {t[1][0]:9.3f} {t[2][0]:9.3f} {t[1][1]:11.3f}")
ex = next(r for r in recs if r.get("content_words"))
print("\nexample content words:", ex["content_words"])
