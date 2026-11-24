"""Merge shards and print beyond/inside-2048 tables plus paired log-prob differences against row B (last)."""
import json, glob, math, os, sys
from collections import defaultdict
D = sys.argv[1]; W = 2048
runs = {}
for name in sorted({os.path.basename(f).split(".s")[0] for f in glob.glob(D + "/*.s*.json")}):
    recs = []
    for f in sorted(glob.glob(f"{D}/{name}.s*.json")):
        recs += json.load(open(f))["records"]
    recs.sort(key=lambda r: (r["row"], r["query_index"]))
    json.dump({"records": recs}, open(f"{D}/{name}.json", "w"))
    runs[name] = recs
cats = ["ALL", "single_hop", "multi_hop", "temporal", "open_domain"]
for b in ("beyond", "inside"):
    print(f"\n### evidence {b} the 2048-token window: accuracy / mean answer log-prob (n)")
    print(f"{'run':16s}" + "".join(f"{c:>24s}" for c in cats))
    for name, recs in runs.items():
        g = defaultdict(list)
        for r in recs:
            if (r["distance_tokens"] >= W) == (b == "beyond"):
                g[r["category"]].append(r); g["ALL"].append(r)
        print(f"{name:16s}" + "".join(
            f"{sum(r['exact'] for r in g[c])/max(1,len(g[c])):>9.3f} {sum(r['logprob'] for r in g[c])/max(1,len(g[c])):>8.2f} ({len(g[c]):>4d})" for c in cats))
ref = [k for k in runs if k.startswith("B")]
if ref:
    B = runs[ref[0]]
    print(f"\n### paired log-prob difference vs {ref[0]}, evidence beyond 2048")
    for name, recs in runs.items():
        if len(recs) != len(B):
            print(f"{name:16s} incomplete ({len(recs)} vs {len(B)})"); continue
        assert all((a["row"], a["query_index"]) == (b["row"], b["query_index"]) for a, b in zip(recs, B))
        d = [a["logprob"] - b["logprob"] for a, b in zip(recs, B) if b["distance_tokens"] >= W]
        m = sum(d) / len(d); sd = math.sqrt(sum((x - m) ** 2 for x in d) / (len(d) - 1))
        print(f"{name:16s} mean {m:+7.2f}  se {sd/math.sqrt(len(d)):.2f}  frac(better) {sum(x>0 for x in d)/len(d):.2f}  n={len(d)}")
