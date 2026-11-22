"""Summarise LoCoMo runs: accuracy (teacher-forced exact match) and mean answer log-prob by category, split by
whether the LAST evidence session ends more than 2048 tokens before the question (attention cannot see it)."""
import json, glob, os, sys
from collections import defaultdict
W = 2048
rows = []
for f in sorted(glob.glob(sys.argv[1] + "/*.json")):
    recs = json.load(open(f))["records"]
    name = os.path.basename(f)[:-5]
    g = defaultdict(list)
    for r in recs:
        b = "beyond" if r["distance_tokens"] >= W else "inside"
        g[(r["category"], b)].append(r); g[("ALL", b)].append(r)
    rows.append((name, g))
cats = ["ALL", "single_hop", "multi_hop", "temporal", "open_domain"]
for b in ("beyond", "inside"):
    print(f"\n### evidence {b} the 2048-token window: accuracy / mean answer log-prob (n)")
    print(f"{'run':28s}" + "".join(f"{c:>24s}" for c in cats))
    for name, g in rows:
        line = f"{name:28s}"
        for c in cats:
            rs = g.get((c, b), [])
            line += f"{(sum(r['exact'] for r in rs) / len(rs) if rs else 0):>9.3f} {(sum(r['logprob'] for r in rs) / len(rs) if rs else 0):>8.2f} ({len(rs):>3d})"
        print(line)
