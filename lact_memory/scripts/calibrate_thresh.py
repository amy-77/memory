"""Pick a streaming write threshold on a DEV set (never on the test set).

Input: the --dump-turn-scores file of a dev run. The threshold is the q-quantile of the scores of the turns that
state the asked fact (retention questions), so that about (1 - q) of the facts would be written; also printed:
the share of all turns that would be written at that threshold.
"""
import json, sys
path, q = sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 0.05
rows = [json.loads(l) for l in open(path)]
rows = [r for r in rows if r["category"] == "retention"]
fact = sorted(r["scores"][r["fact_turn"]] for r in rows)
seen = {}
for r in rows:
    seen[r["dialogue"], r["query_index"]] = r["scores"]
for qq in (0.01, 0.02, 0.05, 0.10, 0.20):
    th = fact[int(qq * (len(fact) - 1))]
    share = sum(sum(s >= th for s in sc) for sc in seen.values()) / sum(len(sc) for sc in seen.values())
    print(f"q={qq:.2f}  threshold {th:.6g}  fact turns kept {sum(f >= th for f in fact) / len(fact):.3f}  share of turns written {share:.3f}")
th = fact[int(q * (len(fact) - 1))]
print(f"CHOSEN {th:.8g}")
