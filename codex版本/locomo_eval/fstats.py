import json, glob
from collections import Counter
w = e = 0; per = Counter(); n = 0
for f in sorted(glob.glob("out/full/F_evidence.s*.json")):
    d = json.load(open(f)); s = d["summary"]; w += s["written_chunks"]; e += s["earlier_chunks"]
    for r in d["records"]:
        if r["distance_tokens"] >= 2048:
            per[r["evidence_chunks"]] += 1; n += 1
print(f"all questions: chunks written {w} of {e} earlier chunks ({w/e:.1%})")
print("beyond-window questions, number of chunks written:", dict(sorted(per.items())), "n =", n)
print("mean chunks written (beyond):", sum(k * v for k, v in per.items()) / n)
rows = [json.loads(l) for l in open("data/locomo2_g4.jsonl")]
ev = Counter(len(q.get("evidence_turns") or [q["fact_turn"]]) for r in rows for q in r["queries"])
print("evidence sessions per question:", dict(sorted(ev.items())))
