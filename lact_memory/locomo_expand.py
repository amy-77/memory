"""Expand the compact LoCoMo file (one line per conversation, all questions) into groups of --group questions."""
import json, sys
src, out, group = sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 4
n = 0
with open(out, "w") as fo:
    for line in open(src):
        d = json.loads(line)
        for g in range(0, len(d["queries"]), group):
            fo.write(json.dumps({"id": f"{d['id']}#g{g // group}", "turns": d["turns"], "queries": d["queries"][g : g + group]}) + "\n"); n += 1
print(n, "rows ->", out)
