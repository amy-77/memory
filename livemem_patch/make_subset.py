"""Balanced LongMemEval-s subset (20 questions per question type, fixed seed) + registry entries."""
import json, os, random, sys
ROOT = os.getcwd()
sys.path.insert(0, os.path.join(ROOT, "tools", "data_process"))
from common import read_parquet, write_parquet, PROC_DIR
rows = list(read_parquet(os.path.join(PROC_DIR, "longmemeval_s", "test_single.parquet")))
by = {}
for r in rows:
    m = r["meta"]; m = json.loads(m) if isinstance(m, str) else m
    by.setdefault(m.get("question_type"), []).append(r)
rng = random.Random(0); pick = []
for t in sorted(by):
    pick += rng.sample(by[t], min(20, len(by[t])))
keep = {r["id"] for r in pick}
out = [r for r in rows if r["id"] in keep]
write_parquet(os.path.join(PROC_DIR, "longmemeval_s", "test_sub120.parquet"), out)
print("subset", len(out), {t: sum(1 for r in pick if r in by[t]) for t in by})
p = "eval/datasets.py"; s = open(p).read()
if "longmemeval_s_sub" not in s:
    s = s.replace('_add("longmemeval_s", "longmemeval_s", "test_single", "single", "lm")\n',
                  '_add("longmemeval_s", "longmemeval_s", "test_single", "single", "lm")\n_add("longmemeval_s_sub", "longmemeval_s", "test_sub120", "single", "lm")\n')
    open(p, "w").write(s)
p = "eval/matrix.py"; s = open(p).read()
if "longmemeval_s_sub" not in s:
    s += '\n# balanced 120-question subset of LongMemEval-s (tools: sel/make_subset.py)\n_TIER_OF["longmemeval_s_sub"] = (STATE_32K, TRUNCATE_256K, TRUNCATE_32K)\n'
    open(p, "w").write(s)
print("registry ok")
