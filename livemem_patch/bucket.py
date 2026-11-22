"""LongMemEval subset: accuracy split by where the evidence lies relative to the 32K attention window."""
import json, os, numpy as np
W = "/data0/qyl/livemem_repro_20261121"; R = W + "/LiveMem/results/eval_sub"; WIN = 32768
idx = json.load(open(W + "/masks/bm25sub_10_0/_index.json"))["index"]
info = {}
for e in idx:
    m = np.load(f"{W}/masks/oracle_0/{e['key']}.npy").astype(np.float32)
    z = np.flatnonzero(m == 0)
    if len(z) == 0 or not e["gold"]:
        info[e["id"]] = ("no evidence label", 0, len(m)); continue
    ev = np.flatnonzero(m[z[0]:z[-1] + 1] == 1) + z[0]        # evidence tokens (ones between the first and last zero)
    tail = np.flatnonzero(m[z[-1] + 1:] == 1)                  # ones after the last zero: last session(s) if gold + question
    cut = len(m) - WIN
    inwin = float((ev >= cut).mean()) if len(ev) else 1.0
    b = "evidence all outside window" if inwin == 0 and len(ev) else ("evidence all inside window" if inwin == 1 else "evidence partly inside")
    info[e["id"]] = (b, len(ev), len(m))
def load(run):
    p = f"{R}/{run}/longmemeval_s_sub/records.jsonl"
    if not os.path.exists(p): return None
    return {json.loads(l)["id"]: json.loads(l)["items"][0]["correct"] for l in open(p)}
runs = [("full context (sft)", "sft-ones/truncate-256k"), ("truncate 32k (sft)", "sft-ones/truncate-32k"), ("full context (rl)", "rl-ones/truncate-256k"),
        ("truncate 32k (rl)", "rl-ones/truncate-32k"), ("sel40 memory, write all", "sel40-ones/state-32k"), ("sel40 memory, write evidence only", "sel40-oracle_0/state-32k"),
        ("sel40 truncate 32k", "sel40-ones/truncate-32k"), ("sel40 full context", "sel40-ones/truncate-256k")]
bs = ["evidence all outside window", "evidence partly inside", "evidence all inside window", "no evidence label"]
print("prompt tokens: mean %.0f min %d max %d" % (np.mean([v[2] for v in info.values()]), min(v[2] for v in info.values()), max(v[2] for v in info.values())))
print("evidence tokens per question: mean %.0f" % np.mean([v[1] for v in info.values() if v[1]]))
print("%-36s" % "run" + "".join("%30s" % b for b in bs) + "   all")
for name, run in runs:
    d = load(run)
    if d is None: print("%-36s (not finished)" % name); continue
    row = ""
    for b in bs:
        ids = [i for i in d if info[i.split("#")[0] if i not in info else i][0] == b]
        row += "%30s" % (f"{sum(d[i] for i in ids)}/{len(ids)} = {sum(d[i] for i in ids) / max(len(ids), 1):.2f}")
    print("%-36s" % name + row + "   %.3f" % (sum(d.values()) / len(d)))
a, o = load("sel40-ones/state-32k"), load("sel40-oracle_0/state-32k"); t = load("sft-ones/truncate-32k")
print("write-all vs evidence-only: both right %d, only all %d, only evidence %d, both wrong %d" % (
    sum(a[i] and o[i] for i in a), sum(a[i] and not o[i] for i in a), sum(o[i] and not a[i] for i in a), sum(not a[i] and not o[i] for i in a)))
g = lambda run: {json.loads(l)["id"]: json.loads(l)["raw"] for l in open(f"{R}/{run}/longmemeval_s_sub/records.jsonl")}
ga, go, gt = g("sel40-ones/state-32k"), g("sel40-oracle_0/state-32k"), g("sft-ones/truncate-32k")
print("identical outputs: write-all vs evidence-only %d/%d ; write-all(sel40) vs truncation(sft) %d/%d" % (sum(ga[i] == go[i] for i in ga), len(ga), sum(ga[i] == gt[i] for i in ga), len(ga)))
