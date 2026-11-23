"""Where does rehearsal help or hurt? Rehearsal only acts on a write that lands on a non-empty memory, i.e. on every
written chunk except the first. With K = 2 the earlier written chunk is written plainly, the later one is written
together with the rehearsal pairs (question probe -> read-out of the memory before this write).
Split K2_attc (and F) questions by where the answer chunks sit among the written chunks, compare with / without
rehearsal; also check that questions with a single written chunk give identical answers (rehearsal never runs)."""
import glob, json
def load(n):
    return {(r["row"], r["query_index"]): r for r in json.load(open(f"out/full/{n}.judged.json"))["records"]
            if r["distance_tokens"] >= 2048}
SC = {}
for pat in ("out/strat/s*.json", "out/strat2/s*.json"):
    for f in glob.glob(pat):
        for r in json.load(open(f)):
            k = (r["row"], r["query_index"])
            if k in SC: SC[k]["scores"].update(r.get("scores", {}))
            else: SC[k] = r
def att_c(r):
    tot = [0.0] * r["n_chunks"]
    for l in range(9, 18):
        row = r["scores"]["att_c_max"][l]; z = sum(row) or 1.0
        for j, x in enumerate(row): tot[j] += x / z
    return tot
def report(title, X, Y, groups):
    print(title)
    for g, ks in groups.items():
        a = sum(X[k]["judge"] and not Y[k]["judge"] for k in ks); b = sum(Y[k]["judge"] and not X[k]["judge"] for k in ks)
        same = sum(X[k]["pred"] == Y[k]["pred"] for k in ks)
        print(f"  {g:48s} n={len(ks):4d}  with {sum(X[k]['judge'] for k in ks):4d}  without {sum(Y[k]['judge'] for k in ks):4d}"
              f"  only-with {a:3d} only-without {b:3d}  identical answers {same}")
X, Y = load("K2_attc_R16"), load("K2_attc"); G = {}
for k in sorted(set(X) & set(Y)):
    r = SC[(int(k[0]), int(k[1]))] if (int(k[0]), int(k[1])) in SC else SC[k]
    sc = att_c(r); top = sorted(sorted(range(r["n_chunks"]), key=lambda j: -sc[j])[:2]); ev = set(r["evidence"])
    if len(top) < 2: g = "one chunk only (no rehearsal)"
    else:
        e0, e1 = top[0] in ev, top[1] in ev
        g = {(False, True): "answer in the later chunk (written with rehearsal)", (True, False): "answer in the earlier chunk (protected by rehearsal)",
             (True, True): "answer in both", (False, False): "answer in neither"}[(e0, e1)]
    G.setdefault(g, []).append(k)
report("K2_attc: with vs without rehearsal", X, Y, G)
X, Y = load("F_evidence_R16"), load("F_evidence"); G = {}
for k in sorted(set(X) & set(Y)):
    n = X[k]["evidence_chunks"]; G.setdefault(f"{min(n, 3)}{'+' if n >= 3 else ''} answer chunk(s) written", []).append(k)
report("F_evidence: with vs without rehearsal", X, Y, dict(sorted(G.items())))

# Gated rehearsal for K = 2, computed exactly from the two runs: with K = 2 rehearsal only acts on the second write, so
# "rehearse or not" is a per-question choice between the *_R16 answer and the plain answer.
# Rule (label-free): rehearse only when the incoming (later) chunk scores lower than the chunk already in memory.
import os
for base, scf in [("K2_attc", att_c), ("K2_bm25", lambda r: r["scores"]["bm25"])]:
    if not os.path.exists(f"out/full/{base}_R16.judged.json"):
        continue
    X, Y = load(f"{base}_R16"), load(base); ks = sorted(set(X) & set(Y))
    gated = oracle = n_on = 0; G = {}
    for k in ks:
        r = SC[k]; sc = scf(r); top = sorted(sorted(range(r["n_chunks"]), key=lambda j: -sc[j])[:2])
        on = len(top) == 2 and sc[top[1]] < sc[top[0]]
        n_on += on; gated += (X if on else Y)[k]["judge"]
        oracle += X[k]["judge"] if not (len(top) == 2 and top[1] in set(r["evidence"])) else Y[k]["judge"]
        ev = set(r["evidence"])
        if len(top) == 2:
            G.setdefault({(False, True): "answer in the later chunk", (True, False): "answer in the earlier chunk",
                          (True, True): "answer in both", (False, False): "answer in neither"}[(top[0] in ev, top[1] in ev)], []).append(k)
    if base != "K2_attc":
        report(f"{base}: with vs without rehearsal", X, Y, G)
    print(f"\n{base}, gated rehearsal (rehearse only if the new chunk scores lower than the stored one): {gated}/{len(ks)}"
          f"  (rehearsal on for {n_on} questions)   plain {sum(Y[k]['judge'] for k in ks)}   always {sum(X[k]['judge'] for k in ks)}"
          f"   gate by label (off when the new chunk holds the answer) {oracle}")
