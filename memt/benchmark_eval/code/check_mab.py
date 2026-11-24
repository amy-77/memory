"""CPU checks for adapters_mab.py and score_mab.py (no GPU, no model).

usage: python check_mab.py <data root> <tokenizer dir> <tmp dir>
"""
import json
import os
import sys
import time
from collections import defaultdict

import adapters
import score_mab as S
from adapters_mab import load_mab

root, tok_dir, tmp = sys.argv[1:4]
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(tok_dir)

t0 = time.time()
items = load_mab(root, tok)
print(f"loaded {len(items)} histories, {sum(len(i['questions']) for i in items)} questions in {time.time() - t0:.0f}s")

# ---- memory-prompt lengths (prompts longer than 2048 tokens get their leading chunks written by bench_memt.py)
lens = defaultdict(list)
for it in items:
    for q in it["questions"]:
        lens[q["meta"]["family"]].append(len(tok(adapters.render(tok, q, None, "chat"), add_special_tokens=False).input_ids))
print("memory prompt tokens (chat) min/mean/max per family:",
      {f: (min(v), int(sum(v) / len(v)), max(v)) for f, v in lens.items()})
SKIP = os.environ.get("SKIP_TOKENS") == "1"

# ---- counts and history tokens per row
per_source = defaultdict(lambda: [0, 0, 0, 0])            # rows, questions, history tokens, chunks
for it in ([] if SKIP else items):
    s = it["questions"][0]["meta"]["source"]
    n_tok = len(tok(it["history"], add_special_tokens=False).input_ids)
    n_chunks = it["history"].count("<User> The following context is")
    per_source[s][0] += 1
    per_source[s][1] += len(it["questions"])
    per_source[s][2] += n_tok
    per_source[s][3] += n_chunks
    print(f"  row {it['id']:40s} questions {len(it['questions']):4d}  history tokens {n_tok:9d}  memorize chunks {n_chunks}")
tot = [0, 0, 0]
print(f"\n{'source':34s} {'rows':>4s} {'questions':>9s} {'history tok':>12s} {'2048-chunk writes':>17s}")
for s, (r, q, t, c) in sorted(per_source.items()):
    print(f"{s:34s} {r:4d} {q:9d} {t:12d} {t // 2048:17d}")
    tot[0] += q; tot[1] += t; tot[2] += t // 2048
print(f"{'TOTAL':34s} {len(items):4d} {tot[0]:9d} {tot[1]:12d} {tot[2]:17d}")

# ---- one rendered prompt per source (memory prompt, plain and chat); history head
seen = set()
for it in ([] if SKIP else items):
    q = it["questions"][0]
    s = q["meta"]["source"]
    fam = q["meta"]["family"]
    if fam in seen:
        continue
    seen.add(fam)
    plain = adapters.render(tok, q, None, "plain")
    chat = adapters.render(tok, q, None, "chat")
    cue_plain = plain[: -len("\nAnswer:")] + q["plain_cue"] if plain.endswith("\nAnswer:") and not \
        "\n\n".join(m["content"] for m in q["memory"]).rstrip().endswith("Answer:") else plain
    print(f"\n===== {fam} ({s}) max_new {q['max_new']} stop {q['stop']!r} plain_cue {q['plain_cue']!r} gold {q['gold'][:2]!r}")
    print("--- history head:", repr(it["history"][:260]))
    print("--- plain (current render), last 330 chars:", repr(plain[-330:]))
    if cue_plain != plain:
        print("--- plain with plain_cue honoured, last 120 chars:", repr(cue_plain[-120:]))
    print("--- chat, last 260 chars:", repr(chat[-260:]))

# ---- metrics on fake answers: gold -> max, garbage -> 0
qs = [dict(q, **q["meta"], bench="mab") for it in items for q in it["questions"]]
rec_gold, rec_bad = [], []
for q in qs:
    g = q["gold"]
    if q["family"] == "recsys":
        name_to_id = json.load(open(os.path.join(root, "MemoryAgentBench/entity2id.json")))
        id2name = {int(i): n for n, i in name_to_id.items()}
        good = "\n".join(f"{k + 1}. {id2name[int(x)].split('/')[-1].strip('<>').replace('_', ' ')}" for k, x in enumerate(g))
    elif q["family"] == "icl":
        good = g[0]
    else:
        good = g[0]
    rec_gold.append(dict(q, pred=good))
    rec_bad.append(dict(q, pred="zzqx flurble"))
recsys_n = 20                                                 # recsys matching is slow-ish: check a subset
sub = lambda recs: [r for r in recs if r["family"] != "recsys"] + [r for r in recs if r["family"] == "recsys"][:recsys_n]
matcher = S.MovieMatcher(os.path.join(root, "MemoryAgentBench/entity2id.json"))
for name, recs in (("gold answers", sub(rec_gold)), ("garbage answers", sub(rec_bad))):
    t1 = time.time()
    scores = {r["qid"]: S.rule_score(r, matcher) for r in recs if r["family"] not in S.LLM_FAMILIES}
    # fake judge outputs: gold -> "yes" / perfect summary scores; garbage -> "no" / zero
    good = name == "gold answers"
    outs = {}
    for rq in S.judge_requests([r for r in recs if r["family"] in S.LLM_FAMILIES]):
        r = next(x for x in recs if x["qid"] == rq["qid"])
        if rq["kind"] == "lme":
            outs[rq["key"]] = "yes" if good else "no"
        elif rq["kind"] == "fluency":
            outs[rq["key"]] = 'Output: {"fluency": %d}' % (1 if good else 0)
        elif rq["kind"] == "recall":
            outs[rq["key"]] = 'Output: {"supported_key_points": [], "recall": %d}' % (len(r["keypoints"]) if good else 0)
        else:
            outs[rq["key"]] = 'Output: {"precision": %d, "sentence_count": 10}' % (10 if good else 0)
    scores.update(S.judged_scores([r for r in recs if r["family"] in S.LLM_FAMILIES], outs))
    res = S.aggregate(recs, scores)
    print(f"\n##### {name} ({time.time() - t1:.0f}s scoring)")
    print(json.dumps(res, indent=1))

reqs = S.judge_requests(rec_gold)
kinds = defaultdict(int)
for r in reqs:
    kinds[r["kind"]] += 1
print("\njudge requests:", dict(kinds))
print("example LME judge prompt:", repr(next(r for r in reqs if r["kind"] == "lme")["messages"][0]["content"][-420:]))
print("example summary recall prompt tail:", repr(next(r for r in reqs if r["kind"] == "recall")["messages"][1]["content"][-300:]))
print("CHECK_DONE")
