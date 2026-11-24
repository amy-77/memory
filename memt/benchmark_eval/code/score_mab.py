"""Official MemoryAgentBench metrics for bench_memt.py answers (main set of adapters_mab.py).

Metric per source (MAB README "Clarification on Evaluation Metrics" + utils/eval_other_utils.py post_process):
  ruler_qa1/2, factconsolidation_sh/mh   substring_exact_match, max over (raw answer, parse_output(answer))  [default]
  eventqa_full                           substring_exact_match of parse_output(answer)                     [_process_eventqa]
  icl_* (5 sources)                      exact_match of parse_output(answer)                               [_process_icl]
  detective_qa                           exact_match, max over (raw, parsed)                               [default]
  recsys_redial_full                     Recall@5: answer list -> nearest movie names (edit distance) vs gold ids
  longmemeval_s*                         LLM judge, yes/no (longmem_qa_evaluate.get_anscheck_prompt; gpt-4o in MAB)
  infbench_sum_eng_shots2                LLM judge F1 = fluency * 2PR/(P+R) from 3 calls (summarization_evaluate.py,
                                         gpt-4o-2024-05-13 at temperature 0.1 in MAB); answers whose judge output
                                         cannot be parsed are left out of the mean, as in MAB
Aggregation (MAB paper): score of a source = mean over its questions (x100); category (AR, TTL, LRU, CR) = mean of its
sources, every source weighted equally (the 5 ICL sources count as 5 TTL sources; --icl-as-one averages them first);
overall = mean of the 4 categories.

The answer scored is the record's "pred" (bench_memt.py's answer after its stop strings; MAB scores the raw output,
and its parse_output keeps only the first line anyway).

Judge interface (for judge_bench.py):
  judge_requests(records) -> [{"key", "qid", "kind", "messages", "temperature", "max_tokens"}]
  judged_scores(records, outputs) -> {qid: score or None}, outputs = {key: judge text}
CLI:
  python score_mab.py requests <answers.jsonl ...> --out requests.jsonl
  python score_mab.py score <answers.jsonl ...> --entity2id <MemoryAgentBench/entity2id.json> [--judged outputs.jsonl]
                      [--icl-as-one] [--json scores.json]        (outputs.jsonl: lines {"key", "text"})
"""
import argparse
import json
from collections import defaultdict

import numpy as np

import mab_official as M

FAMILY_CATEGORY = {"ruler_qa": "AR", "eventqa": "AR", "longmemeval": "AR", "icl": "TTL", "recsys": "TTL",
                   "infbench_sum": "LRU", "detective_qa": "LRU", "factconsolidation": "CR"}
LLM_FAMILIES = {"longmemeval", "infbench_sum"}
SUMMARY_JUDGE_SYSTEM = "You are a helpful assistant."          # MAB OpenAIModel default system message


def load_records(paths):
    recs = {}
    for p in paths:
        for line in open(p):
            r = json.loads(line)
            recs[r["qid"]] = r
    return list(recs.values())


def _gold(r):
    g = r["gold"]
    return json.loads(g) if isinstance(g, str) and g.startswith("[") else ([g] if isinstance(g, str) else g)


def _max_raw_parsed(metric, pred, gold):
    """MAB default_post_process: the better of the raw output and parse_output(output)."""
    score = M.drqa_metric_max_over_ground_truths(metric, pred, gold)
    parsed = M.parse_output(pred)
    if parsed is not None:
        score = max(score, M.drqa_metric_max_over_ground_truths(metric, parsed, gold))
    return float(score)


# ------------------------------------------------------------------------------------------------ recsys
class MovieMatcher:
    """MAB's find_nearest_movie with a vectorised Levenshtein distance (editdistance is not installed here).
    Lower-cased like MAB. Ties: MAB takes the first of list(set(...)), whose order is hash-random; we take the first
    in sorted order, so results are deterministic."""

    def __init__(self, entity2id_path):
        name_to_id = json.load(open(entity2id_path))
        self.id_to_name = {int(i): M.extract_movie_name(name) for name, i in name_to_id.items()}
        self.candidates = sorted(set(self.id_to_name.values()))
        lowered = [c.lower() for c in self.candidates]           # lower() can change the length (e.g. "İ")
        width = max(len(c) for c in lowered)
        self.codes = np.zeros((len(lowered), width), dtype=np.int32)
        self.lens = np.array([len(c) for c in lowered])
        for k, c in enumerate(lowered):
            self.codes[k, :len(c)] = [ord(ch) for ch in c]
        self.exact = {}                                                 # lowered name -> first candidate (sorted)
        for c, low in zip(self.candidates, lowered):
            self.exact.setdefault(low, c)
        self.cache = {}

    def nearest(self, name):
        """Same result as the full search below: an exact (lower-cased) match has distance 0, and among equal
        distances the first candidate in sorted order wins in both paths. Results are cached per name."""
        s = name.lower()
        if s in self.exact:
            return self.exact[s]
        if s not in self.cache:
            self.cache[s] = self._search(s)
        return self.cache[s]

    def _search(self, s):
        n, width = len(self.candidates), self.codes.shape[1]
        prev = np.tile(np.arange(width + 1), (n, 1))                    # distance("" , cand[:j])
        for i, ch in enumerate(s, 1):
            cur = np.empty_like(prev)
            cur[:, 0] = i
            sub = prev[:, :-1] + (self.codes != ord(ch))
            dele = prev[:, 1:] + 1
            best = np.minimum(sub, dele)
            for j in range(1, width + 1):                               # insertion is a running min along j
                cur[:, j] = np.minimum(best[:, j - 1], cur[:, j - 1] + 1)
            prev = cur
        dist = prev[np.arange(n), self.lens]
        return self.candidates[int(np.argmin(dist))]

    def recall_at(self, pred, gold_ids, k=5):
        items, _ = M.extract_recommendation_list(pred)                  # raw items; matching done here
        predicted = [self.nearest(it) for it in items[:k]]
        gt = [self.id_to_name[int(x.strip())] for x in gold_ids]
        return sum(m in predicted[:k] for m in gt) / len(gt)


# ------------------------------------------------------------------------------------------------ rule-based
def rule_score(r, matcher=None):
    fam, pred, gold = r["family"], r["pred"] or "", _gold(r)
    if fam in ("ruler_qa", "factconsolidation"):
        return _max_raw_parsed(M.substring_exact_match_score, pred, gold)
    if fam == "detective_qa":
        return _max_raw_parsed(M.drqa_exact_match_score, pred, gold)
    if fam == "eventqa":
        return float(M.drqa_metric_max_over_ground_truths(M.substring_exact_match_score, M.parse_output(pred), gold))
    if fam == "icl":
        return float(M.drqa_metric_max_over_ground_truths(M.drqa_exact_match_score, M.parse_output(pred), gold))
    if fam == "recsys":
        return matcher.recall_at(pred, gold, 5)
    raise KeyError(fam)


# ------------------------------------------------------------------------------------------------ LLM judge
def judge_requests(records):
    reqs = []
    for r in records:
        fam, pred, gold = r["family"], (r["pred"] or "").strip(), _gold(r)
        if fam == "longmemeval":
            # MAB passes the reference answer list itself to .format(), so the prompt shows e.g. "['50']"
            prompt = M.get_anscheck_prompt(r["question_type"], r["question"], gold, pred,
                                           abstention="_abs" in r["question_id"])
            reqs.append({"key": f"{r['qid']}|lme", "qid": r["qid"], "kind": "lme", "temperature": 0, "max_tokens": 10,
                         "messages": [{"role": "user", "content": prompt}]})
        elif fam == "infbench_sum":
            kp = "\n".join(f"{i + 1}. {k}" for i, k in enumerate(r["keypoints"]))
            prompts = {"fluency": M.fluency_prompt_book.format(text=pred),
                       "recall": M.recall_prompt_book.format(keypoints=kp, summary=pred),
                       "precision": M.precision_prompt_book.format(expert_summary=gold[0], summary=pred)}
            for kind, prompt in prompts.items():
                reqs.append({"key": f"{r['qid']}|{kind}", "qid": r["qid"], "kind": kind, "temperature": 0.1,
                             "max_tokens": 4096, "messages": [{"role": "system", "content": SUMMARY_JUDGE_SYSTEM},
                                                              {"role": "user", "content": prompt}]})
    return reqs


def judged_scores(records, outputs):
    scores = {}
    for r in records:
        fam, q = r["family"], r["qid"]
        if fam == "longmemeval":
            t = outputs.get(f"{q}|lme")
            scores[q] = None if t is None else float("yes" in t.lower())
        elif fam == "infbench_sum":
            got = [outputs.get(f"{q}|{k}") for k in ("fluency", "recall", "precision")]
            parsed = [M.parse_json(t) if t is not None else None for t in got]
            if any(p is None for p in parsed):
                scores[q] = None
                continue
            f, rc, pc = parsed
            try:
                rec = rc["recall"] / len(r["keypoints"]) if len(r["keypoints"]) > 0 else 0
                prec = pc["precision"] / pc["sentence_count"] if pc["sentence_count"] > 0 else 0
                scores[q] = f["fluency"] * 2 * (rec * prec) / (rec + prec) if rec + prec > 0 else 0.0
            except (KeyError, TypeError, ZeroDivisionError):
                scores[q] = None
    return scores


# ------------------------------------------------------------------------------------------------ aggregate
def aggregate(records, scores, icl_as_one=False):
    per_source = defaultdict(list)
    missing = defaultdict(int)
    for r in records:
        s = scores.get(r["qid"])
        if s is None:
            missing[r["source"]] += 1
        else:
            per_source[r["source"]].append(s)
    src = {s: 100 * float(np.mean(v)) for s, v in per_source.items()}
    groups = defaultdict(list)
    for s, v in src.items():
        fam = next(r["family"] for r in records if r["source"] == s)
        groups[FAMILY_CATEGORY[fam]].append((s, v))
    cat = {}
    for c, lst in groups.items():
        if icl_as_one and c == "TTL":
            icl = [v for s, v in lst if s.startswith("icl")]
            rest = [v for s, v in lst if not s.startswith("icl")]
            vals = ([float(np.mean(icl))] if icl else []) + rest
        else:
            vals = [v for _, v in lst]
        cat[c] = float(np.mean(vals))
    overall = float(np.mean([cat[c] for c in ("AR", "TTL", "LRU", "CR") if c in cat])) if cat else None
    return {"per_source": {s: {"score": round(v, 2), "n": len(per_source[s]), "unscored": missing.get(s, 0)}
                           for s, v in sorted(src.items())},
            "per_category": {c: round(v, 2) for c, v in sorted(cat.items())},
            "overall": None if overall is None else round(overall, 2),
            "complete": all(c in cat for c in ("AR", "TTL", "LRU", "CR"))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["requests", "score"])
    ap.add_argument("answers", nargs="+")
    ap.add_argument("--out")
    ap.add_argument("--entity2id")
    ap.add_argument("--judged", help="jsonl of {key, text} judge outputs")
    ap.add_argument("--icl-as-one", action="store_true")
    ap.add_argument("--json")
    a = ap.parse_args()
    recs = load_records(a.answers)
    if a.cmd == "requests":
        reqs = judge_requests(recs)
        with open(a.out, "w") as f:
            for q in reqs:
                f.write(json.dumps(q, ensure_ascii=False) + "\n")
        print(f"{len(reqs)} judge requests for {len({q['qid'] for q in reqs})} answers -> {a.out}")
        return
    matcher = MovieMatcher(a.entity2id) if any(r["family"] == "recsys" for r in recs) else None
    scores = {r["qid"]: rule_score(r, matcher) for r in recs if r["family"] not in LLM_FAMILIES}
    if a.judged:
        outs = {}
        for line in open(a.judged):
            o = json.loads(line)
            outs[o["key"]] = o["text"]
        scores.update(judged_scores([r for r in recs if r["family"] in LLM_FAMILIES], outs))
    res = aggregate(recs, scores, a.icl_as_one)
    print(json.dumps(res, indent=1))
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
