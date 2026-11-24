"""Score the answers of bench_memt.py runs.

usage: python judge_bench.py <out dir> [--judge <model path>] [--tp N] [--no-llm]
  <out dir>/<bench>/<format>_<mode>.s*.jsonl  -> one run per (bench, format, mode); shards are merged.

  locomo, longmemeval_s   LLM judge, OmniMemEval JUDGE_PROMPT (the prompt MemT-1.0 used; MemT's judge model was
                          gpt-4.1-mini, here a local model), temperature 0, CORRECT/WRONG
  personamem_32k          the option letter after <final_answer> (else the first "(x)") equals the gold option
  mab                     MemoryAgentBench official metrics per source (score_mab.py): rule-based, plus LLM judge
                          prompts for longmemeval_s* and infbench_sum; categories AR/TTL/LRU/CR, overall = their mean
  beam_1m                 OmniMemEval BEAM judge: every rubric item judged 0 / 0.5 / 1 and averaged (nugget score);
                          event ordering: the judge locates each reference event in the answer, score = normalised
                          Kendall tau-b x coverage. Benchmark score = mean over questions x 100.
Verdicts are cached in <out dir>/judge_cache.json, keyed by (bench, question, gold, prediction), so equal answers in
different runs get the same verdict and a rerun only judges new answers. Writes <out dir>/scores.json and prints a table.
"""
import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import adapters  # noqa: E402
import score_mab  # noqa: E402

JUDGE_SYSTEM_PROMPT = ("You are an expert grader that determines if answers to questions "
                       "match a gold standard answer.")
# verbatim from OmniMemEval scripts/utils/prompts.py (Apache-2.0)
JUDGE_PROMPT = """
    Your task is to label an answer to a question as 'CORRECT' or 'WRONG'. You will be given the following data:
        (1) a question (posed by one user to another user),
        (2) a 'gold' (ground truth) answer,
        (3) a generated answer
    which you will score as CORRECT/WRONG.

    The point of the question is to ask about something one user should know about the other user based on their prior conversations.
    The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
    Question: Do you remember what I got the last time I went to Hawaii?
    Gold answer: A shell necklace
    The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT.

    For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like "last Tuesday" or "next month"), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it's the same date.

    Now it's time for the real question:
    Question: {question}
    Gold answer: {golden_answer}
    Generated answer: {response}

    First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG.
    Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.

    Just return the label CORRECT or WRONG in a json format with the key as "label".
    """
LOCOMO_CATEGORY = {"1": "multi hop", "2": "temporal", "3": "open domain", "4": "single hop"}
LLM_JUDGED = {"locomo", "longmemeval_s"}
BEAM_SYSTEM = "You are a precise and fair evaluation judge."
# verbatim from OmniMemEval scripts/utils/prompts.py (Apache-2.0)
BEAM_RUBRIC_ITEM_JUDGE_PROMPT = """\
You are an expert evaluator tasked with judging whether the LLM's response \
demonstrates compliance with the specified RUBRIC CRITERION.

## QUESTION:
{question}

## RUBRIC CRITERION (what to check):
{rubric_item}

## RESPONSE TO EVALUATE:
{response}

## SCORING:
- **1.0** = Fully satisfied: The response clearly and completely addresses \
this rubric criterion.
- **0.5** = Partially satisfied: The response addresses this criterion but \
is incomplete, vague, or only partially correct.
- **0.0** = Not satisfied: The response does not address this criterion at \
all, or is incorrect.

## SEMANTIC TOLERANCE:
Judge by meaning, not exact wording. Accept paraphrases and synonyms that \
preserve intent. Numbers/currencies/dates may appear in equivalent forms \
(e.g. "$68,000", "68k"). Treat them as equal when numerically equivalent.

Evaluate the response against ONLY this specific rubric criterion.

Return your evaluation as JSON:
{{"score": <0.0 or 0.5 or 1.0>, "reason": "<brief explanation>"}}\
"""
BEAM_EVENT_ORDERING_JUDGE_PROMPT = """\
You are an expert evaluator. The question asked the user to list events in \
chronological order. The reference ordering of events is given below as a \
numbered list. Your task is to determine the order in which these same \
events appear in the response.

## QUESTION:
{question}

## REFERENCE ORDERING (correct chronological order):
{reference_ordering}

## RESPONSE TO EVALUATE:
{response}

## INSTRUCTIONS:
For each event in the reference list, determine its position (1-based) in \
the response. If an event is NOT mentioned in the response at all, assign \
position -1.

Return a JSON object mapping each event label to its detected position:
{{"positions": [{{"event": "<event label>", "position": <int or -1>}}, ...]}}\
"""


def beam_rubric(r):
    return r.get("rubric") or [f"LLM response should contain: {r['gold']}"]


def beam_requests(r):
    """(cache key, chat messages) of every judge call one BEAM answer needs."""
    rub = beam_rubric(r)
    if r["category"] == "event_ordering":
        listing = "\n".join(f"{i + 1}. {x}" for i, x in enumerate(rub))
        prompt = BEAM_EVENT_ORDERING_JUDGE_PROMPT.format(question=r["question"], reference_ordering=listing, response=r["pred"])
        return [(json.dumps(["beam_order", r["question"], rub, r["pred"]]),
                 [{"role": "system", "content": BEAM_SYSTEM}, {"role": "user", "content": prompt}])]
    return [(json.dumps(["beam_item", r["question"], item, r["pred"]]),
             [{"role": "system", "content": BEAM_SYSTEM},
              {"role": "user", "content": BEAM_RUBRIC_ITEM_JUDGE_PROMPT.format(question=r["question"], rubric_item=item,
                                                                              response=r["pred"])}]) for item in rub]


def beam_item_score(text):                                     # OmniMemEval _extract_item_score
    m = re.search(r'\{[^{}]*"score"\s*:\s*([0-9.]+)[^{}]*\}', text) or re.search(r'"?score"?\s*[:=]\s*([0-9.]+)', text)
    if not m:
        return 0.0
    raw = float(m.group(1))
    return 1.0 if raw >= 0.75 else 0.5 if raw >= 0.25 else 0.0


def kendall_tau_b(ref, pred):                                  # OmniMemEval _kendall_tau_b
    n = len(ref)
    if n < 2:
        return 1.0
    conc = disc = tr = tp = 0
    for i in range(n):
        for j in range(i + 1, n):
            a, b = ref[i] - ref[j], pred[i] - pred[j]
            if a == 0 and b == 0:
                tr += 1; tp += 1
            elif a == 0:
                tr += 1
            elif b == 0:
                tp += 1
            elif (a > 0) == (b > 0):
                conc += 1
            else:
                disc += 1
    pairs = n * (n - 1) // 2
    den = ((pairs - tr) * (pairs - tp)) ** 0.5
    return 0.0 if den == 0 else (conc - disc) / den


def beam_order_score(text, n_items):                           # OmniMemEval _judge_event_ordering
    m = re.search(r'\{[^{}]*"positions"\s*:\s*\[.*?\]\s*\}', text, re.DOTALL)
    if not m:
        return 0.0
    try:
        detected = [e.get("position", -1) for e in json.loads(m.group(0)).get("positions", [])]
    except (json.JSONDecodeError, AttributeError):
        return 0.0
    detected = (detected + [-1] * n_items)[:n_items]
    found = [(i, p) for i, p in enumerate(detected) if isinstance(p, int) and p > 0]
    coverage = len(found) / n_items
    if len(found) < 2:
        return 0.0 if coverage == 0 else coverage * 0.5
    tau = kendall_tau_b([i for i, _ in found], [p for _, p in found])
    return max(0.0, (tau + 1.0) / 2.0) * coverage


def beam_score(r, cache):
    """Nugget score of one BEAM answer in [0, 1], or None while a judge call is missing."""
    reqs = beam_requests(r)
    if any(k not in cache for k, _ in reqs):
        return None
    if r["category"] == "event_ordering":
        return beam_order_score(cache[reqs[0][0]]["judge_text"], len(beam_rubric(r)))
    return sum(beam_item_score(cache[k]["judge_text"]) for k, _ in reqs) / len(reqs)


def option_letter(pred):
    m = re.search(r"<final_answer>\s*\(?([a-dA-D])\)?", pred) or re.search(r"\(([a-dA-D])\)", pred) \
        or re.match(r"\s*([a-dA-D])[\s\.\):]", pred + " ")
    return f"({m.group(1).lower()})" if m else None


def plain_stops(r):
    """Stop strings of a plain-text answer (records written before "stop" was stored get today's rule)."""
    if "stop" in r:
        return r["stop"]
    if r["bench"] == "beam_1m":
        return adapters.beam_stop(r["category"])
    return adapters.SHORT_STOP


def final_answer(r):
    """The answer that is judged: plain-text output cut at the first stop string, the same rule for every run
    (early stopping during decoding only saves time; it never changes the text before the first stop string)."""
    if r.get("format") == "plain" and "raw" in r:
        pred = adapters.strip_lead(r["raw"])
        for s in plain_stops(r):
            pred = pred.split(s)[0]
        r["pred"] = pred.strip()
    return r


def parse_label(text):
    m = re.search(r'"label"\s*:\s*"?(CORRECT|WRONG)', text, re.I)
    if m:
        return m.group(1).upper() == "CORRECT"
    up = text.upper()
    return "CORRECT" in up and "WRONG" not in up


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--judge", default=None, help="local judge model for vLLM")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--mem", type=float, default=0.85, help="vLLM gpu_memory_utilization")
    ap.add_argument("--eager", action="store_true", help="vLLM enforce_eager (no CUDA graphs, less memory)")
    ap.add_argument("--judge-only", action="store_true", help="only run and cache the judge calls, then exit")
    ap.add_argument("--no-llm", action="store_true", help="only rule-scored benchmarks and cached verdicts")
    ap.add_argument("--entity2id", default=None, help="MemoryAgentBench/entity2id.json (recsys scoring)")
    args = ap.parse_args()

    runs = defaultdict(list)                                   # (bench, format, mode) -> records
    for f in glob.glob(os.path.join(args.out, "*", "*.jsonl")):
        bench = os.path.basename(os.path.dirname(f))
        fmt, mode = os.path.basename(f).split(".s")[0].split("_", 1)
        runs[(bench, fmt, mode)] += [final_answer(json.loads(l)) for l in open(f)]
    cache_path = os.path.join(args.out, "judge_cache.json")
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
    key = lambda r: json.dumps([r["bench"], r["question"], str(r["gold"]), r["pred"]])

    requests = {}                                              # cache key -> chat messages
    for (bench, _, _), rs in runs.items():
        for r in rs:
            if bench in LLM_JUDGED:
                requests[key(r)] = [{"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                                    {"role": "user", "content": JUDGE_PROMPT.format(question=r["question"], golden_answer=str(r["gold"]),
                                                                                    response=r["pred"] or "(empty)")}]
            elif bench == "beam_1m":
                requests.update(beam_requests(r))
    sampling = {}                                              # cache key -> (temperature, max_tokens) if not default
    mab_keys = {}                                              # (run, request key) -> cache key
    for (bench, fmt, mode), rs in runs.items():
        if bench != "mab":
            continue
        for q in score_mab.judge_requests(rs):
            ck = json.dumps(["mab", q["kind"], q["messages"]])
            requests[ck] = q["messages"]
            sampling[ck] = (q["temperature"], q["max_tokens"])
            mab_keys[((bench, fmt, mode), q["key"])] = ck
    todo = sorted(set(requests) - set(cache))
    print(f"runs {len(runs)}, judge calls needed {len(requests)}, not cached {len(todo)}", flush=True)
    if todo and not args.no_llm:
        from vllm import LLM, SamplingParams
        llm = LLM(model=args.judge, tensor_parallel_size=args.tp, max_model_len=16384, gpu_memory_utilization=args.mem,
                  enforce_eager=args.eager, limit_mm_per_prompt={"image": 0, "video": 0})
        groups = defaultdict(list)                             # one vLLM call per sampling setting
        for k in todo:
            groups[sampling.get(k, (0.0, 512))].append(k)
        for (temp, max_tok), keys in groups.items():
            outs = llm.chat([requests[k] for k in keys], SamplingParams(temperature=temp, max_tokens=max_tok),
                            chat_template_kwargs={"enable_thinking": False})
            for k, o in zip(keys, outs):
                cache[k] = {"judge_text": o.outputs[0].text[-6000:]}
            json.dump(cache, open(cache_path, "w"))
    if args.judge_only:                                       # free the GPUs before the CPU-only scoring
        print("judge calls done", flush=True)
        return

    scores = {}
    print(f"\n{'bench':16s} {'format':6s} {'mode':9s} {'n':>5s} {'score':>6s}   per category")
    for (bench, fmt, mode), rs in sorted(runs.items()):
        if bench == "mab":
            matcher = score_mab.MovieMatcher(args.entity2id) if any(r["family"] == "recsys" for r in rs) else None
            sc = {r["qid"]: score_mab.rule_score(r, matcher) for r in rs if r["family"] not in score_mab.LLM_FAMILIES}
            outs = {k: cache[ck]["judge_text"] for (run, k), ck in mab_keys.items() if run == (bench, fmt, mode) and ck in cache}
            sc.update(score_mab.judged_scores([r for r in rs if r["family"] in score_mab.LLM_FAMILIES], outs))
            res = score_mab.aggregate(rs, sc)
            scores[f"{bench}/{fmt}/{mode}"] = {"n": len(rs), "score": res["overall"], "mab": res}
            per = "  ".join(f"{c}:{v}" for c, v in res["per_category"].items())
            print(f"{bench:16s} {fmt:6s} {mode:9s} {len(rs):5d} {res['overall'] if res['overall'] is not None else float('nan'):6.1f}   {per}")
            continue
        cat = defaultdict(list)
        for r in rs:
            if bench in LLM_JUDGED:
                v = cache.get(key(r))
                if v is None:
                    continue
                ok = parse_label(v["judge_text"])
            elif bench == "beam_1m":
                ok = beam_score(r, cache)
                if ok is None:
                    continue
            else:
                ok = option_letter(r["pred"]) == r["gold"].strip()
            c = LOCOMO_CATEGORY.get(r["category"], r["category"]) if bench == "locomo" else r["category"]
            cat[c].append(ok)
            cat["ALL"].append(ok)
        if not cat["ALL"]:
            continue
        acc = {c: round(100 * sum(v) / len(v), 1) for c, v in cat.items()}
        scores[f"{bench}/{fmt}/{mode}"] = {"n": len(cat["ALL"]), "total_in_files": len(rs), "score": acc["ALL"],
                                          "per_category": {c: [acc[c], len(cat[c])] for c in cat if c != "ALL"}}
        per = "  ".join(f"{c}:{acc[c]}({len(cat[c])})" for c in sorted(cat) if c != "ALL")
        print(f"{bench:16s} {fmt:6s} {mode:9s} {len(cat['ALL']):5d} {acc['ALL']:6.1f}   {per}")
    json.dump(scores, open(os.path.join(args.out, "scores.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
