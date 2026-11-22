"""Judge generated LoCoMo answers: LLM judge (CORRECT / WRONG, Mem0-style "J" score) + LoCoMo token F1.
usage: python judge.py <dir with merged <run>.json files> [--limit N]
Writes <run>.judged.json next to each run and prints correct / total per run, beyond / inside the 2048 window."""
import argparse, glob, json, os, re, string
from collections import Counter, defaultdict

JUDGE = "/data/models/Qwen/Qwen3.6-35B-A3B"
PROMPT = """Your task is to label an answer to a question as CORRECT or WRONG.
You are given a question, the gold (ground truth) answer, and a generated answer.
The gold answer is usually short. Be generous: the generated answer is CORRECT if it states the same fact,
person, place, thing or time as the gold answer, even in other words or another date format
(for example "May 7th" and "7 May 2023" are the same day). If the gold answer is a list, the generated answer must
name at least one item of it and nothing that contradicts it. An empty answer, "I don't know", or an answer about
a different fact is WRONG.

Question: {q}
Gold answer: {gold}
Generated answer: {pred}

Reply with one word: CORRECT or WRONG."""


def norm(s):
    s = str(s).lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the|and)\b", " ", s)
    return s.split()


def f1(pred, gold):
    p, g = norm(pred), norm(gold)
    common = sum((Counter(p) & Counter(g)).values())
    if not p or not g or not common:
        return 0.0
    pr, rc = common / len(p), common / len(g)
    return 2 * pr * rc / (pr + rc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--cache", default="out/judge_cache.json",
                    help="verdict per (question, gold, prediction), shared by all runs so equal answers get equal verdicts")
    ap.add_argument("--seed-from", default=None, help="add the verdicts of <dir>/*.judged.json to the cache first")
    args = ap.parse_args()
    cache = json.load(open(args.cache)) if os.path.exists(args.cache) else {}
    ck = lambda q, g, p: json.dumps([q, g, p])
    if args.seed_from:
        for f in glob.glob(args.seed_from + "/*.judged.json"):
            for r in json.load(open(f))["records"]:
                cache.setdefault(ck(r["question"], str(r["answer"]), r["pred"]), bool(r["judge"]))
    runs = {os.path.basename(f)[:-5]: json.load(open(f))["records"]
            for f in sorted(glob.glob(args.dir + "/*.json")) if ".s" not in os.path.basename(f) and "judged" not in f}
    runs = {k: v for k, v in runs.items() if v and v[0].get("pred") is not None}
    allkeys = sorted({(r["question"], str(r["answer"]), r["pred"]) for rs in runs.values() for r in rs})
    keys = [k for k in allkeys if ck(*k) not in cache]
    print("runs", list(runs), "distinct triples", len(allkeys), "already in the cache", len(allkeys) - len(keys), flush=True)
    verdict = {k: cache[ck(*k)] for k in allkeys if ck(*k) in cache}
    if not keys:
        LLM = None
    else:
        from vllm import LLM, SamplingParams
    if keys:
      llm = LLM(model=JUDGE, tensor_parallel_size=args.tp, max_model_len=4096, gpu_memory_utilization=0.85)
      msgs = [[{"role": "user", "content": PROMPT.format(q=q.replace("Question:", "").replace("Answer:", "").strip(), gold=g, pred=p or "(empty)")}]
            for q, g, p in keys]
      outs = llm.chat(msgs, SamplingParams(temperature=0, max_tokens=8), chat_template_kwargs={"enable_thinking": False})
      for k, o in zip(keys, outs):
          verdict[k] = "CORRECT" in o.outputs[0].text.upper() and "WRONG" not in o.outputs[0].text.upper()
          cache[ck(*k)] = verdict[k]
      json.dump(cache, open(args.cache, "w"))
    cats = ["ALL", "single_hop", "multi_hop", "temporal", "open_domain"]
    for b in ("beyond", "inside"):
        print(f"\n### evidence {b} the 2048-token window: judged correct / total (mean F1)")
        print(f"{'run':14s}" + "".join(f"{c:>22s}" for c in cats))
        for name, rs in runs.items():
            g = defaultdict(list)
            for r in rs:
                r["judge"] = verdict[(r["question"], str(r["answer"]), r["pred"])]
                r["f1"] = f1(r["pred"] or "", r["answer"])
                if (r["distance_tokens"] >= 2048) == (b == "beyond"):
                    g[r["category"]].append(r); g["ALL"].append(r)
            print(f"{name:14s}" + "".join(
                f"{sum(r['judge'] for r in g[c]):>7d}/{len(g[c]):<5d}({sum(r['f1'] for r in g[c])/max(1,len(g[c])):.2f})   " for c in cats))
    for name, rs in runs.items():
        json.dump({"records": rs}, open(f"{args.dir}/{name}.judged.json", "w"))


if __name__ == "__main__":
    main()
