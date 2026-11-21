"""Write per-token write multipliers for LongMemEval prompts (see models/vllm/memory_qwen3/select_write.py).

For every request the prompt is built exactly as eval/pipeline.py builds it for a `state` profile (no
truncation), tokenised with the chat template, and each history session (one memory document) gets one
multiplier for all its tokens. System prompt, question and template tokens always get 1.

Policies (--policy name[:args]):
  ones                 every token 1 (must reproduce the released behaviour)
  oracle:A             sessions that contain the answer (dataset label) 1, every other session A
  bm25:K:A             the K sessions with the highest BM25 against the question 1, the others A
  random:K:A           K random sessions 1, the others A (control for "writing less" by itself)
  last:K:A             the K most recent sessions 1, the others A
The oracle uses the gold labels and is an upper bound, not a method.

Usage (from the LiveMem repo root, TOKENIZER_PATH set):
  python tools/build_select_masks.py --out masks/oracle_a0 --policy oracle:0 [--limit N]
"""
import argparse, hashlib, json, math, os, random, sys
from collections import Counter

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools", "data_process"))
from common import read_parquet, PROC_DIR  # noqa: E402
from eval.materialize import materialize_requests, TOK_PATH  # noqa: E402
from eval.prompt import MEM_SEP  # noqa: E402


def key_of(ids):
    return hashlib.sha1(np.asarray(list(ids), dtype=np.int32).tobytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="longmemeval_s")
    ap.add_argument("--out", required=True)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOK_PATH)
    os.makedirs(args.out, exist_ok=True)
    name, *pa = args.policy.split(":")
    reqs = materialize_requests(args.dataset, 10, args.limit)
    from eval.datasets import get as get_dataset
    cfg = get_dataset(args.dataset)
    rows = {r["id"]: r for r in read_parquet(os.path.join(PROC_DIR, cfg.name, f"{cfg.split}.parquet"))}
    rng = random.Random(args.seed)
    stats = Counter(); index = []

    for req in reqs:
        row = rows[req.id.split("#")[0]]
        qa = row["qa"]; qa = json.loads(qa) if isinstance(qa, str) else qa
        qi = int(req.id.split("#")[1]) if "#" in req.id else 0
        gold = set(qa[qi].get("evidence_doc_idx") or [])
        docs = req.memory_docs
        text = tok.apply_chat_template(req.messages, tokenize=False, add_generation_prompt=True)
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        ids, offs = enc.input_ids, enc.offset_mapping
        ids2 = tok.apply_chat_template(req.messages, tokenize=True, add_generation_prompt=True)
        ids2 = list(ids2["input_ids"]) if hasattr(ids2, "keys") else list(ids2)
        if ids2 and isinstance(ids2[0], list):
            ids2 = ids2[0]
        assert list(ids) == list(ids2), f"tokenisation mismatch for {req.id}"
        # character span of every memory document inside the rendered prompt
        mem = MEM_SEP.join(docs)
        m0 = text.find(mem)
        assert m0 >= 0, f"memory not found verbatim in the prompt of {req.id}"
        spans, c = [], m0
        for d in docs:
            spans.append((c, c + len(d))); c += len(d) + len(MEM_SEP)
        starts = np.array([o[0] for o in offs]); ends = np.array([o[1] for o in offs])
        tok_spans = [(int(np.searchsorted(ends, a, side="right")), int(np.searchsorted(starts, b, side="left"))) for a, b in spans]

        n = len(docs)
        if name == "ones":
            keep, alpha = set(range(n)), 1.0
        elif name == "oracle":
            keep, alpha = gold, float(pa[0])
        elif name in ("bm25", "random", "last"):
            k, alpha = int(pa[0]), float(pa[1])
            if name == "random":
                keep = set(rng.sample(range(n), min(k, n)))
            elif name == "last":
                keep = set(range(max(0, n - k), n))
            else:
                q = set(tok(req.questions[0], add_special_tokens=False).input_ids)
                dt = [tok(d, add_special_tokens=False).input_ids for d in docs]
                df = Counter(t for d in dt for t in set(d)); avg = sum(map(len, dt)) / n
                idf = {t: math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5)) for t in q}
                sc = []
                for d in dt:
                    cnt = Counter(d)
                    sc.append(sum(idf[t] * cnt[t] * 2.5 / (cnt[t] + 1.5 * (0.25 + 0.75 * len(d) / avg)) for t in q if t in cnt))
                keep = set(sorted(range(n), key=lambda i: -sc[i])[:k])
        else:
            raise SystemExit(f"unknown policy {args.policy}")

        w = np.ones(len(ids), dtype=np.float16)
        for i, (a, b) in enumerate(tok_spans):
            if i not in keep:
                w[a:b] = alpha
        np.save(os.path.join(args.out, key_of(ids) + ".npy"), w)
        kept_tok = sum(b - a for i, (a, b) in enumerate(tok_spans) if i in keep)
        mem_tok = sum(b - a for a, b in tok_spans)
        stats["n"] += 1; stats["gold_kept"] += len(gold & keep); stats["gold"] += len(gold)
        stats["kept_tok"] += kept_tok; stats["mem_tok"] += mem_tok; stats["prompt_tok"] += len(ids)
        index.append({"id": req.id, "key": key_of(ids), "prompt_len": len(ids), "n_docs": n, "kept": sorted(keep), "gold": sorted(gold)})

    json.dump({"policy": args.policy, "stats": dict(stats), "index": index}, open(os.path.join(args.out, "_index.json"), "w"))
    print(f"{args.policy}: {stats['n']} prompts, mean prompt {stats['prompt_tok'] / stats['n']:.0f} tok, "
          f"written share of history {stats['kept_tok'] / max(stats['mem_tok'], 1):.3f}, "
          f"gold sessions kept {stats['gold_kept']}/{stats['gold']}")


if __name__ == "__main__":
    main()
