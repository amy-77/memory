"""Score retention, correction and recent-write accuracy on the synthetic task.

The dialogue is fed as one sequence, cut into write segments at turn boundaries, and
attention only sees the last --window tokens. The question and answer form a final
read-only segment, so answers are never written into memory.

Run the same data under several settings to get the comparison table, e.g.
    # upper bound: full attention, no memory
    python scripts/eval_synth.py --data D.jsonl --window 0
    # lower bound: window only (with 4 attention sinks so the frozen model stays sane)
    python scripts/eval_synth.py --data D.jsonl --window 2048 --sinks 4
    # window + trained memory, with and without replay
    python scripts/eval_synth.py --data D.jsonl --window 2048 --memory ckpt/ --replay none
    python scripts/eval_synth.py --data D.jsonl --window 2048 --memory ckpt/ --replay oracle_key
    python scripts/eval_synth.py --data D.jsonl --window 2048 --memory ckpt/ --replay gmm
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lact_qwen.dialog import build_query_batch, encode_history, score_rows
from lact_qwen.loading import add_model_args, autocast, load_model, load_tokenizer
from lact_qwen.synth import read_jsonl


def distance_bucket(dist: int, window: int) -> str:
    if window <= 0:
        return "full_attention"
    return "inside_window" if dist < window else "beyond_window"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True)
    p.add_argument("--limit", type=int, default=None, help="only the first N dialogues")
    p.add_argument("--max-segment", type=int, default=4096, help="split longer turns into several writes")
    p.add_argument("--out", default=None, help="write per-query results and the summary as json")
    p.add_argument(
        "--min-write-tokens",
        type=int,
        default=0,
        help="join turns into write units of at least this many tokens; use the value the model was trained with",
    )
    p.add_argument(
        "--no-write",
        action="store_true",
        help="never update the fast weights; the gap to the normal run is what storing history contributes",
    )
    add_model_args(p)
    args = p.parse_args()

    tok = load_tokenizer(args)
    model, mem_cfg = load_model(args, tok)
    ctx = getattr(model, "lact_ctx", None)
    dialogues = read_jsonl(args.data)[: args.limit]

    records = []
    for di, dlg in enumerate(dialogues):
        enc = encode_history(dlg, tok)
        batch = build_query_batch(
            enc, dlg["queries"], tok, tok.pad_token_id, args.max_segment, args.min_write_tokens
        )
        if args.no_write:
            batch.update_flags = [False] * len(batch.update_flags)
        if ctx is not None:
            ctx.set_segments(batch.segments, batch.update_flags)
        total = batch.input_ids.shape[1]
        keep = total - min(r["answer_span"][0] for r in batch.rows) + 1
        with torch.no_grad(), autocast(args):
            logits = model(input_ids=batch.input_ids.to(args.device), logits_to_keep=keep).logits
        scored = score_rows(logits, batch, offset=total - keep)

        by_query = defaultdict(dict)
        for s in scored:
            by_query[s["query_index"]][s["which"]] = s
        n_hist = len(enc.history_ids)
        for qi, q in enumerate(dlg["queries"]):
            res = by_query[qi]
            fact_end = enc.turn_spans[q["fact_turn"]][1]
            dist = n_hist - fact_end
            rec = {
                "dialogue": dlg["id"],
                "category": q["category"],
                "distance_tokens": dist,
                "bucket": distance_bucket(dist, args.window),
                "exact": res["answer"]["exact"],
                "logprob": res["answer"]["logprob"],
            }
            if "old_answer" in res:
                rec["prefers_old"] = res["old_answer"]["logprob"] > res["answer"]["logprob"]
            records.append(rec)
        print(f"[{di + 1}/{len(dialogues)}] history={n_hist} tokens", flush=True)

    summary = summarize(records)
    print_summary(summary, args)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"args": vars(args), "summary": summary, "records": records}, f, indent=2)


def summarize(records):
    out = {}
    groups = defaultdict(list)
    for r in records:
        groups[(r["category"], "all")].append(r)
        groups[(r["category"], r["bucket"])].append(r)
    for (cat, bucket), rs in sorted(groups.items()):
        entry = {
            "n": len(rs),
            "accuracy": sum(r["exact"] for r in rs) / len(rs),
            "answer_logprob": sum(r["logprob"] for r in rs) / len(rs),
        }
        olds = [r["prefers_old"] for r in rs if "prefers_old" in r]
        if olds:
            entry["prefers_old_rate"] = sum(olds) / len(olds)
        out[f"{cat}/{bucket}"] = entry
    return out


def print_summary(summary, args):
    setting = (
        f"window={args.window or 'full'} sinks={args.sinks} memory={args.memory} "
        f"replay={args.replay or 'from-config'} no_write={args.no_write} min_write={args.min_write_tokens}"
    )
    print(f"\n{setting}")
    print(f"{'category/bucket':32s} {'n':>5s} {'acc':>7s} {'logprob':>9s} {'prefers_old':>12s}")
    for key, e in summary.items():
        old = f"{e['prefers_old_rate']:.3f}" if "prefers_old_rate" in e else ""
        print(f"{key:32s} {e['n']:5d} {e['accuracy']:7.3f} {e['answer_logprob']:9.3f} {old:>12s}")


if __name__ == "__main__":
    main()
