"""Stage-2 (post-training) data: full-context teacher roll-outs, as in MemT-1.0 Sec. 3.5.

Input: a .jsonl with one object per line holding a long context and a question about it,
    {"id": ..., "context": "...", "question": "..."}
(any public long-context QA / instruction set can be converted to this: LongAlpaca, LongAlign, LongMIT,
MuSiQue, 2Wiki, NarrativeQA, QASPER are the ones LiveMem trains on).

The frozen base model reads the whole context with full attention and answers greedily; its answer is
the response the student is later teacher-forced on, while the loss is the KL to the teacher's
distribution (train_memory.py --data post:<out> --ce-weight 0 --distill-weight 1 --distill-top-k 64).

Output rows: {"id", "history_ids", "query_ids", "response_ids", "teacher_generated": true}.
history = chat prefix + context (written into memory step by step); query = question + assistant header.

    python scripts/make_post_data.py --model /path/Qwen3-4B --input qa.jsonl --out post.jsonl \
        --max-history 30000 --max-new-tokens 256 [--shard 0 --num-shards 2]
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--segment", type=int, default=2048)
    p.add_argument("--max-history", type=int, default=30000, help="longer contexts are skipped, not truncated")
    p.add_argument("--min-history", type=int, default=2048, help="shorter contexts need no memory")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="sdpa")
    model.to(args.device).eval()
    enc = lambda text: tok(text, add_special_tokens=False).input_ids
    done = set()
    if os.path.exists(args.out):  # resumable
        with open(args.out) as f:
            done = {json.loads(line)["id"] for line in f if line.strip()}
    kept = skipped = 0
    with open(args.input) as src, open(args.out, "a") as dst:
        for n, line in enumerate(src):
            if n % args.num_shards != args.shard or not line.strip():
                continue
            row = json.loads(line)
            if row["id"] in done:
                continue
            history = enc("<|im_start|>user\n" + row["context"].strip() + "\n\n")
            # Qwen3 chat format with thinking off, written out so that the history/query split is exact
            query = enc(row["question"].strip() + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
            if not args.min_history <= len(history) <= args.max_history or len(query) + args.max_new_tokens > args.segment:
                skipped += 1
                continue
            ids = torch.tensor([history + query], device=args.device)
            with torch.no_grad():
                out = model.generate(ids, max_new_tokens=args.max_new_tokens, do_sample=False,
                                     eos_token_id=tok.convert_tokens_to_ids("<|im_end|>"), pad_token_id=tok.pad_token_id)
            response = out[0, ids.shape[1]:].tolist()
            if not response:
                skipped += 1
                continue
            dst.write(json.dumps({"id": row["id"], "history_ids": history, "query_ids": query,
                                  "response_ids": response, "teacher_generated": True}) + "\n")
            dst.flush()
            kept += 1
            if kept % 20 == 0:
                print(f"{kept} written, {skipped} skipped", flush=True)
            if args.limit and kept >= args.limit:
                break
    print(f"done: {kept} written, {skipped} skipped")


if __name__ == "__main__":
    main()
