"""Precompute frozen full-context teacher Top-K targets for validated post-training JSONL.

This does not generate answers: response_ids must already be frozen-teacher trajectories.
The teacher cache removes the second 4B model from student training GPU memory.
"""
import argparse
import json
import os
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lact_qwen.loading import add_model_args, autocast, load_teacher, load_tokenizer
from lact_qwen.post_data import encode_post_row
from lact_qwen.teacher_cache import file_sha256, make_topk_targets, row_key, token_digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--segment", type=int, default=2048)
    parser.add_argument("--max-length", type=int, default=32768)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=1.)
    parser.add_argument("--loss-chunk-size", type=int, default=128)
    add_model_args(parser)
    args = parser.parse_args()
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=False)
    tokenizer = load_tokenizer(args)
    teacher = load_teacher(args, tokenizer)
    digest = file_sha256(args.data)
    seen = set()
    with open(args.data) as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            key = row_key(row)
            if key in seen:
                raise ValueError(f"duplicate trajectory ID: {row['id']}")
            seen.add(key)
            batch = encode_post_row(row, args.segment, args.max_length)
            x, y = batch["input_ids"].to(args.device), batch["labels"].to(args.device)
            with torch.no_grad(), autocast(args):
                hidden = teacher.model(input_ids=x, use_cache=False).last_hidden_state
                indices, log_probs = make_topk_targets(hidden, teacher.lm_head, y,
                    args.top_k, args.temperature, args.loss_chunk_size)
            torch.save({"indices": indices, "log_probs": log_probs,
                        "token_digest": token_digest(batch)}, output / (key + ".pt"))
    if file_sha256(args.data) != digest or not seen:
        raise ValueError("input changed while caching or no valid trajectories were found")
    metadata = dict(format_version=1, data_sha256=digest, model=args.model,
                    model_revision=getattr(teacher.config, "_commit_hash", None),
                    top_k=args.top_k, temperature=args.temperature, dtype=args.dtype,
                    segment=args.segment, max_length=args.max_length, rows=len(seen))
    temporary = output / "manifest.json.tmp"
    temporary.write_text(json.dumps(metadata, indent=2))
    os.replace(temporary, output / "manifest.json")
    print(f"Cached {len(seen)} teacher trajectories in {output}")


if __name__ == "__main__":
    main()
