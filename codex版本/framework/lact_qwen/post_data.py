"""One-pass, response-only post-training on frozen-teacher trajectories.

Rows contain history_ids, query_ids, response_ids, id, teacher_generated=True.
History must end at a storage-step boundary. The complete query/response fits
one current step. No padding is inserted into the memory or attention stream.
"""
from __future__ import annotations

import json
import torch


def encode_post_row(row: dict, segment: int, max_length: int) -> dict:
    history, query, response = (row[k] for k in ("history_ids", "query_ids", "response_ids"))
    if row.get("teacher_generated") is not True:
        raise ValueError("post-training requires a recorded frozen-teacher response")
    if not history or len(history) % segment:
        raise ValueError("history must contain complete storage steps")
    if not query or not response or len(query) + len(response) > segment:
        raise ValueError("query and response must fit one nonempty current segment")
    ids = history + query + response
    if len(ids) > max_length:
        raise ValueError("trajectory exceeds max_length; silent truncation is not allowed")
    if any(type(i) is not int or i < 0 for i in ids):
        raise ValueError("token IDs must be nonnegative integers")
    segments = [(a, a + segment) for a in range(0, len(history), segment)]
    segments.append((len(history), len(ids)))
    # Position immediately before the first answer token predicts that answer.
    labels = [-100] * (len(history) + len(query)) + response
    return {
        "input_ids": torch.tensor([ids]), "labels": torch.tensor([labels]),
        "segments": segments, "update_flags": [True] * (len(segments) - 1) + [False],
    }


def post_batches(path, segment, max_length, rank=0, world=1, teacher_cache=None):
    seen = set()
    with open(path) as stream:
        for index, line in enumerate(line for line in stream if line.strip()):
            row = json.loads(line)
            identity = row["id"]
            if identity in seen:
                raise ValueError(f"duplicate post-training trajectory: {identity}")
            seen.add(identity)
            if index % world == rank:
                batch = encode_post_row(row, segment, max_length)
                if teacher_cache is not None:
                    batch["teacher_topk"] = teacher_cache.get(row, batch)
                yield batch


def supervised_token_scale(local_count: torch.Tensor, global_count: torch.Tensor, world: int):
    """DDP averages rank gradients; recover a mean over supervised tokens."""
    return local_count * world / global_count.clamp_min(1)
