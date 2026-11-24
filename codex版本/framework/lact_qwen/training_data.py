"""Synchronized finite iterators and token weighting for gradient accumulation."""
import torch
import torch.distributed as dist


def optimizer_batch(batches, accumulation, device, dummy_token):
    """All ranks execute the same microbatch count, including an uneven final batch.

    Exhausted ranks contribute differentiable zero losses. No real examples are
    duplicated or dropped. Counts exclude padding and are reduced once per step.
    Invalid input on any rank is reported on all ranks before model collectives.
    """
    microbatches = []
    error = None
    try:
        for _ in range(accumulation):
            batch = next(batches, None)
            if batch is None:
                break
            microbatches.append(batch)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    counts = torch.tensor([
        sum(int((b["labels"][:, 1:] != -100).sum()) for b in microbatches),
        sum(int(b["attention_mask"].sum()) if b.get("attention_mask") is not None
            else b["input_ids"].numel() for b in microbatches),
        int(error is not None),
    ], dtype=torch.long, device=device)
    distributed = dist.is_initialized()
    if distributed:
        dist.all_reduce(counts)
    if counts[2].item():
        errors = [error]
        if distributed:
            errors = [None] * dist.get_world_size()
            dist.all_gather_object(errors, error)
        raise ValueError(f"invalid training data on a rank: {errors}")
    if counts[0].item() == 0:
        return None, counts
    while len(microbatches) < accumulation:
        microbatches.append({
            "input_ids": torch.tensor([[dummy_token, dummy_token]]),
            "labels": torch.full((1, 2), -100, dtype=torch.long),
            "segments": None, "update_flags": None,
        })
    return microbatches, counts
