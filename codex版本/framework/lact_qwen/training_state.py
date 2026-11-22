"""Per-rank data cursors and RNG snapshots for interrupted training.

Snapshots contain tensors and Python primitives, so weights_only=True is sufficient.
The iterator is replayed before restoring RNG, isolating data replay from model dropout.
"""
import random

import numpy as np
import torch
import torch.distributed as dist

from .checkpointing import save_training_checkpoint


class CountedIterator:
    def __init__(self, batches, consumed=0):
        self.batches = iter(batches)
        self.consumed = consumed

    def __iter__(self):
        return self

    def __next__(self):
        batch = next(self.batches)
        self.consumed += 1
        return batch


def capture_runtime(device, consumed, data_seed):
    numpy_state = np.random.get_state()
    return {
        "consumed_batches": consumed, "data_seed": data_seed,
        "device_type": torch.device(device).type,
        "python_rng": random.getstate(), "torch_rng": torch.get_rng_state(),
        "numpy_rng": (numpy_state[0], torch.tensor(numpy_state[1].astype(np.int64)),
                      *numpy_state[2:]),
        "cuda_rng": torch.cuda.get_rng_state(device) if torch.device(device).type == "cuda" else None,
    }


def restore_rng(state, device):
    if state["device_type"] != torch.device(device).type:
        raise ValueError("resume requires the same device type; use --memory for a new stage")
    random.setstate(state["python_rng"])
    torch.set_rng_state(state["torch_rng"].cpu())
    algorithm, keys, position, has_gauss, cached_gauss = state["numpy_rng"]
    np.random.set_state((algorithm, keys.cpu().numpy().astype(np.uint32), position, has_gauss, cached_gauss))
    if state["cuda_rng"] is not None:
        torch.cuda.set_rng_state(state["cuda_rng"].cpu(), device)


def raise_rank_errors(error, operation):
    errors = [error]
    if dist.is_initialized():
        errors = [None] * dist.get_world_size()
        dist.all_gather_object(errors, error)
    if any(message is not None for message in errors):
        raise RuntimeError(f"{operation} failed on a rank: {errors}")


def replay_iterator(batches, consumed):
    """Seek by replay, reporting malformed/short data to every rank before forward."""
    error = None
    try:
        if type(consumed) is not int or consumed < 0:
            raise ValueError("invalid checkpoint data cursor")
        for _ in range(consumed):
            try:
                next(batches)
            except StopIteration as exc:
                raise ValueError("saved cursor exceeds the current training data") from exc
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    raise_rank_errors(error, "data replay")
    return CountedIterator(batches, consumed)


def save_rank_checkpoint(model, optimizer, destination, metadata, runtime):
    """All ranks participate; publish only after rank states and optimizer are saved."""
    states = [runtime]
    if dist.is_initialized():
        states = [None] * dist.get_world_size()
        dist.all_gather_object(states, runtime)
    error = None
    if not dist.is_initialized() or dist.get_rank() == 0:
        try:
            save_training_checkpoint(model, optimizer, destination, metadata,
                                     runtime_state={"format_version": 1, "ranks": states})
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
    raise_rank_errors(error, "checkpoint publication")
