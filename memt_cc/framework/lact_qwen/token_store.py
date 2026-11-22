"""Pre-tokenised documents on disk (written once by scripts/pretokenize.py).

    <dir>/tokens.u32    all kept documents, each followed by EOS, uint32
    <dir>/offsets.npy   int64 [num_docs + 1], start of every document in tokens.u32
    <dir>/meta.json

Training reads windows through np.memmap: no tokenisation in the training loop, every rank touches only
its own windows, and resuming at optimizer step N is an index computation, not a replay of the stream.

Every training example is a window of ONE document, starting at the document start or a multiple of
seq_len inside it. Memory starts at zero for every example and the 2K attention/memory steps are aligned
to the window start, as in MemT-1.0 Sec. 3.4 ("initialized to zero at the beginning of each independent
sequence") and 4.1 ("memory ... resets at trajectory boundaries").
"""

from __future__ import annotations

import json
import os
from typing import Iterator

import numpy as np


def document_windows(lengths, seq_len: int, min_window: int):
    """[(doc, start, end)] over documents of the given token lengths; tails below min_window are dropped."""
    out = []
    for d, n in enumerate(lengths):
        for s in range(0, int(n), seq_len):
            e = min(s + seq_len, int(n))
            if e - s >= min_window:
                out.append((d, s, e))
    return out


class TokenStore:
    def __init__(self, path: str):
        with open(os.path.join(path, "meta.json")) as f:
            self.meta = json.load(f)
        self.offsets = np.load(os.path.join(path, "offsets.npy"))
        self.tokens = np.memmap(os.path.join(path, "tokens.u32"), dtype=np.uint32, mode="r")
        if int(self.offsets[-1]) != len(self.tokens):
            raise ValueError("offsets.npy does not match tokens.u32 (incomplete pretokenisation?)")

    def windows(self, seq_len: int, min_window: int) -> np.ndarray:
        lengths = np.diff(self.offsets)
        w = document_windows(lengths, seq_len, min_window)
        if not w:
            raise ValueError("no document window is long enough")
        w = np.asarray(w, dtype=np.int64)
        w[:, 1] += self.offsets[w[:, 0]]
        w[:, 2] += self.offsets[w[:, 0]]
        return w[:, 1:]  # absolute [start, end)

    def stream(self, seq_len: int, min_window: int, rank: int = 0, world: int = 1, seed: int = 0,
               skip: int = 0) -> Iterator[list[int]]:
        """Endless stream of windows for this rank. Each epoch is a seeded permutation of all windows,
        cut into complete rounds of `world` (rank r takes position r of every round), so ranks never
        share a window within an epoch. skip = number of examples this rank has already consumed."""
        windows = self.windows(seq_len, min_window)
        per_epoch = len(windows) // world
        if per_epoch == 0:
            raise ValueError(f"only {len(windows)} windows for {world} ranks")
        epoch, i = divmod(skip, per_epoch)
        while True:
            order = np.random.default_rng([seed, epoch]).permutation(len(windows))
            while i < per_epoch:
                s, e = windows[order[i * world + rank]]
                yield self.tokens[s:e].astype(np.int64).tolist()
                i += 1
            epoch, i = epoch + 1, 0
