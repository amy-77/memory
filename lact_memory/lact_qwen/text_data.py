from __future__ import annotations

import glob
import itertools
import json
import os
from typing import Iterator, Optional


def iter_documents(
    spec: str,
    rank: int = 0,
    world: int = 1,
    shuffle_seed: Optional[int] = None,
    shuffle_buffer: int = 1000,
) -> Iterator[str]:
    """Yield raw documents from 'hf:<name>[:<split>[:<field>]]', a ``save_to_disk`` directory,
    a glob of .arrow shards, a .txt file, or a .jsonl file.

    With world > 1 every rank gets a disjoint share of the documents (a single .txt file is
    given to every rank). shuffle_seed shuffles the shard order and then documents within a
    buffer of shuffle_buffer; it only applies to Hugging Face sources.
    """
    if spec.startswith("hf:") or os.path.isdir(spec) or spec.endswith(".arrow"):
        yield from _iter_hf(spec, rank, world, shuffle_seed, shuffle_buffer)
        return

    if spec.endswith(".jsonl"):
        with open(spec, encoding="utf-8") as f:
            rows = (json.loads(line)["text"] for line in f if line.strip())
            yield from itertools.islice(rows, rank, None, world)
        return

    with open(spec, encoding="utf-8") as f:
        yield f.read()


def _iter_hf(spec, rank, world, shuffle_seed, shuffle_buffer):
    from datasets import load_dataset, load_from_disk
    from datasets.distributed import split_dataset_by_node

    if os.path.isdir(spec):
        field = "text"
        ds = load_from_disk(spec).select_columns([field])
        ds = ds.to_iterable_dataset(num_shards=max(1, len(ds.cache_files)))
    elif spec.endswith(".arrow"):
        field = "text"
        files = sorted(glob.glob(spec))
        if not files:
            raise FileNotFoundError(f"no files match {spec}")
        ds = load_dataset("arrow", data_files=files, split="train", streaming=True).select_columns([field])
    else:
        parts = spec[3:].split(":")
        name = parts[0]
        split = parts[1] if len(parts) > 1 and parts[1] else "train"
        field = parts[2] if len(parts) > 2 and parts[2] else "text"
        ds = load_dataset(name, split=split, streaming=True)

    if shuffle_seed is not None:
        ds = ds.shuffle(seed=shuffle_seed, buffer_size=shuffle_buffer)
    if world > 1:
        ds = split_dataset_by_node(ds, rank=rank, world_size=world)
    for row in ds:
        yield row[field]


def packed_sequences(
    spec: str,
    tokenizer,
    seq_len: int,
    repeat: bool = True,
    rank: int = 0,
    world: int = 1,
    shuffle_seed: Optional[int] = None,
    shuffle_buffer: int = 1000,
    min_doc_tokens: int = 0,
) -> Iterator[list[int]]:
    """Concatenate documents (separated by EOS) and cut them into seq_len token blocks.

    Documents shorter than min_doc_tokens are skipped. Each pass over the data reshuffles
    with shuffle_seed + pass index.
    """
    eos = tokenizer.eos_token_id
    buf: list[int] = []
    for epoch in itertools.count():
        produced = False
        seed = None if shuffle_seed is None else shuffle_seed + epoch
        for text in iter_documents(spec, rank, world, seed, shuffle_buffer):
            ids = tokenizer(text, add_special_tokens=False).input_ids
            if len(ids) < min_doc_tokens:
                continue
            buf.extend(ids)
            buf.append(eos)
            while len(buf) >= seq_len:
                yield buf[:seq_len]
                buf = buf[seq_len:]
                produced = True
        if not repeat or not produced:
            return
