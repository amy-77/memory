"""Tokenise a text source once into a TokenStore directory (see lact_qwen/token_store.py).

    python scripts/pretokenize.py --data "/path/ldc/train/data-*.arrow" --tokenizer /path/Qwen3-4B \
        --out /path/ldc_tok --min-doc-tokens 4096 --workers 32

Documents shorter than --min-doc-tokens are dropped here, once, instead of being tokenised and thrown
away by every rank at every epoch. --max-docs / --max-tokens cut the run short (for tests / subsets).
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lact_qwen.text_data import iter_documents

_tok = None


def _init(name):
    global _tok
    from transformers import AutoTokenizer

    _tok = AutoTokenizer.from_pretrained(name)


def _encode(texts):
    return [np.asarray(ids, dtype=np.uint32) for ids in _tok(texts, add_special_tokens=False).input_ids]


def _batches(spec, size, max_docs):
    batch, n = [], 0
    for text in iter_documents(spec):
        batch.append(text)
        n += 1
        if len(batch) == size:
            yield batch
            batch = []
        if max_docs and n >= max_docs:
            break
    if batch:
        yield batch


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="same specs as train_memory.py text: (arrow glob, hf:..., .jsonl)")
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--min-doc-tokens", type=int, default=4096)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--batch", type=int, default=16, help="documents per tokenizer call")
    p.add_argument("--max-docs", type=int, default=0)
    p.add_argument("--max-tokens", type=int, default=0)
    args = p.parse_args()

    from multiprocessing import Pool
    from transformers import AutoTokenizer

    eos = AutoTokenizer.from_pretrained(args.tokenizer).eos_token_id
    os.makedirs(args.out, exist_ok=False)
    offsets, total, seen, t0 = [0], 0, 0, time.time()
    with open(os.path.join(args.out, "tokens.u32"), "wb") as f, Pool(args.workers, _init, (args.tokenizer,)) as pool:
        for docs in pool.imap(_encode, _batches(args.data, args.batch, args.max_docs), chunksize=1):
            for ids in docs:
                seen += 1
                if len(ids) < args.min_doc_tokens:
                    continue
                f.write(ids.tobytes())
                f.write(np.asarray([eos], dtype=np.uint32).tobytes())
                total += len(ids) + 1
                offsets.append(total)
            if seen % 2000 < args.batch:
                print(f"{seen} documents read, {len(offsets) - 1} kept, {total / 1e6:.0f}M tokens, "
                      f"{time.time() - t0:.0f}s", flush=True)
            if args.max_tokens and total >= args.max_tokens:
                break
    np.save(os.path.join(args.out, "offsets.npy"), np.asarray(offsets, dtype=np.int64))
    meta = {"data": args.data, "tokenizer": args.tokenizer, "eos_token_id": eos, "documents_read": seen,
            "documents_kept": len(offsets) - 1, "tokens": total, "min_doc_tokens": args.min_doc_tokens}
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
