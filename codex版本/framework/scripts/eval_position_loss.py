"""Per-position next-token loss on long documents (the LaCT language-model metric).

If the model uses context beyond the attention window, the loss keeps falling at
positions past the window; a window-only model flattens out there.

    python scripts/eval_position_loss.py --data hf:emozilla/pg19:test --seq-len 16384 --window 0
    python scripts/eval_position_loss.py --data hf:emozilla/pg19:test --seq-len 16384 --window 2048
    python scripts/eval_position_loss.py --data hf:emozilla/pg19:test --seq-len 16384 --window 2048 --sinks 4
    python scripts/eval_position_loss.py --data hf:emozilla/pg19:test --seq-len 16384 --window 2048 --memory ckpt/

--data accepts  hf:<dataset>[:<split>[:<text field>]]  or a local .txt / .jsonl file
(jsonl rows need a "text" field).
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lact_qwen.loading import add_model_args, autocast, load_model, load_tokenizer
from lact_qwen.text_data import iter_documents


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True)
    p.add_argument("--seq-len", type=int, default=16384)
    p.add_argument("--num-docs", type=int, default=32)
    p.add_argument("--bin", type=int, default=1024, help="report the mean loss per bin of positions")
    p.add_argument("--out", default=None)
    p.add_argument(
        "--no-write",
        action="store_true",
        help="never update the fast weights: the branch then acts as a fixed adapter, which separates "
        "what it gains from adapting to the text from what it gains by storing history",
    )
    add_model_args(p)
    args = p.parse_args()

    tok = load_tokenizer(args)
    model, mem_cfg = load_model(args, tok)
    if args.no_write:
        if mem_cfg is None:
            raise SystemExit("--no-write needs --memory")
        from lact_qwen.fast_weight import fixed_segments

        segments = fixed_segments(args.seq_len, mem_cfg.chunk_size)
        model.lact_ctx.set_segments(segments, [False] * len(segments))

    total = torch.zeros(args.seq_len - 1, dtype=torch.float64)
    used = 0
    for text in iter_documents(args.data):
        ids = tok(text, add_special_tokens=False).input_ids
        if len(ids) < args.seq_len:
            continue
        x = torch.tensor(ids[: args.seq_len])[None].to(args.device)
        with torch.no_grad(), autocast(args):
            logits = model(input_ids=x).logits
        # in pieces: a float copy of all [seq_len, vocab] logits needs about 40 GB at 32K tokens
        n = x.shape[1] - 1
        nll = torch.cat(
            [
                torch.nn.functional.cross_entropy(
                    logits[0, a : min(a + 2048, n)].float(), x[0, a + 1 : min(a + 2048, n) + 1], reduction="none"
                )
                for a in range(0, n, 2048)
            ]
        )
        total += nll.double().cpu()
        used += 1
        print(f"[{used}/{args.num_docs}] mean loss {nll.mean():.4f}", flush=True)
        if used >= args.num_docs:
            break
    if used == 0:
        raise SystemExit(f"no document in {args.data} has {args.seq_len} tokens")

    mean = total / used
    bins = []
    for s in range(0, args.seq_len - 1, args.bin):
        e = min(s + args.bin, args.seq_len - 1)
        bins.append({"start": s, "end": e, "loss": float(mean[s:e].mean())})

    print(f"\nwindow={args.window or 'full'} sinks={args.sinks} memory={args.memory} no_write={args.no_write} docs={used}")
    print(f"{'positions':>16s} {'loss':>8s}")
    for b in bins:
        print(f"{b['start']:>7d}-{b['end']:<8d} {b['loss']:8.4f}")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"args": vars(args), "docs": used, "bins": bins}, f, indent=2)


if __name__ == "__main__":
    main()
