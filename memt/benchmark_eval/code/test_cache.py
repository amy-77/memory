"""Is KV-cached decoding exact? Teacher-forced comparison on the same token sequence.

For a few LoCoMo questions (history written to memory, plain-text prompt): decode 64 tokens without cache, then
  (a) one uncached forward over prompt + those tokens   -> logits at every answer position
  (b) the cached path: prompt, then the same tokens one at a time  -> logits at every answer position
and report the largest |a - b| and, wherever the argmax differs, the gap between the two best logits of (a).
"""
import argparse
import os
import sys

import torch

import adapters
from bench_memt import MemoryModel, generate_uncached


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--framework", default=os.environ.get("MEMT_FRAMEWORK"))
    p.add_argument("--n", type=int, default=4)
    args, rest = p.parse_known_args()
    sys.path.insert(0, args.framework)
    from lact_qwen.loading import add_model_args, autocast, load_model
    mp = argparse.ArgumentParser()
    add_model_args(mp)
    margs = mp.parse_args(rest)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(margs.model)
    model, _ = load_model(margs)
    device = next(model.parameters()).device
    mm = MemoryModel(model, device)
    item = adapters.BENCHMARKS["locomo"](args.data, tok)[0]
    hist = tok(adapters.history_text(tok, item, "plain"), add_special_tokens=False).input_ids
    with torch.no_grad(), autocast(margs):
        mm.reset()
        mm.write(hist)
        for q in item["questions"][:args.n]:
            prompt = tok(adapters.render(tok, q, None, "plain"), add_special_tokens=False).input_ids
            gen = generate_uncached(mm, prompt, len(hist), 64, set())
            seq = prompt + gen
            full = mm.forward_chunk(model, torch.tensor([seq], device=device), len(hist), update=False)[0].float()
            a = full[len(prompt) - 1:len(seq) - 1]                       # logits predicting gen[0..]
            rows, past = [], None
            logits, past = mm._step(prompt, len(hist), None)
            rows.append(logits)
            for k, t in enumerate(gen[:-1]):
                logits, past = mm._step([t], len(hist) + len(prompt) + k, past)
                rows.append(logits)
            b = torch.stack(rows)
            diff = (a - b).abs().max(-1).values
            flips = (a.argmax(-1) != b.argmax(-1)).nonzero().flatten().tolist()
            top2 = a.topk(2, -1).values
            gaps = [(k, round(float(top2[k, 0] - top2[k, 1]), 4)) for k in flips]
            print(f"{q['qid']}: {len(gen)} tokens, max |logit diff| {float(diff.max()):.4f} (mean {float(diff.mean()):.4f}), "
                  f"logit scale {float(a.abs().max()):.1f}; argmax differs at {gaps if gaps else 'no position'}", flush=True)
    mm.end(model)


if __name__ == "__main__":
    main()
