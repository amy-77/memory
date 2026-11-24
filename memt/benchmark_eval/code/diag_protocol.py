"""Why does the 4B model answer with paper openings under the MemT protocol? Same LoCoMo questions, five inputs:

  A  empty memory,   chat template, question only
  B  history written, chat template, question only                         (MemT protocol: what failed)
  C  history written, plain text "... Question: ...\nAnswer:", question only
  D  history written, last ~1.9K history tokens + question, chat template
  E  history written, last ~1.9K history tokens + plain-text question        (our earlier LoCoMo protocol)
"""
import argparse
import os
import sys

import torch

import adapters
from bench_memt import CHUNK, MemoryModel


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--framework", default=os.environ.get("MEMT_FRAMEWORK"))
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--max-new", type=int, default=48)
    args, rest = p.parse_known_args()
    sys.path.insert(0, args.framework)
    from lact_qwen.loading import add_model_args, autocast, load_model
    mp = argparse.ArgumentParser()
    add_model_args(mp)
    margs = mp.parse_args(rest)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(margs.model)
    model, cfg = load_model(margs)
    device = next(model.parameters()).device
    stop = {i for i in (tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")) if isinstance(i, int)}
    mm = MemoryModel(model, device)
    item = adapters.BENCHMARKS["locomo"](args.data, tok)[0]
    hist = tok(item["history"], add_special_tokens=False).input_ids
    print(f"history {len(hist)} tokens", flush=True)

    def show(tag, ids):
        return tok.decode(ids, skip_special_tokens=True).strip().replace("\n", " | ")[:160]

    with torch.no_grad(), autocast(margs):
        for q in item["questions"][:args.n]:
            body = q["memory"][0]["content"]
            plain = tok(body + "\nAnswer:", add_special_tokens=False).input_ids
            chat_ids = tok(adapters.render(tok, q, None), add_special_tokens=False).input_ids
            print(f"\nQ {q['question']!r}  gold {q['gold']!r}", flush=True)
            mm.reset()
            print("  A empty memory, chat      :", show("A", mm.generate(chat_ids, len(hist), args.max_new, stop)), flush=True)
            mm.reset()
            mm.write(hist)
            print("  B memory, chat            :", show("B", mm.generate(chat_ids, len(hist), args.max_new, stop)), flush=True)
            print("  C memory, plain text      :", show("C", mm.generate(plain, len(hist), args.max_new, stop)), flush=True)
            # D / E: the memory holds everything before the last segment; the last segment is in attention
            tail_room = CHUNK - 64 - max(len(chat_ids), len(plain))
            cut = len(hist) - tail_room
            mm.reset()
            mm.write(hist[:cut])
            tail = hist[cut:]
            chat_tail = tok(adapters.chat(tok, [{"role": "user", "content": tok.decode(tail) + "\n\n" + body}]),
                            add_special_tokens=False).input_ids
            print("  D memory+tail, chat       :", show("D", mm.generate(chat_tail, cut, args.max_new, stop)), flush=True)
            plain_tail = tail + tok("\n\n" + body + "\nAnswer:", add_special_tokens=False).input_ids
            print("  E memory+tail, plain text :", show("E", mm.generate(plain_tail, cut, args.max_new, stop)), flush=True)
    mm.end(model)


if __name__ == "__main__":
    main()
