"""Question-conditioned relevance of every memory document from the frozen backbone's attention.

The memory is cut into chunks of at most --chunk tokens (consecutive documents are packed together; a longer
document is split). The question is placed after each chunk and one forward of the frozen base model gives,
for every chunk token, the attention it receives from the question tokens (mean over heads, over the question
and trailing template tokens, and over --layers). A document's score is the attention MASS on its tokens
(sum; max over the pieces of a split document). Mass, unlike per-token probability, is comparable across
chunks: what the question does not spend on the chunk it spends on itself and the template.

Only the question and the history are used (no answer, no evidence label). Output: json
{request id: {"doc_scores": [...], "n_docs": N}} and, with --tok-dir, the per-token scores as
<tok-dir>/<sha1 of the request id>.npy (float16, one value per memory token in document order).

Run from the LiveMem repo root:
  TOKENIZER_PATH=$BASE python tools/score_attn.py --model $BASE --datasets longmemeval_s_sub --out attn_lme_sub.json
"""
import argparse, hashlib, json, os, sys, time

import numpy as np
import torch

ROOT = os.getcwd()
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools", "data_process"))
from eval.materialize import materialize_requests  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="frozen base model (no memory branch)")
    ap.add_argument("--datasets", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--layers", default="12-35")
    ap.add_argument("--max-question-tokens", type=int, default=512)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tok-dir", default=None)
    ap.add_argument("--no-pack", action="store_true", help="never put two documents in one chunk")
    args = ap.parse_args()
    lo, hi = (int(x) for x in args.layers.split("-"))
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="eager").eval().cuda()
    layers = model.model.layers
    sel = {}
    acc = {}

    def hook(li):
        def f(mod, inp, out):
            w = out[1]
            if w is None or lo > li or li > hi:
                return
            a = w[0].float()[:, sel["q"]][:, :, sel["k"]]  # [heads, nq, nk]
            acc["sum"] = acc.get("sum", 0) + a.mean(dim=(0, 1))
            acc["n"] = acc.get("n", 0) + 1
        return f

    for li, layer in enumerate(layers):
        layer.self_attn.register_forward_hook(hook(li))

    pre = tok.apply_chat_template([{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "<<<SPLIT>>>"}],
                                  tokenize=False, add_generation_prompt=True)
    head, tail = pre.split("<<<SPLIT>>>")
    h_ids = tok(head, add_special_tokens=False).input_ids
    t_ids = tok(tail, add_special_tokens=False).input_ids
    out = json.load(open(args.out)) if os.path.exists(args.out) else {}
    if args.tok_dir:
        os.makedirs(args.tok_dir, exist_ok=True)

    for ds in args.datasets.split(","):
        reqs = materialize_requests(ds, 10, args.limit)
        t0 = time.time()
        for ri, req in enumerate(reqs):
            if req.id in out:
                continue
            q_ids = tok("\n\n" + "\n".join(req.questions), add_special_tokens=False).input_ids[: args.max_question_tokens]
            docs = [tok(d, add_special_tokens=False).input_ids for d in req.memory_docs]
            # pieces: (doc index, start, end) in document-token coordinates, packed into chunks
            chunks, cur, used = [], [], 0
            for di, d in enumerate(docs):
                pos = 0
                if args.no_pack and cur:
                    chunks.append(cur); cur, used = [], 0
                while pos < len(d):
                    room = args.chunk - used
                    take = min(room, len(d) - pos)
                    if take <= 0 or (pos == 0 and len(d) <= args.chunk and len(d) > room):
                        chunks.append(cur); cur, used = [], 0
                        continue
                    cur.append((di, pos, pos + take)); used += take; pos += take
                    if used >= args.chunk:
                        chunks.append(cur); cur, used = [], 0
            if cur:
                chunks.append(cur)
            score = [0.0] * len(docs)
            tok_scores = [np.zeros(len(d), dtype=np.float32) for d in docs]
            for ch in chunks:
                body = [t for di, s, e in ch for t in docs[di][s:e]]
                ids = torch.tensor([h_ids + body + q_ids + t_ids]).cuda()
                sel["k"] = torch.arange(len(h_ids), len(h_ids) + len(body)).cuda()
                sel["q"] = torch.arange(len(h_ids) + len(body), ids.shape[1]).cuda()
                acc.clear()
                with torch.no_grad():
                    model(input_ids=ids, logits_to_keep=1)
                a = (acc["sum"] / acc["n"]).cpu().numpy()
                off = 0
                for di, s, e in ch:
                    piece = a[off : off + (e - s)]
                    score[di] = max(score[di], float(piece.sum()))
                    tok_scores[di][s:e] = piece
                    off += e - s
            out[req.id] = {"doc_scores": score, "n_docs": len(docs)}
            if args.tok_dir:
                np.save(os.path.join(args.tok_dir, hashlib.sha1(req.id.encode()).hexdigest() + ".npy"), np.concatenate(tok_scores).astype(np.float16))
            if (ri + 1) % 10 == 0:
                json.dump(out, open(args.out, "w"))
                print(f"[{ds}] {ri + 1}/{len(reqs)}  {time.time() - t0:.0f}s", flush=True)
        json.dump(out, open(args.out, "w"))
    print("SCORE_ATTN_DONE", len(out))


if __name__ == "__main__":
    main()
