"""LoCoMo with the official LiveMem-4B (frozen Qwen3-4B-Instruct-2507 + GDN2 side memory), aligned with the
4B MemT protocol in eval_locomo_memt.py:

  * same 1536 questions (data/locomo2_g4.jsonl), same judge, same "beyond window" definition;
  * the prompt is LiveMem's own (system prompt; user = sessions joined by blank lines + its LoCoMo instruction);
  * the prompt is cut into chunks of 2048 counted back from (prompt + 64 reserved tokens); attention only sees the
    current chunk (each chunk is a separate forward without attention KV cache), no attention sinks;
    older chunks reach the model only through the GDN2 recurrent state, carried from chunk to chunk.

modes
  full      memory output forced to 0, full attention over the whole prompt                         (row A)
  last      memory output forced to 0, only the last chunk                                            (row B)
  write     every earlier chunk written into memory (= released LiveMem, design X)                    (row C)
  evidence  only earlier chunks that overlap an annotated evidence session are written (labels used)  (row F)
  random    as many chunks as 'evidence', chosen at random                                            (row G)
A chunk that is not written is not run at all (as in the MemT runs), so it neither writes nor makes the state decay.
The last chunk and the generated tokens are always run with the ordinary update (LiveMem scans every token it reads).
Greedy decoding, up to --gen tokens; the answer is taken from LiveMem's "Answer: ..." format.
"""
import argparse, copy, json, random, re, sys, time, zlib
from pathlib import Path

import torch

LM = "/data/qyl/store_179/livemem_repro/LiveMem"
sys.path.insert(0, LM + "/tools/data_process")
import format as F  # noqa: E402  LiveMem's prompt builder (train / eval share it)


class MemCache:
    """Per-layer GDN2 state {recurrent_state, conv_state}, the interface fla's layer cache helpers expect."""

    def __init__(self):
        self.layers = []

    def __len__(self):
        return len(self.layers)

    def __getitem__(self, i):
        return self.layers[i]

    def update(self, layer_idx, recurrent_state=None, conv_state=None, offset=0, **kw):
        while len(self.layers) <= layer_idx:
            self.layers.append(None)
        self.layers[layer_idx] = {"recurrent_state": recurrent_state, "conv_state": conv_state}
        return self.layers[layer_idx]

    def clone(self):
        c = MemCache()
        c.layers = [None if x is None else {"recurrent_state": None if x["recurrent_state"] is None else x["recurrent_state"].clone(),
                                            "conv_state": None if x["conv_state"] is None else tuple(None if t is None else t.clone() for t in x["conv_state"])}
                    for x in self.layers]
        return c


def build(dlg, q, tok):
    raw = re.search(r"Question:\s*(.*?)\s*\nAnswer:\s*$", q["question"], re.S).group(1)
    user = "\n\n".join(dlg["turns"]) + "\n\n" + F.user_single("locomo", raw, reason=False)
    msgs = [{"role": "system", "content": F.SYSTEM_PROMPT}, {"role": "user", "content": user}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    ref = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)
    ref = ref["input_ids"] if hasattr(ref, "keys") else ref   # transformers 5 returns a BatchEncoding
    assert list(ref) == list(ids), "tokenising the rendered text differs from apply_chat_template"
    spans, pos = [], 0
    for body in dlg["turns"]:
        c0 = text.index(body, pos); c1 = c0 + len(body); pos = c1
        t0 = next(i for i, (a, b) in enumerate(offs) if b > c0)
        t1 = next(i for i, (a, b) in enumerate(offs) if a >= c1)
        spans.append((t0, t1))
    hist_end = spans[-1][1]
    return raw, ids, spans, hist_end


def extract(text):
    m = re.search(r"Answer:\s*(.+)", text)
    s = m.group(1) if m else text
    return s.split("```")[0].strip().strip("`").strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/data/qyl/memt_locomo_20261122/livemem_models/LiveMem-RL")
    p.add_argument("--data", default="data/locomo2_g4.jsonl")
    p.add_argument("--mode", required=True, choices=["full", "last", "write", "evidence", "random"])
    p.add_argument("--chunk", type=int, default=2048)
    p.add_argument("--reserve", type=int, default=64)
    p.add_argument("--gen", type=int, default=48)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--nshards", type=int, default=1)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, trust_remote_code=True,
                                                 attn_implementation="sdpa").cuda().eval()
    dev = next(model.parameters()).device
    mems = [l.self_attn.mem for l in model.model.layers]
    if args.mode in ("full", "last"):
        for m in mems:
            m.o_proj.weight.data.zero_()          # memory branch contributes exactly 0: the plain backbone
    stop = {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")}
    C = args.chunk

    def decode(ids, offset, mc):
        """Prefill ids (positions offset..), then greedy decoding with attention KV cache + GDN2 cache."""
        out = model(input_ids=torch.tensor([ids], device=dev),
                    position_ids=torch.arange(offset, offset + len(ids), device=dev)[None],
                    use_cache=True, mem_cache=mc, logits_to_keep=1)
        pkv, gen, pos = out.past_key_values, [], offset + len(ids)
        for _ in range(args.gen):
            nxt = int(out.logits[0, -1].argmax())
            if nxt in stop:
                break
            gen.append(nxt)
            out = model(input_ids=torch.tensor([[nxt]], device=dev), position_ids=torch.tensor([[pos]], device=dev),
                        past_key_values=pkv, use_cache=True, mem_cache=mc, logits_to_keep=1)
            pkv, pos = out.past_key_values, pos + 1
        return tok.decode(gen)

    rows = [json.loads(l) for l in open(args.data)]
    if args.limit:
        rows = rows[:args.limit]
    recs, t0, nw, nc, ne, nh = [], time.time(), 0, 0, 0, 0
    for ri, dlg in enumerate(rows):
        if ri % args.nshards != args.shard:
            continue
        for qi, q in enumerate(dlg["queries"]):
            raw, ids, spans, hist_end = build(dlg, q, tok)
            P = len(ids)
            bounds = [(max(0, e - C), e) for e in range(P + args.reserve, 0, -C)][::-1]
            earlier = list(range(len(bounds) - 1))
            fs = bounds[-1][0]
            ev = q.get("evidence_turns") or [q["fact_turn"]]
            evid = {j for j in earlier if any(spans[t][0] < bounds[j][1] and spans[t][1] > bounds[j][0] for t in ev)}
            with torch.no_grad():
                if args.mode == "full":
                    text = decode(ids, 0, None)
                    flags = set()
                elif args.mode == "last":
                    text = decode(ids[fs:], fs, None)
                    flags = set()
                else:
                    if args.mode == "write":
                        flags = set(earlier)
                    elif args.mode == "evidence":
                        flags = set(evid)
                    else:
                        rng = random.Random(zlib.crc32(f"{dlg['id']}#{qi}".encode()))
                        flags = set(rng.sample(earlier, len(evid)))
                    mc = MemCache()
                    for j in earlier:
                        if j in flags:
                            s, e = bounds[j]
                            model(input_ids=torch.tensor([ids[s:e]], device=dev),
                                  position_ids=torch.arange(s, e, device=dev)[None],
                                  use_cache=True, mem_cache=mc, logits_to_keep=1)
                    text = decode(ids[fs:], fs, mc)
                    nw += len(flags); nc += len(earlier); ne += len(evid); nh += len(evid & flags)
            ft = q["fact_turn"]
            recs.append({"row": ri, "query_index": qi, "dialogue": dlg["id"], "category": q["category"],
                         "distance_tokens": hist_end - spans[ft][1], "fact_turn": ft, "prompt_len": P,
                         "question": raw, "answer": q["answer"], "raw_output": text, "pred": extract(text),
                         "exact": None, "logprob": 0.0, "evidence_chunks": len(evid),
                         "chunks_written": len(flags)})
        if recs and ri % 20 == args.shard:
            print(f"row {ri+1}/{len(rows)} questions {len(recs)} {time.time()-t0:.0f}s", flush=True)
    summ = {"written_chunks": nw, "earlier_chunks": nc, "evidence_chunks": ne, "evidence_chunks_written": nh}
    print("chunks written", nw, "of", nc, "; evidence chunks written", nh, "of", ne, flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"args": vars(args), "summary": summ, "records": recs}, open(args.out, "w"))
    print("DONE", len(recs), f"{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
