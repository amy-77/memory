"""The MemT-1.0 benchmarks for an official MemT-1.0 checkpoint (MemT-1.0-Nano) with HF transformers and the checkpoint's
own modeling_memt.py (trust_remote_code), i.e. without the SGLang server.

Protocol (report sec. 6.1, the same items and prompts as bench_memt.py; chat template, thinking off, greedy):
  mode memory    reset the memory, write the history part as one Document (the model's own document write:
                 the history is wrapped as a user message, model.commit_memory(..., document=True) writes it in
                 2048-token chunks and resets the native context between chunks, like session.ingest()),
                 then answer every question of that history with only the query in context:
                 model.generate(query, memory_state=<written memory>) -- the memory is only read.
  mode question  no memory (memory_state=None), the query alone.
Output records as bench_memt.py (format "chat"), so judge_bench.py scores them unchanged.
"""
import argparse
import json
import os
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import adapters


def transformers_compat():
    """modeling_memt.py was written for transformers 5.12.1; this env has 5.20, where qwen3_5_moe merged the torch and
    kernel versions of its GatedDeltaNet helpers. modeling_memt only looks these names up in __init__ (the inherited
    5.20 forward calls 5.20's own functions), so map the old names: the torch reference, and None for the optional
    accelerated kernels (what 5.12 has when fla / causal-conv1d are not installed, as here)."""
    from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as q
    for name, value in {"torch_causal_conv1d_update": q.causal_conv1d_update, "chunk_gated_delta_rule": None,
                        "fused_recurrent_gated_delta_rule": None}.items():
        if not hasattr(q, name):
            setattr(q, name, value)


def post_load_compat(model):
    """5.12 decoder layers call their kind `layer_type`; the 5.20 forward reads `block_type` (same values)."""
    for layer in model.model.language_model.layers:
        if not hasattr(layer, "block_type"):
            layer.block_type = layer.layer_type
    return model


def render(tok, msgs, generation):
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=generation, enable_thinking=False,
                                   preserve_thinking=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bench", required=True, choices=list(adapters.BENCHMARKS))
    p.add_argument("--data", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--mode", default="memory", choices=["memory", "question"])
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--nshards", type=int, default=1)
    p.add_argument("--limit", type=int, default=0, help="only the first N questions of this shard")
    p.add_argument("--out", required=True)
    p.add_argument("--skip", nargs="*", default=[], help="other result files whose qids count as done")
    a = p.parse_args()

    transformers_compat()
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(a.model, trust_remote_code=True, dtype=torch.bfloat16,
                                                 attn_implementation="sdpa").to("cuda:0").eval()   # no accelerate here
    post_load_compat(model)
    dev = model.get_input_embeddings().weight.device
    eos = model.generation_config.eos_token_id
    items = adapters.BENCHMARKS[a.bench](a.data, tok)
    mine = [i for i in range(len(items)) if i % a.nshards == a.shard]
    jobs = [(i, j) for i in mine for j in range(len(items[i]["questions"]))]
    if a.limit:
        jobs = jobs[:a.limit]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["qid"] for f in [out, *map(Path, a.skip)] if f.exists() for l in open(f)}
    fout, t0, written_for, state, write_s = open(out, "a"), time.time(), None, None, 0.0
    print(f"nano {a.bench} {a.mode}: shard {a.shard}/{a.nshards}: {len(mine)} histories, {len(jobs)} questions", flush=True)
    with torch.no_grad():
        for n, (i, j) in enumerate(jobs):
            item, q = items[i], items[i]["questions"][j]
            if q["qid"] in done:
                continue
            if a.mode == "memory" and written_for != i:
                doc = render(tok, [{"role": "user", "content": adapters.history_text(tok, item, "plain")}], False)
                ids = tok.encode(doc, add_special_tokens=False)
                tw = time.time()
                state = model.commit_memory(torch.tensor([ids], device=dev), None, document=True)
                written_for, write_s, hist_len = i, time.time() - tw, len(ids)
                print(f"history {item['id']}: wrote {len(ids)} tokens in {write_s:.0f}s", flush=True)
            if a.mode == "question":
                state, hist_len = None, 0
            prompt = tok.encode(render(tok, q["memory"], True), add_special_tokens=False)
            x = torch.tensor([prompt], device=dev)
            tq = time.time()
            kw = dict(do_sample=True, temperature=q["temperature"]) if q["temperature"] > 0 else dict(do_sample=False)
            o = model.generate(x, attention_mask=torch.ones_like(x), memory_state=state or None,
                               max_new_tokens=q["max_new"], **kw)
            gen = o[0, len(prompt):].tolist()
            eos_ids = [eos] if isinstance(eos, int) else list(eos or [])
            gen_vis = gen[:-1] if gen and gen[-1] in eos_ids else gen
            pred = tok.decode(gen_vis, skip_special_tokens=True).strip()
            rec = {"bench": a.bench, "mode": a.mode, "item": item["id"], "qid": q["qid"], "category": q["category"],
                   "question": q["question"], "gold": q["gold"], "pred": pred, "history_tokens": hist_len,
                   "prompt_tokens": len(prompt), "new_tokens": len(gen), "hit_max": len(gen) >= q["max_new"],
                   "format": "chat", "seconds": round(time.time() - tq, 2), "write_seconds": round(write_s, 1),
                   "model": os.path.basename(a.model.rstrip("/")), **q.get("meta", {})}
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            if n % 10 == 0:
                print(f"{n + 1}/{len(jobs)} {time.time() - t0:.0f}s Q {q['question'][:60]!r} gold {str(q['gold'])[:40]!r} "
                      f"pred {pred[:60]!r}", flush=True)
    print("DONE", len(jobs), f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
