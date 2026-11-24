"""Load MemT-1.0-Nano with HF transformers (its own modeling_memt.py) and check memory write/read.
1. the model card's kitten example through the official session API
2. our protocol on LoCoMo conv-26: commit the history as a document (2048-token chunks), answer 3 questions from memory
"""
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import adapters
from bench_nano_hf import post_load_compat, transformers_compat

path, data = sys.argv[1], sys.argv[2]
t0 = time.time()
transformers_compat()
tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, dtype=torch.bfloat16,
                                             attn_implementation="sdpa").to("cuda:0").eval()
post_load_compat(model)
print(f"loaded in {time.time() - t0:.0f}s, {torch.cuda.memory_allocated() / 2**30:.1f} GiB", flush=True)

with torch.no_grad():
    s = model.create_session(tok)
    s.chat("I have a three-month-old kitten named memmy who loves eating small shrimp and is excellent at remembering "
           "things.", enable_memory=True, enable_thinking=False, max_new_tokens=128)
    print("KITTEN:", s.chat("What are my kitten's name, age, favorite food, and special trait?", enable_thinking=False,
                            max_new_tokens=128), flush=True)
    s.close()

    item = adapters.BENCHMARKS["locomo"](data, tok)[0]
    doc = tok.apply_chat_template([{"role": "user", "content": item["history"]}], tokenize=False, add_generation_prompt=False,
                                  enable_thinking=False, preserve_thinking=True)
    ids = tok.encode(doc, add_special_tokens=False)
    t1 = time.time()
    state = model.commit_memory(torch.tensor([ids], device="cuda:0"), None, document=True)
    print(f"LoCoMo conv-26: wrote {len(ids)} tokens in {time.time() - t1:.0f}s", flush=True)
    for q in item["questions"][:3]:
        prompt = tok.apply_chat_template(q["memory"], tokenize=False, add_generation_prompt=True, enable_thinking=False,
                                         preserve_thinking=True)
        x = torch.tensor([tok.encode(prompt, add_special_tokens=False)], device="cuda:0")
        t2 = time.time()
        out = model.generate(x, attention_mask=torch.ones_like(x), memory_state=state, max_new_tokens=64, do_sample=False)
        ans = tok.decode(out[0, x.shape[1]:], skip_special_tokens=True)
        print(f"Q {q['question']!r} gold {q['gold']!r} -> {ans.strip()[:150]!r} ({time.time() - t2:.0f}s)", flush=True)
print("NANO_SMOKE_DONE", flush=True)
