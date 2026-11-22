"""Needle test for the LiveMem memory and for the write multipliers.

A short session stating a random 5-digit code is put near the start of real LongMemEval sessions (after the
first ~2500 tokens, i.e. behind the 1088-token attention sink), followed by filler sessions up to --lengths
tokens; the question asks for the code. With an 8K attention window the needle is far outside attention, so
the answer can only come from the memory state. Four write policies (one mask directory each):
  all      every token written (released behaviour)
  only     only the needle session written
  without  everything except the needle session written
  none     no history token written (memory empty: equivalent of truncation)
Run from the LiveMem repo root. Output: <out>/prompts.jsonl and <out>/{all,only,without,none}/<sha1>.npy
"""
import argparse, hashlib, json, os, random, sys
import numpy as np
sys.path.insert(0, os.getcwd()); sys.path.insert(0, "tools/data_process")
from eval.materialize import materialize_requests, TOK_PATH
from eval.prompt import MEM_SEP
from transformers import AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=30)
ap.add_argument("--lengths", default="20000,45000")
args = ap.parse_args()
tok = AutoTokenizer.from_pretrained(TOK_PATH)
reqs = materialize_requests(os.environ.get("NEEDLE_SRC", "longmemeval_s_sub"), 10, 130)
rng = random.Random(0)
for p in ("all", "only", "without", "none"):
    os.makedirs(os.path.join(args.out, p), exist_ok=True)
key_of = lambda ids: hashlib.sha1(np.asarray(list(ids), dtype=np.int32).tobytes()).hexdigest()
ntok = lambda s: len(tok(s, add_special_tokens=False).input_ids)
fo = open(os.path.join(args.out, "prompts.jsonl"), "w")
for L in [int(x) for x in args.lengths.split(",")]:
    for j in range(args.n):
        r = reqs[(j * 3) % len(reqs)]
        if os.environ.get("NEEDLE_KIND", "digits") == "word":
            code = rng.choice(["Biscuit", "Pepper", "Mochi", "Waffles", "Juniper", "Maple", "Ziggy", "Clover", "Pickle", "Nutmeg", "Tofu", "Banjo"])
            needle = (f"[Date: 2023/05/01 (Mon) 09:00]\nuser: I just adopted a dog and named her {code}.\n"
                      f"assistant: Congratulations! {code} is a lovely name for your new dog.")
            q = "What is the name of the dog I adopted?"
        else:
            code = str(rng.randint(10000, 99999))
            needle = (f"[Date: 2023/05/01 (Mon) 09:00]\nuser: Please remember this for me: my gym locker code is {code}.\n"
                      f"assistant: Got it. Your gym locker code is {code}. I will remember that.")
            q = "What is my gym locker code?"
        docs, used, pos = [], 0, None
        for d in r.memory_docs:
            if "locker" in d.lower() or " dog" in d.lower():
                continue
            n = ntok(d)
            if pos is None and used >= 2500:
                pos = len(docs); docs.append(needle); used += ntok(needle)
            if used + n > L:
                break
            docs.append(d); used += n
        assert pos is not None
        user_text = r.user_text.replace(r.questions[0], q); assert q in user_text
        messages = [r.messages[0], {"role": "user", "content": f"{MEM_SEP.join(docs)}{MEM_SEP}{user_text}"}]
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        ids, offs = enc.input_ids, enc.offset_mapping
        mem = MEM_SEP.join(docs); m0 = text.find(mem); assert m0 >= 0
        starts = np.array([o[0] for o in offs]); ends = np.array([o[1] for o in offs])
        t = lambda a, b: (int(np.searchsorted(ends, a, side="right")), int(np.searchsorted(starts, b, side="left")))
        ma, mb = t(m0, m0 + len(mem))
        c0 = m0 + sum(len(d) + len(MEM_SEP) for d in docs[:pos]); na, nb = t(c0, c0 + len(needle))
        assert str(code) in tok.decode(ids[na:nb])
        w = {p: np.ones(len(ids), dtype=np.float16) for p in ("all", "only", "without", "none")}
        w["only"][ma:mb] = 0; w["only"][na:nb] = 1
        w["without"][na:nb] = 0
        w["none"][ma:mb] = 0
        k = key_of(ids)
        for p in w:
            np.save(os.path.join(args.out, p, k + ".npy"), w[p])
        fo.write(json.dumps({"id": f"L{L}-{j}", "length": len(ids), "target": L, "code": code, "needle_tokens": [na, nb],
                             "distance_to_end": len(ids) - nb, "key": k, "messages": messages}) + "\n")
fo.close()
print("built", args.n, "prompts per length", args.lengths)
