"""LiveMem + LongMemEval-s: can the QUESTION's memory read-vectors (q) pick out the history tokens whose
memory keys (k) hold the answer?

Per question: (1) the question alone goes through the model once and the per-layer q of its tokens are kept;
(2) every history session goes through the model on its own and the per-layer k of its tokens are kept;
(3) token score = similarity of its k with the question's q (mean over heads), per layer.
A session's score is the mean of its 20 best tokens. Reported per layer and per choice of question tokens:
how often the sessions that hold the answer are among the 5 / 10 best of ~48, and the AUROC of
"token lies in a turn marked has_answer" (0.5 = chance). BM25 on the same questions is printed for reference.

Approximation: sessions are encoded independently (no earlier history in attention or memory).
"""
import argparse, json, math, os, random, sys
from collections import Counter, defaultdict

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from models import MemoryQwen3ForCausalLM  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

SYSTEM = "You are a helpful assistant."


def render_session(turns, date):
    head = f"[Date: {date}]\n" if date else ""
    return head + "\n".join(f"{t.get('role', '')}: {t.get('content', '')}" for t in turns)


def auroc(scores, labels):
    order = torch.argsort(scores)
    ranks = torch.empty_like(order, dtype=torch.float64)
    ranks[order] = torch.arange(1, len(scores) + 1, dtype=torch.float64)
    pos = labels.bool()
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if n1 == 0 or n0 == 0:
        return None
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, default=36)
    ap.add_argument("--max-session-tokens", type=int, default=6000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = MemoryQwen3ForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="sdpa").eval().cuda()
    layers = model.model.layers
    H, D = layers[0].self_attn.mem.num_heads, layers[0].self_attn.mem.head_k_dim
    cap = {}

    def hook(name, li):
        def f(mod, inp, out):
            x = out[0] if isinstance(out, tuple) else out
            x = x[0].float().view(-1, H, D)
            cap[(name, li)] = torch.nn.functional.normalize(x, dim=-1).flatten(1) / math.sqrt(H)  # dot = mean cosine over heads
        return f

    for li, layer in enumerate(layers):
        layer.self_attn.mem.q_conv1d.register_forward_hook(hook("q", li))
        layer.self_attn.mem.k_conv1d.register_forward_hook(hook("k", li))
    L = len(layers)

    def encode(user_text, want):
        text = tok.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_text}],
                                       tokenize=False, add_generation_prompt=True)
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        ids = torch.tensor([enc.input_ids]).cuda()
        with torch.no_grad():
            model(input_ids=ids, logits_to_keep=1)
        c0 = text.find(user_text)
        return text, enc.offset_mapping, c0, torch.stack([cap[(want, li)] for li in range(L)])  # [L, T, H*D]

    data = json.load(open(args.data))
    by_type = defaultdict(list)
    for x in data:
        by_type[x["question_type"]].append(x)
    rng = random.Random(0)
    picked = [x for t in sorted(by_type) for x in rng.sample(by_type[t], min(len(by_type[t]), max(1, args.n // len(by_type))))]
    variants = ("all", "last4", "last1", "gen")
    res = defaultdict(list)

    for qi, x in enumerate(picked):
        qtext = x["question"]
        text, offs, c0, Q = encode(qtext, "q")
        qtok = [j for j, (a, b) in enumerate(offs) if a >= c0 and b <= c0 + len(qtext) and b > a]
        gen = list(range(qtok[-1] + 1, len(offs)))  # template tokens after the question ("<|im_end|> <|im_start|>assistant")
        qsets = {"all": qtok, "last4": qtok[-4:], "last1": qtok[-1:], "gen": gen[-2:]}
        gold_s = {i for i, s in enumerate(x["haystack_session_ids"]) if s in set(x["answer_session_ids"])}
        sess_scores = {(li, v): [] for li in range(L) for v in variants}
        tok_scores = {(li, v): [] for li in range(L) for v in variants}
        tok_labels = []
        dates = x.get("haystack_dates") or [None] * len(x["haystack_sessions"])
        docs = []
        for si, (sess, date) in enumerate(zip(x["haystack_sessions"], dates)):
            doc = render_session(sess, date)
            docs.append(doc)
            dtext, doffs, d0, K = encode(doc, "k")
            keep = [j for j, (a, b) in enumerate(doffs) if a >= d0 and b <= d0 + len(doc) and b > a][: args.max_session_tokens]
            lab = torch.zeros(len(keep))
            pos = d0 + (len(f"[Date: {date}]\n") if date else 0)
            for t in sess:
                line = f"{t.get('role', '')}: {t.get('content', '')}"
                if t.get("has_answer"):
                    for jj, j in enumerate(keep):
                        if doffs[j][0] >= pos and doffs[j][1] <= pos + len(line):
                            lab[jj] = 1
                pos += len(line) + 1
            tok_labels.append(lab)
            Kk = K[:, keep]
            for v in variants:
                sim = torch.einsum("ltd,lqd->ltq", Kk, Q[:, qsets[v]]).amax(dim=-1).cpu()  # [L, T]
                top = sim.topk(min(20, sim.shape[1]), dim=1).values.mean(dim=1)
                for li in range(L):
                    sess_scores[(li, v)].append(float(top[li])); tok_scores[(li, v)].append(sim[li])
        labels = torch.cat(tok_labels)
        for (li, v), sc in sess_scores.items():
            order = sorted(range(len(sc)), key=lambda i: -sc[i])
            for k in (5, 10):
                res[(li, v, f"recall@{k}")].append(len(set(order[:k]) & gold_s) / len(gold_s))
            a = auroc(torch.cat(tok_scores[(li, v)]).double(), labels)
            if a is not None:
                res[(li, v, "token_auroc")].append(a)
        # BM25 reference on the same question
        q = set(tok(qtext, add_special_tokens=False).input_ids)
        dt = [tok(d, add_special_tokens=False).input_ids for d in docs]
        n = len(dt); df = Counter(t for d in dt for t in set(d)); avg = sum(map(len, dt)) / n
        bm = []
        for d in dt:
            c = Counter(d)
            bm.append(sum(math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5)) * c[t] * 2.5 / (c[t] + 1.5 * (0.25 + 0.75 * len(d) / avg)) for t in q if t in c))
        order = sorted(range(n), key=lambda i: -bm[i])
        for k in (5, 10):
            res[("bm25", "-", f"recall@{k}")].append(len(set(order[:k]) & gold_s) / len(gold_s))
        res[("random", "-", "recall@5")].append(5 / n); res[("random", "-", "recall@10")].append(10 / n)
        print(f"[{qi + 1}/{len(picked)}] {x['question_type']} sessions={n}", flush=True)

    out = {}
    print(f"\n{'layer':>6s} {'qtokens':>7s} {'recall@5':>9s} {'recall@10':>10s} {'tok_auroc':>10s}")
    for key in [("random", "-"), ("bm25", "-")] + [(li, v) for v in variants for li in range(L)]:
        g = lambda m: (sum(res[(*key, m)]) / len(res[(*key, m)])) if res.get((*key, m)) else float("nan")
        out[f"{key[0]}|{key[1]}"] = {m: g(m) for m in ("recall@5", "recall@10", "token_auroc")}
        print(f"{str(key[0]):>6s} {key[1]:>7s} {g('recall@5'):9.3f} {g('recall@10'):10.3f} {g('token_auroc'):10.3f}")
    if args.out:
        json.dump(out, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
