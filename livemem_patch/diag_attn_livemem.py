"""LiveMem + LongMemEval-s, SnapKV-style scoring: the question is placed after each chunk of history and the
attention that the question tokens pay to the chunk tokens (frozen backbone attention, softmax, with RoPE) is
the token score. (File derived from diag_qk_livemem.py; the docstring below describes the q.k variant.)

--- q.k variant ---
Can the QUESTION's memory read-vectors (q) pick out the history tokens whose memory keys (k) hold the answer?

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
    model = MemoryQwen3ForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="eager").eval().cuda()
    layers = model.model.layers
    H, D = layers[0].self_attn.mem.num_heads, layers[0].self_attn.mem.head_k_dim
    cap = {}
    sel = {"q": None, "k": None}

    def hook(li):
        def f(mod, inp, out):
            w = out[1]  # [1, heads, T, T] attention probabilities
            if w is None or sel["q"] is None:
                return
            a = w[0].float()[:, sel["q"]][:, :, sel["k"]]  # [heads, nq, nk]
            cap[li] = a.mean(dim=(0, 1)).cpu()  # mean over heads and question tokens -> [nk]
            cap[("max", li)] = a.amax(dim=1).mean(dim=0).cpu()  # per head: max over question tokens; mean over heads
        return f

    for li, layer in enumerate(layers):
        layer.self_attn.attn.register_forward_hook(hook(li))
    L = len(layers)
    CH = 2048

    data = json.load(open(args.data))
    by_type = defaultdict(list)
    for x in data:
        by_type[x["question_type"]].append(x)
    rng = random.Random(0)
    picked = [x for t in sorted(by_type) for x in rng.sample(by_type[t], min(len(by_type[t]), max(1, args.n // len(by_type))))]
    variants = ("mean", "max")
    groups = {"all": list(range(L)), "low": list(range(0, 12)), "mid": list(range(12, 24)), "high": list(range(24, 36))}
    res = defaultdict(list)

    for qi, x in enumerate(picked):
        qtext = x["question"]
        gold_s = {i for i, s_ in enumerate(x["haystack_session_ids"]) if s_ in set(x["answer_session_ids"])}
        sess_scores = defaultdict(list); tok_scores = defaultdict(list); tok_labels = []
        dates = x.get("haystack_dates") or [None] * len(x["haystack_sessions"])
        docs = []
        for si, (sess, date) in enumerate(zip(x["haystack_sessions"], dates)):
            doc = render_session(sess, date)
            docs.append(doc)
            denc = tok(doc, add_special_tokens=False, return_offsets_mapping=True)
            dids, doffs = denc.input_ids[: args.max_session_tokens], denc.offset_mapping[: args.max_session_tokens]
            lab = torch.zeros(len(dids))
            pos = len(f"[Date: {date}]\n") if date else 0
            for t in sess:
                line = f"{t.get('role', '')}: {t.get('content', '')}"
                if t.get("has_answer"):
                    for j, (a_, b_) in enumerate(doffs):
                        if a_ >= pos and b_ <= pos + len(line):
                            lab[j] = 1
                pos += len(line) + 1
            tok_labels.append(lab)
            per = {(g, v): [] for g in groups for v in variants}
            for c0 in range(0, len(dids), CH):
                chunk = dids[c0 : c0 + CH]
                pre = tok.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": "<<<SPLIT>>>"}],
                                              tokenize=False, add_generation_prompt=True)
                head, tail = pre.split("<<<SPLIT>>>")
                h_ids = tok(head, add_special_tokens=False).input_ids
                q_ids = tok("\n\n" + qtext, add_special_tokens=False).input_ids
                t_ids = tok(tail, add_special_tokens=False).input_ids
                ids = torch.tensor([h_ids + chunk + q_ids + t_ids]).cuda()
                sel["k"] = torch.arange(len(h_ids), len(h_ids) + len(chunk)).cuda()
                sel["q"] = torch.arange(len(h_ids) + len(chunk), ids.shape[1]).cuda()
                cap.clear()
                with torch.no_grad():
                    model(input_ids=ids, logits_to_keep=1)
                for g, ls in groups.items():
                    per[(g, "mean")].append(torch.stack([cap[li] for li in ls]).mean(dim=0))
                    per[(g, "max")].append(torch.stack([cap[("max", li)] for li in ls]).mean(dim=0))
            for key, parts in per.items():
                sc = torch.cat(parts)
                sess_scores[key].append(float(sc.topk(min(20, len(sc))).values.mean())); tok_scores[key].append(sc)
                # attention MASS the question puts on the chunk as a whole (the rest goes to the question itself and
                # the template): comparable across chunks, unlike per-token probabilities that are normalised per chunk
                sess_scores[(key[0], key[1] + "_mass")].append(max(float(p.sum()) for p in parts))
                sess_scores[(key[0], key[1] + "_top20sum")].append(max(float(p.topk(min(20, len(p))).values.sum()) for p in parts))
        labels = torch.cat(tok_labels)
        for key, sc in sess_scores.items():
            order = sorted(range(len(sc)), key=lambda i: -sc[i])
            for k in (5, 10):
                res[(*key, f"recall@{k}")].append(len(set(order[:k]) & gold_s) / len(gold_s))
            if key in tok_scores:
                a = auroc(torch.cat(tok_scores[key]).double(), labels)
                if a is not None:
                    res[(*key, "token_auroc")].append(a)
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
    print(f"\n{'layers':>6s} {'pool':>13s} {'recall@5':>9s} {'recall@10':>10s} {'tok_auroc':>10s}")
    for key in [("random", "-"), ("bm25", "-")] + [(g, v + sfx) for g in groups for v in variants for sfx in ("", "_mass", "_top20sum")]:
        g = lambda m: (sum(res[(*key, m)]) / len(res[(*key, m)])) if res.get((*key, m)) else float("nan")
        out[f"{key[0]}|{key[1]}"] = {m: g(m) for m in ("recall@5", "recall@10", "token_auroc")}
        print(f"{str(key[0]):>6s} {key[1]:>13s} {g('recall@5'):9.3f} {g('recall@10'):10.3f} {g('token_auroc'):10.3f}")
    if args.out:
        json.dump(out, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
