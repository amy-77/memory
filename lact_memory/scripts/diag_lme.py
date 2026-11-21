"""LongMemEval-s: can a cheap question-conditioned score find the evidence?

No model forward. For every question the ~50 history sessions (and their turns) are ranked by a score
computed from token ids only, and we report how often the sessions / turns that hold the answer
(`answer_session_ids`, `has_answer`) are among the best ones, per question type.

Scores (unit = a turn or a session; a session's tokens are those of all its turns):
  bm25         BM25 of the unit against the question (Qwen tokens)
  lex_idf      sum over the distinct question tokens found in the unit of their rarity (idf)
  lex_top5     mean of the 5 best token scores, token score = rarity of the question token it equals
  emb_idfmax   like lex_top5 but with cosine of input embeddings instead of exact match (needs --emb)
Rarity comes from a background corpus: the histories of `--bg` other questions (fixed table, streaming-safe).
"""
import argparse, json, math, sys
from collections import Counter, defaultdict

import torch
from transformers import AutoTokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--limit", type=int, default=200)
    p.add_argument("--bg", type=int, default=30)
    p.add_argument("--emb", default=None, help="safetensors file holding model.embed_tokens.weight")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    data = json.load(open(args.data))
    enc = lambda texts: tok(texts, add_special_tokens=False).input_ids

    # background statistics from questions that are not evaluated
    df, n_docs, tf_total = Counter(), 0, Counter()
    for x in data[-args.bg:]:
        for sess in x["haystack_sessions"]:
            for ids in enc([t["content"] for t in sess]):
                df.update(set(ids)); tf_total.update(ids); n_docs += 1
    idf = lambda t: math.log(1.0 + (n_docs - df.get(t, 0) + 0.5) / (df.get(t, 0) + 0.5))
    emb = None
    if args.emb:
        from safetensors import safe_open
        with safe_open(args.emb, "pt") as f:
            key = next(k for k in f.keys() if k.endswith("embed_tokens.weight"))
            emb = torch.nn.functional.normalize(f.get_tensor(key).float(), dim=-1)

    stats = defaultdict(lambda: defaultdict(list))

    def rank_metrics(name, qtype, scores, gold, ks):
        order = sorted(range(len(scores)), key=lambda i: -scores[i])
        for k in ks:
            top = set(order[:k])
            stats[(name, f"recall@{k}")][qtype].append(len(top & gold) / len(gold))
            stats[(name, f"all@{k}")][qtype].append(float(gold <= top))

    for qi, x in enumerate(data[: args.limit]):
        qids = enc([x["question"]])[0]
        qset = set(qids)
        qw = {t: idf(t) for t in qset}
        mx = max(qw.values())
        rar = {t: w / mx for t, w in qw.items()}
        sess_ids, turn_ids, turn_sess, turn_gold = [], [], [], set()
        for si, sess in enumerate(x["haystack_sessions"]):
            ids_list = enc([t["content"] for t in sess])
            flat = []
            for t, ids in zip(sess, ids_list):
                if t.get("has_answer"):
                    turn_gold.add(len(turn_ids))
                turn_ids.append(ids); turn_sess.append(si); flat.extend(ids)
            sess_ids.append(flat)
        gold_s = {i for i, s in enumerate(x["haystack_session_ids"]) if s in set(x["answer_session_ids"])}
        if not gold_s:
            continue
        qt = x["question_type"]
        for unit, docs, gold, ks in (("session", sess_ids, gold_s, (1, 3, 5, 10)), ("turn", turn_ids, turn_gold, (5, 10, 20, 50))):
            if not gold:
                continue
            avg = sum(len(d) for d in docs) / len(docs)
            bm, lex, top5, embs = [], [], [], []
            for d in docs:
                c = Counter(d)
                bm.append(sum(idf(t) * c[t] * 2.5 / (c[t] + 1.5 * (0.25 + 0.75 * len(d) / avg)) for t in qset if t in c))
                lex.append(sum(rar[t] for t in qset if t in c and rar[t] >= 0.3))
                tsc = sorted((rar[t] for t in d if t in rar), reverse=True)[:5]
                top5.append(sum(tsc) / 5.0)
                if emb is not None and d:
                    de, qe = emb[torch.tensor(d)], emb[torch.tensor(qids)]
                    cos = (de @ qe.T).clamp_min(0) ** 8 * torch.tensor([rar[t] for t in qids])[None, :]
                    v = cos.max(dim=1).values
                    embs.append(float(v.topk(min(5, len(d))).values.mean()))
            rank_metrics(f"{unit}/bm25", qt, bm, gold, ks)
            rank_metrics(f"{unit}/lex_idf", qt, lex, gold, ks)
            rank_metrics(f"{unit}/lex_top5", qt, top5, gold, ks)
            if embs:
                rank_metrics(f"{unit}/emb_idfmax", qt, embs, gold, ks)
        if unit == "turn":
            total = sum(len(d) for d in turn_ids)
            stats[("info", "history_tokens")][qt].append(total)
            stats[("info", "gold_turn_token_share")][qt].append(sum(len(turn_ids[i]) for i in turn_gold) / total)
            stats[("info", "n_sessions")][qt].append(len(sess_ids))
        if (qi + 1) % 20 == 0:
            print(f"[{qi + 1}]", flush=True)

    types = sorted({t for v in stats.values() for t in v})
    print(f"\n{'score':24s} {'metric':22s} {'ALL':>7s} " + " ".join(f"{t[:14]:>14s}" for t in types))
    res = {}
    for (name, metric), by in sorted(stats.items()):
        allv = [v for vs in by.values() for v in vs]
        row = {t: (sum(by[t]) / len(by[t]) if by.get(t) else float("nan")) for t in types}
        res[f"{name}|{metric}"] = {"ALL": sum(allv) / len(allv), **row}
        print(f"{name:24s} {metric:22s} {sum(allv) / len(allv):7.3f} " + " ".join(f"{row[t]:14.3f}" for t in types))
    if args.out:
        json.dump(res, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
