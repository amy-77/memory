"""Can a score tell the turn that states the queried fact from the other turns?

For every question, every history token gets a score from several sources; a turn's score is the mean
of its 5 best tokens. Reported per source: how often the fact's turn is ranked first, its mean
reciprocal rank, and the share of the 20 best tokens of the whole history that lie in the fact sentence.

Sources: the memory branch's q.k per layer (question tokens: all / last 1 / 2 / 4), exact token match
weighted by rarity ("lexical"), and cosine of input embeddings ("embed").
"""
import argparse, json, os, sys
from collections import defaultdict

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_select import encode_with_offsets, fact_sentence_tokens
from lact_qwen.dialog import build_query_batch
from lact_qwen.loading import add_model_args, autocast, load_model, load_tokenizer
from lact_qwen.synth import read_jsonl


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--out", default=None)
    add_model_args(p)
    args = p.parse_args()
    tok = load_tokenizer(args)
    model, _ = load_model(args, tok)
    ctx = model.lact_ctx
    emb = model.get_input_embeddings().weight.detach().float()
    stats = defaultdict(lambda: [0.0, 0.0, 0.0, 0])

    def add(name, score, spans, fact_turn, fs):
        """score: [L_hist] for one row."""
        turn = torch.stack([score[s:e].topk(min(5, e - s)).values.mean() for s, e in spans])
        rank = int((turn >= turn[fact_turn]).sum())  # ties count against the fact's turn
        top = score.topk(20).indices
        inside = float(((top >= fs[0]) & (top < fs[1])).float().mean())
        st = stats[name]
        st[0] += rank == 1; st[1] += 1.0 / rank; st[2] += inside; st[3] += 1

    for di, dlg in enumerate(read_jsonl(args.data)[: args.limit]):
        enc, offsets, texts = encode_with_offsets(dlg, tok)
        batch = build_query_batch(enc, dlg["queries"], tok, tok.pad_token_id, 4096, 0)
        rows = [(i, r) for i, r in enumerate(batch.rows) if r["which"] == "answer"]
        n_hist = len(enc.history_ids)
        hist = torch.tensor(enc.history_ids)
        nq = max(len(r["q_ids"]) for r in batch.rows)
        qb = torch.full((len(batch.rows), nq), tok.pad_token_id, dtype=torch.long)
        qmask = torch.zeros(len(batch.rows), nq, dtype=torch.bool)
        for i, r in enumerate(batch.rows):
            qb[i, : len(r["q_ids"])] = torch.tensor(r["q_ids"]); qmask[i, : len(r["q_ids"])] = True
        ctx.query_q, ctx.capture_query, ctx.select = {}, True, None
        ctx.set_segments([(0, nq)], [False])
        with torch.no_grad(), autocast(args):
            model(input_ids=qb.to(args.device), logits_to_keep=1)
        ctx.capture_query = False
        ctx.set_segments(batch.segments, batch.update_flags)
        ctx.query_mask, ctx.query_span, ctx.debug_scores = qmask, None, {}
        ctx.debug_max_layer = 1
        with torch.no_grad(), autocast(args):
            model(input_ids=batch.input_ids.to(args.device), logits_to_keep=1)
        counts = torch.bincount(hist, minlength=emb.shape[0]).float()
        he = torch.nn.functional.normalize(emb[hist.to(emb.device)], dim=-1)
        for i, r in rows:
            q = dlg["queries"][r["query_index"]]
            fs = fact_sentence_tokens(q, enc, offsets, texts)
            if fs is None:
                continue
            for (layer, name), sc in ctx.debug_scores.items():
                add(f"qk_L{layer:02d}_{name}", sc[i, :n_hist].float(), enc.turn_spans, q["fact_turn"], fs)
            qi = torch.tensor(r["q_ids"])
            match = (hist[:, None] == qi[None, :]).any(dim=1).float()
            add("lexical_idf", match / counts[hist].clamp_min(1.0), enc.turn_spans, q["fact_turn"], fs)
            add("lexical_idf2", match / counts[hist].clamp_min(1.0) ** 2, enc.turn_spans, q["fact_turn"], fs)
            # rarity known only from the turns seen so far (streaming): count of the token up to and including its turn
            run = torch.zeros_like(counts); sofar = torch.empty(len(hist))
            for (s_, e_) in enc.turn_spans:
                run += torch.bincount(hist[s_:e_], minlength=emb.shape[0]).float()
                sofar[s_:e_] = run[hist[s_:e_]]
            add("lexical_idf_stream", match / sofar.clamp_min(1.0), enc.turn_spans, q["fact_turn"], fs)
            qe = torch.nn.functional.normalize(emb[qi.to(emb.device)], dim=-1)
            cos = (he @ qe.T).cpu()
            add("embed_max", cos.max(dim=1).values, enc.turn_spans, q["fact_turn"], fs)
            w = 1.0 / counts[qi].clamp_min(1.0)  # question tokens that are rare in the history count more
            add("embed_idf", (cos.clamp_min(0) ** 8 * w[None, :]).sum(dim=1), enc.turn_spans, q["fact_turn"], fs)
            add("embed_idfmax", (cos.clamp_min(0) ** 8 * w[None, :]).max(dim=1).values, enc.turn_spans, q["fact_turn"], fs)
            add("embed_idfmax_stream", (cos.clamp_min(0) ** 8).max(dim=1).values / sofar.clamp_min(1.0), enc.turn_spans, q["fact_turn"], fs)
        ctx.debug_scores = None
        print(f"[{di + 1}] history={n_hist}", flush=True)

    print(f"\n{'source':22s} {'top1':>6s} {'mrr':>6s} {'top20_in_fact':>14s} {'n':>5s}")
    res = {}
    for name in sorted(stats):
        a, b, c, n = stats[name]
        res[name] = {"top1": a / n, "mrr": b / n, "top20_in_fact": c / n, "n": n}
        print(f"{name:22s} {a / n:6.3f} {b / n:6.3f} {c / n:14.3f} {n:5d}")
    if args.out:
        json.dump(res, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
