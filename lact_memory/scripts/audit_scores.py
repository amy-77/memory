"""Audit of the negative result "the question's q.k cannot find its answer" on the LaCT dialogue model.

For each question of a synthetic dialogue every history token gets a score; a turn's score is the mean of its
5 best tokens; we report how often the turn that states the queried fact is ranked first (ties count against
it) and its mean reciprocal rank. Checks, in this order:

  controls   answer_match (tokens equal to the answer's tokens: must be ~1.0, validates labels and the metric)
             random (must be ~1/num_turns)
  causal     leave-one-turn-out: skip the write of one turn and measure how much the answer's log-prob drops.
             The fact's turn must be the most important one if the memory really stores the fact there.
  memory     question read-vector vs token key, per layer: raw cosine and the LaCT feature-map kernel
             <phi(q), phi(k)>, phi(x) = silu(w0 x) * (w2 x); question alone vs the vector actually used in
             context for the first answer token.
  attention  frozen base model, the question placed after each turn: attention mass / best tokens.
"""
import argparse, json, os, random, sys
from collections import defaultdict

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_select import encode_with_offsets, fact_sentence_tokens
from lact_qwen.dialog import build_query_batch, score_rows
from lact_qwen.loading import add_model_args, autocast, load_model, load_tokenizer
from lact_qwen.synth import read_jsonl


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--limit", type=int, default=12)
    p.add_argument("--out", default=None)
    p.add_argument("--no-causal", action="store_true")
    add_model_args(p)
    args = p.parse_args()
    tok = load_tokenizer(args)
    model, _ = load_model(args, tok)
    ctx = model.lact_ctx
    from transformers import AutoModelForCausalLM
    base = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="eager").eval().to(args.device)
    nl = len(base.model.layers)
    acc, sel = {}, {}

    def hook(li):
        def f(mod, inp, out):
            w = out[1]
            if w is None or li < nl // 3:
                return
            a = w[0].float()[:, sel["q"]][:, :, sel["k"]].mean(dim=(0, 1))
            acc["sum"] = acc.get("sum", 0) + a; acc["n"] = acc.get("n", 0) + 1
        return f

    for li, layer in enumerate(base.model.layers):
        layer.self_attn.register_forward_hook(hook(li))

    stats = defaultdict(lambda: [0.0, 0.0, 0])
    rng = random.Random(0)

    def add(name, turn_scores, fact_turn):
        t = torch.as_tensor(turn_scores, dtype=torch.float64)
        rank = int((t >= t[fact_turn]).sum())
        st = stats[name]; st[0] += rank == 1; st[1] += 1.0 / rank; st[2] += 1

    def turn_agg(score, spans):
        return [float(score[s:e].topk(min(5, e - s)).values.mean()) for s, e in spans]

    for di, dlg in enumerate(read_jsonl(args.data)[: args.limit]):
        enc, offsets, texts = encode_with_offsets(dlg, tok)
        batch = build_query_batch(enc, dlg["queries"], tok, tok.pad_token_id, 4096, 0)
        rows = [(i, r) for i, r in enumerate(batch.rows) if r["which"] == "answer"]
        n_hist, total = len(enc.history_ids), batch.input_ids.shape[1]
        hist = torch.tensor(enc.history_ids)
        spans = enc.turn_spans
        # question alone
        nq = max(len(r["q_ids"]) for r in batch.rows)
        qb = torch.full((len(batch.rows), nq), tok.pad_token_id, dtype=torch.long)
        qmask = torch.zeros(len(batch.rows), nq, dtype=torch.bool)
        for i, r in enumerate(batch.rows):
            qb[i, : len(r["q_ids"])] = torch.tensor(r["q_ids"]); qmask[i, : len(r["q_ids"])] = True
        ctx.select, ctx.debug_scores, ctx.query_q, ctx.capture_query = None, None, {}, True
        ctx.set_segments([(0, nq)], [False])
        with torch.no_grad(), autocast(args):
            model(input_ids=qb.to(args.device), logits_to_keep=1)
        ctx.capture_query = False
        # full forward with all writes: memory-side scores + reference log-probs
        ctx.set_segments(batch.segments, batch.update_flags)
        ctx.query_mask, ctx.query_span, ctx.debug_scores = qmask, None, {}
        ctx.debug_tailpos = torch.tensor([n_hist + len(r["q_ids"]) - 1 for r in batch.rows])
        ctx.debug_span = (n_hist, n_hist + nq)
        keep = total - min(r["answer_span"][0] for r in batch.rows) + 1
        with torch.no_grad(), autocast(args):
            logits = model(input_ids=batch.input_ids.to(args.device), logits_to_keep=keep).logits
        ref = score_rows(logits, batch, offset=total - keep)
        dbg, ctx.debug_scores = ctx.debug_scores, None
        # causal importance of every turn
        drop = None
        if not args.no_causal:
            drop = torch.zeros(len(batch.rows), len(spans))
            for j in range(len(spans)):
                flags = list(batch.update_flags); flags[j] = False
                ctx.set_segments(batch.segments, flags)
                with torch.no_grad(), autocast(args):
                    lg = model(input_ids=batch.input_ids.to(args.device), logits_to_keep=keep).logits
                abl = score_rows(lg, batch, offset=total - keep)
                for i in range(len(batch.rows)):
                    drop[i, j] = ref[i]["logprob"] - abl[i]["logprob"]
        for i, r in rows:
            q = dlg["queries"][r["query_index"]]
            if q["category"] != "retention":
                continue
            ft = q["fact_turn"]
            a_ids = set(tok(" " + q["answer"], add_special_tokens=False).input_ids) | set(tok(q["answer"], add_special_tokens=False).input_ids)
            match = torch.tensor([float(t in a_ids) for t in enc.history_ids])
            add("control/answer_match", [float(match[s:e].sum()) for s, e in spans], ft)
            add("control/random", [rng.random() for _ in spans], ft)
            if drop is not None:
                add("causal/leave_one_turn_out", drop[i].tolist(), ft)
                stats["causal/answered_correctly"][0] += ref[i]["exact"]; stats["causal/answered_correctly"][1] += ref[i]["exact"]; stats["causal/answered_correctly"][2] += 1
            groups = defaultdict(list)
            for (layer, name), sc in dbg.items():
                add(f"memory/{name}/L{layer:02d}", turn_agg(sc[i, :n_hist].float(), spans), ft)
                groups[name].append(sc[i, :n_hist].float())
            for name, lst in groups.items():
                z = torch.stack([(x - x.mean()) / (x.std() + 1e-6) for x in lst]).mean(dim=0)
                add(f"memory/{name}/all_layers_z", turn_agg(z, spans), ft)
            # frozen attention, question after each turn
            mass, best = [], []
            for (s, e) in spans:
                ids = torch.tensor([enc.history_ids[s:e] + r["q_ids"]]).to(args.device)
                sel["k"] = torch.arange(0, e - s, device=args.device); sel["q"] = torch.arange(e - s, ids.shape[1], device=args.device)
                acc.clear()
                with torch.no_grad():
                    base(input_ids=ids, logits_to_keep=1)
                a = acc["sum"] / acc["n"]
                mass.append(float(a.sum())); best.append(float(a.topk(min(5, e - s)).values.mean()))
            add("attention/mass", mass, ft); add("attention/top5_tokens", best, ft)
        print(f"[{di + 1}] history={n_hist} turns={len(spans)}", flush=True)

    res = {}
    print(f"\n{'score':44s} {'top1':>6s} {'mrr':>6s} {'n':>5s}")
    for name in sorted(stats):
        a, bq, n = stats[name]
        res[name] = {"top1": a / max(n, 1), "mrr": bq / max(n, 1), "n": n}
    best_layer = {}
    for name, v in res.items():
        if name.startswith("memory/") and "/L" in name:
            fam = name.rsplit("/", 1)[0]
            if fam not in best_layer or v["top1"] > best_layer[fam][1]["top1"]:
                best_layer[fam] = (name, v)
    show = [n for n in sorted(res) if not (n.startswith("memory/") and "/L" in n)]
    for name in show:
        v = res[name]; print(f"{name:44s} {v['top1']:6.3f} {v['mrr']:6.3f} {v['n']:5d}")
    for fam, (name, v) in sorted(best_layer.items()):
        print(f"{'best layer: ' + name:44s} {v['top1']:6.3f} {v['mrr']:6.3f} {v['n']:5d}")
    if args.out:
        json.dump(res, open(args.out, "w"), indent=1)
    print("AUDIT_DONE")


if __name__ == "__main__":
    main()
