"""Recall on the synthetic dialogues with selective writing and scored replay (no training).

Same protocol as eval_synth.py: the dialogue is one sequence, every turn is one write, attention
sees the last --window tokens, and the question + answer form a final read-only segment. New here:
the question is known while the history is written (it sits in the same forward), so its read
vectors can score every history token (see lact_qwen/select.py). The answer tokens are masked out
of the scoring, so nothing about the answer leaks into the write.

    python scripts/eval_select.py --data D.jsonl --memory CKPT --model Qwen/Qwen3-0.6B --window 2048 --sinks 4 \
        --select query --tau 0.5 --cap 10 [--seg-gain] [--replay scored_kv --replay-keep 16 --replay-samples 256]

`--select oracle` puts the weight on the sentence that states the queried fact (upper bound for
any scoring rule). The printed "enrichment" is the mean weight on that sentence divided by the
mean weight of the turn it is in: 1.0 means the rule does not find the fact.
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lact_qwen.dialog import build_query_batch, score_rows
from lact_qwen.dialog import EncodedDialogue
from lact_qwen.loading import add_model_args, autocast, load_model, load_tokenizer
from lact_qwen.modeling import set_replay
from lact_qwen.select import SELECT_MODES, SelectConfig, lexical_weights, rarity
from lact_qwen.synth import read_jsonl, turn_text

# Turn-level hard selection: only the --turn-k best-scored turns are written (the others are skipped entirely).
# attn: attention mass of the frozen base model, question placed after each turn; qk: cosine between the read
# vector at the last question token (taken from a first pass with every turn written) and the keys of layer
# --turn-layer; oracle: the turn that states the fact scores highest; random: control.
TURN_MODES = ("turn_attn", "turn_qk", "turn_oracle", "turn_random", "turn_evidence")


def encode_with_offsets(dialogue, tokenizer):
    """Token ids, turn spans, and for every turn the character offsets of its tokens."""
    ids, spans, offsets, texts = [], [], [], []
    for i, body in enumerate(dialogue["turns"]):
        text = turn_text(i, body)
        enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        spans.append((len(ids), len(ids) + len(enc.input_ids)))
        ids.extend(enc.input_ids)
        offsets.append(enc.offset_mapping)
        texts.append(text)
    return EncodedDialogue(ids, spans), offsets, texts


def fact_sentence_tokens(q, enc, offsets, texts):
    """Token range [a, b) of the sentence in the fact's turn that contains the answer value."""
    t = q["fact_turn"]
    text = texts[t]
    pos = text.find(q["answer"])
    if pos < 0:
        return None
    start = max(text.rfind(". ", 0, pos), text.rfind("\n", 0, pos)) + 1
    end = text.find(".", pos)
    end = len(text) if end < 0 else end + 1
    toks = [j for j, (a, b) in enumerate(offsets[t]) if a < end and b > start]
    if not toks:
        return None
    base = enc.turn_spans[t][0]
    vend = pos + len(q["answer"])
    vt = [j for j, (a, b) in enumerate(offsets[t]) if a < vend and b > pos]
    return base + toks[0], base + toks[-1] + 1, base + vt[0], base + vt[-1] + 1


def distance_bucket(dist, window):
    if window <= 0:
        return "full_attention"
    return "inside_window" if dist < window else "beyond_window"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--max-segment", type=int, default=4096)
    p.add_argument("--min-write-tokens", type=int, default=0)
    p.add_argument("--out", default=None)
    p.add_argument("--select", default="none", choices=SELECT_MODES + ("lexical",) + TURN_MODES,
                   help="'lexical': boost history tokens that equal a rare token of the question (and the tokens after them); "
                   "computed once from token ids, the same weights are used in every layer")
    p.add_argument("--boost", type=float, default=10.0, help="--select lexical: weight on a full hit")
    p.add_argument("--min-rarity", type=float, default=0.5, help="--select lexical: ignore question tokens less rare than this")
    p.add_argument("--idf", default="history", help="--select lexical: where token rarity comes from: 'history' (this dialogue, "
                   "not available when streaming) or a jsonl of dialogues used as a fixed background corpus")
    p.add_argument("--tau", type=float, default=0.5)
    p.add_argument("--cap", type=float, default=10.0)
    p.add_argument("--floor", type=float, default=0.0)
    p.add_argument("--topk-frac", type=float, default=0.1)
    p.add_argument("--pool", type=int, default=1)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--agnostic-reduce", default="max", choices=["max", "sum"])
    p.add_argument("--agnostic-tau", type=float, default=0.05)
    p.add_argument("--seg-gain", action="store_true")
    p.add_argument("--seg-tau", type=float, default=0.5)
    p.add_argument("--seg-cap", type=float, default=4.0)
    p.add_argument("--score-layer", type=int, default=-1)
    p.add_argument("--query-last", type=int, default=0, help="score with only the last N question tokens (0 = all)")
    p.add_argument("--pool-mode", default="center", choices=["center", "forward"])
    p.add_argument("--keep", type=float, default=0.0, help="SnapKV-style hard selection: share of each turn's tokens written")
    p.add_argument("--probe-src", default=None, choices=["query", "bank", "both"], help="--replay pseudo: rehearsal probes")
    p.add_argument("--pseudo-target", default=None, choices=["delta", "full"], help="--replay pseudo: what is rehearsed")
    p.add_argument(
        "--query-source", default="tail", choices=["tail", "alone", "prefix"],
        help="where the question's read vectors come from: 'tail' = the question at the end of the same forward "
        "(reference: it has seen the end of the history); 'alone' = a forward of the question by itself before "
        "the history, reused for every turn; 'prefix' = the question is put in front of every turn",
    )
    p.add_argument("--turn-k", type=int, default=4, help="turn_* modes: number of turns written")
    p.add_argument("--bank-score", default="lr", choices=["lr", "relevance", "layer_qk"], help="turn_* + --replay pseudo with bank probes: "
                   "'layer_qk' = every layer banks the keys its own question read vectors score highest; "
                   "which keys of a written turn enter the bank: 'lr' = learned write rate (question-agnostic), 'relevance' = "
                   "the question's per-token score (attention for turn_attn / turn_oracle / turn_random, layer q.k for turn_qk)")
    p.add_argument("--protect", action="store_true", help="protected write: project every write onto the null space of the "
                   "banked keys (needs --replay pseudo with --probe-src bank|both; --replay-lr-scale 0 = protection only)")
    p.add_argument("--protect-ridge", type=float, default=None)
    p.add_argument("--probe-ctx", action="store_true", help="--replay pseudo: question probes = its read vectors in context "
                   "(from a first pass with every turn written) instead of the question alone")
    p.add_argument("--qk-tokens", default="last1", choices=["last1", "all", "mean"], help="turn_qk: question tokens whose in-context "
                   "read vectors score the keys: the last one, the best-matching of all, or the mean over all")
    p.add_argument("--qk-source", default="full", choices=["full", "turn"], help="where the question read vectors for q.k come from: "
                   "'full' = one pass over the WHOLE dialogue with every turn written, question after it (offline: uses later turns); "
                   "'turn' = streaming: the question is appended to each turn on its own, memory empty, nothing written; "
                   "the turn's keys are scored by the question tokens read right after that turn")
    p.add_argument("--write-thresh", type=float, default=None, help="turn_* modes, STREAMING decision: write a turn iff its own score "
                   ">= this threshold (decided when the turn arrives, no ranking against later turns); replaces --turn-k. "
                   "Calibrate on a dev set with --dump-turn-scores")
    p.add_argument("--dump-turn-scores", default=None, help="append one json line per question: turn scores and the fact turn")
    p.add_argument("--delta-erase", type=float, default=0.0, help="erase-then-write at inference: target = v - a * (current "
                   "read-out at k); 0 = the trained LaCT rule")
    p.add_argument("--key-select", default="off", choices=["off", "layer", "random"], help="turn_* modes: inside every written turn, "
                   "each layer writes only its own --key-frac best keys, scored with ITS OWN in-context question read vectors "
                   "(--qk-tokens); 'random' = the same share of random tokens per layer (control)")
    p.add_argument("--key-frac", type=float, default=0.25)
    p.add_argument("--key-pool", type=int, default=7, help="--key-select layer: max-pool the token scores over this window first (SnapKV)")
    p.add_argument("--per-layer", default="off", choices=["off", "own", "confident"], help="turn_qk: 'own' = every layer picks its own "
                   "--turn-k turns with its own q.k (no layer is chosen, nothing is merged); 'confident' = the same, but a layer whose "
                   "turn scores are flat (cannot tell the turns apart) writes every turn. --turn-layer is then ignored")
    p.add_argument("--confident-z", type=float, default=2.5, help="--per-layer confident: a layer selects only if its best turn "
                   "is at least this many standard deviations above its mean turn score")
    p.add_argument("--turn-layer", type=int, default=19, help="turn_qk: layer whose keys are scored")
    p.add_argument("--oracle-high", type=float, default=10.0, help="--select oracle: weight on the fact sentence")
    p.add_argument("--oracle-low", type=float, default=0.1, help="--select oracle: weight elsewhere in that turn")
    p.add_argument("--replay-keep", type=int, default=None, help="scored_kv: pairs kept per written segment")
    p.add_argument("--replay-bank", type=int, default=None, help="scored_kv: bank size")
    p.add_argument("--replay-lr-scale", type=float, default=None, help="scored_kv: rate multiplier of replayed pairs")
    add_model_args(p)
    args = p.parse_args()

    tok = load_tokenizer(args)
    model, mem_cfg = load_model(args, tok)
    ctx = getattr(model, "lact_ctx", None)
    extra = {
        "keep_per_segment": args.replay_keep,
        "bank_size": args.replay_bank,
        "kv_lr_scale": args.replay_lr_scale,
        "probe_src": args.probe_src,
        "protect": True if args.protect else None,
        "protect_ridge": args.protect_ridge,
        "pseudo_target": args.pseudo_target,
    }
    extra = {k: v for k, v in extra.items() if v is not None}
    if extra and ctx is not None:
        set_replay(model, **extra)
    if args.delta_erase:
        for mod in model.modules():
            if hasattr(mod, "update_cfg"):
                mod.update_cfg.delta_erase = args.delta_erase
    lexical = args.select == "lexical"
    turn_mode = args.select if args.select in TURN_MODES else None
    if turn_mode:
        args.select = "none"
    base, attn_acc, attn_sel = None, {}, {}
    if turn_mode == "turn_attn":
        from transformers import AutoModelForCausalLM
        base = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="eager").eval().to(args.device)
        n_base = len(base.model.layers)

        def _hook(li):
            def f(mod, inp, out):
                if out[1] is None or li < n_base // 3:
                    return
                a = out[1][0].float()[:, attn_sel["q"]][:, :, attn_sel["k"]].mean(dim=(0, 1))
                attn_acc["sum"] = attn_acc.get("sum", 0) + a
            return f

        for li, layer in enumerate(base.model.layers):
            layer.self_attn.register_forward_hook(_hook(li))
    import random as _random
    rng = _random.Random(0)
    hit = [0, 0]
    layer_stat = [0, 0, 0]
    written = [0, 0]  # turns written / turns seen (turn_* modes)
    key_stat = [0.0, 0]  # per-layer key selection: share of the fact sentence's tokens a layer keeps (mean over layers and rows)
    key_gen = torch.Generator().manual_seed(0)
    n_layers = len(model.model.layers) if hasattr(model, "model") else 0  # per-layer selection: layers that selected, of those how many kept the fact turn, layers in total
    corpus_counts = None
    if lexical and args.idf != "history":
        corpus_counts = torch.zeros(len(tok), dtype=torch.float32)
        for d in read_jsonl(args.idf)[:200]:
            for i_, body in enumerate(d["turns"]):
                corpus_counts += torch.bincount(torch.tensor(tok(turn_text(i_, body), add_special_tokens=False).input_ids), minlength=len(tok)).float()
    sel = SelectConfig(
        mode="oracle" if lexical else args.select, tau=args.tau, cap=args.cap, floor=args.floor, topk_frac=args.topk_frac, pool=args.pool,
        alpha=args.alpha, agnostic_reduce=args.agnostic_reduce, agnostic_tau=args.agnostic_tau,
        seg_gain=args.seg_gain, seg_tau=args.seg_tau, seg_cap=args.seg_cap, score_layer=args.score_layer, keep=args.keep,
        query_last=args.query_last, pool_mode=args.pool_mode,
    )
    if ctx is not None:
        ctx.select = sel
    dialogues = read_jsonl(args.data)[: args.limit]

    records = []

    def run(batch, n_hist, fact_tok, fact_turn_span, rows_q, qmask=None, oracle=None):
        """One forward. fact_tok / fact_turn_span: per row, token ranges of the fact sentence and of its turn."""
        total = batch.input_ids.shape[1]
        if ctx is not None:
            ctx.set_segments(batch.segments, batch.update_flags)
            ctx.last_weights, ctx.last_gains = {}, {}
            ctx.query_span, ctx.query_mask = (None, qmask) if args.query_source != "tail" else ((n_hist, total), qmask)
            ctx.oracle_weight = oracle
        keep = total - min(r["answer_span"][0] for r in batch.rows) + 1
        with torch.no_grad(), autocast(args):
            logits = model(input_ids=batch.input_ids.to(args.device), logits_to_keep=keep).logits
        scored = score_rows(logits, batch, offset=total - keep)
        enrich, gain_fact, venrich = [None] * len(batch.rows), [None] * len(batch.rows), [None] * len(batch.rows)
        if ctx is not None and ctx.last_weights:
            wl = torch.stack([w.float().cpu() for w in ctx.last_weights.values()]).mean(dim=0)
            for i in range(len(batch.rows)):
                if fact_tok[i] is None:
                    continue
                (a_, b_, va_, vb_), (ts, te) = fact_tok[i], fact_turn_span[i]
                turn_mean = wl[i, ts:te].mean().clamp_min(1e-9)
                enrich[i] = float(wl[i, a_:b_].mean() / turn_mean)
                venrich[i] = float(wl[i, va_:vb_].mean() / turn_mean)
        if ctx is not None and ctx.last_gains:
            gl = torch.stack([g.float().cpu() for g in ctx.last_gains.values()]).mean(dim=0)
            for i in range(len(batch.rows)):
                x = fact_turn_span[i][0]
                j = next((j for j, (s_, e_) in enumerate(batch.segments) if s_ <= x < e_), None)
                if j is not None:
                    gain_fact[i] = float(gl[i, j])
        out = []
        for i, (sc, r) in enumerate(zip(scored, batch.rows)):
            out.append({"query_index": sc["query_index"], "which": sc["which"], "exact": sc["exact"],
                        "logprob": sc["logprob"], "enrichment": enrich[i], "fact_gain": gain_fact[i], "venrich": venrich[i],
                        "distance_tokens": n_hist - fact_turn_span[i][1]})
        return out

    for di, dlg in enumerate(dialogues):
        enc, offsets, texts = encode_with_offsets(dlg, tok)
        batch = build_query_batch(enc, dlg["queries"], tok, tok.pad_token_id, args.max_segment, args.min_write_tokens)
        n_hist = len(enc.history_ids)
        total = batch.input_ids.shape[1]
        queries = [dlg["queries"][r["query_index"]] for r in batch.rows]
        fact_tok = [fact_sentence_tokens(q, enc, offsets, texts) for q in queries]
        turn_span = [enc.turn_spans[q["fact_turn"]] for q in queries]
        if ctx is not None:
            ctx.query_q, ctx.query_prefix, ctx.capture_query = {}, 0, False

        if turn_mode:
            spans = enc.turn_spans
            assert list(batch.segments[: len(spans)]) == [tuple(x) for x in spans], "one written segment per turn expected"
            dbg, span_all, nq_max = None, None, max(len(r["q_ids"]) for r in batch.rows)
            if args.probe_ctx and turn_mode != "turn_qk":
                ctx.set_segments(batch.segments, batch.update_flags)
                ctx.query_mask, ctx.query_span, ctx.query_q = None, None, {}
                ctx.capture_span, ctx.span_q = (n_hist, n_hist + nq_max), {}
                with torch.no_grad(), autocast(args):
                    model(input_ids=batch.input_ids.to(args.device), logits_to_keep=1)
                span_all, ctx.capture_span, ctx.span_q = ctx.span_q, None, {}
            dbg_all = {}
            need_qk = turn_mode == "turn_qk" or args.key_select == "layer" or args.bank_score == "layer_qk"
            if need_qk and args.qk_source == "turn":
                # streaming q.k: for every turn, [turn + question] alone (no history, no writes); one forward per turn,
                # batched over the questions (right-padded, so the turn's tokens do not see the padding)
                nq_max, nrows, total = max(len(r["q_ids"]) for r in batch.rows), len(batch.rows), batch.input_ids.shape[1]
                for (s_, e_) in spans:
                    nt = e_ - s_
                    xb = torch.full((nrows, nt + nq_max), tok.pad_token_id, dtype=torch.long)
                    qv = torch.zeros(nrows, nq_max, dtype=torch.bool)
                    for ii, r in enumerate(batch.rows):
                        xb[ii, :nt] = torch.tensor(enc.history_ids[s_:e_]); xb[ii, nt : nt + len(r["q_ids"])] = torch.tensor(r["q_ids"])
                        qv[ii, : len(r["q_ids"])] = True
                    ctx.set_segments([(0, nt), (nt, nt + nq_max)], [False, False])
                    ctx.capture_span, ctx.span_q = (nt, nt + nq_max), {}
                    ctx.capture_key_pos, ctx.key_store = torch.arange(nt), {}
                    with torch.no_grad(), autocast(args):
                        model(input_ids=xb.to(args.device), logits_to_keep=1)
                    hh = None
                    for l, qq in ctx.span_q.items():
                        kk = ctx.key_store[l]
                        hh = qq.shape[0] // nrows
                        qn = F.normalize(qq.float().cpu(), dim=-1).view(nrows, hh, nq_max, -1)
                        kn = F.normalize(kk.float(), dim=-1).view(nrows, hh, nt, -1)
                        simm = torch.einsum("bhtd,bhqd->bhtq", kn, qn).masked_fill(~qv[:, None, None, :], float("-inf"))
                        sc_l = simm.amax(dim=-1).mean(dim=1)  # [nrows, nt]
                        if l not in dbg_all:
                            dbg_all[l] = torch.zeros(nrows, total)
                        dbg_all[l][:, s_:e_] = sc_l
                    ctx.capture_span, ctx.span_q, ctx.capture_key_pos, ctx.key_store = None, {}, None, {}
                dbg = dbg_all.get(args.turn_layer)
            elif need_qk:
                nq = max(len(r["q_ids"]) for r in batch.rows)
                qb = torch.full((len(batch.rows), nq), tok.pad_token_id, dtype=torch.long)
                qm = torch.zeros(len(batch.rows), nq, dtype=torch.bool)
                for i, r in enumerate(batch.rows):
                    qb[i, : len(r["q_ids"])] = torch.tensor(r["q_ids"]); qm[i, : len(r["q_ids"])] = True
                ctx.capture_query = True
                ctx.set_segments([(0, nq)], [False])
                with torch.no_grad(), autocast(args):
                    model(input_ids=qb.to(args.device), logits_to_keep=1)
                ctx.capture_query = False
                # first pass, every turn written: the read vector the model uses at the last question token
                ctx.set_segments(batch.segments, batch.update_flags)
                ctx.query_mask, ctx.query_span, ctx.debug_scores = qm, None, {}
                ctx.debug_tailpos = torch.tensor([n_hist + len(r["q_ids"]) - 1 for r in batch.rows])
                ctx.debug_span = (n_hist, n_hist + nq)
                if args.probe_ctx:
                    ctx.capture_span, ctx.span_q = (n_hist, n_hist + nq_max), {}
                _rmode = model.lact_cfg.replay.mode
                model.lact_cfg.replay.mode = "none"  # the scoring pass writes every turn plainly (no rehearsal / protection)
                with torch.no_grad(), autocast(args):
                    model(input_ids=batch.input_ids.to(args.device), logits_to_keep=1)
                model.lact_cfg.replay.mode = _rmode
                if args.probe_ctx:
                    span_all, ctx.capture_span, ctx.span_q = ctx.span_q, None, {}
                dbg_all = {l: t for (l, name), t in ctx.debug_scores.items() if name == "cos_incontext_" + args.qk_tokens}
                dbg, ctx.debug_scores, ctx.debug_tailpos = dbg_all.get(args.turn_layer), None, None
                ctx.debug_span = None
                ctx.query_q = {}
            results = []
            for i, r in enumerate(batch.rows):
                ft = queries[i]["fact_turn"]
                tokrel = torch.zeros(1, batch.input_ids.shape[1])
                want_attn_tok = args.bank_score == "relevance" and turn_mode != "turn_qk"
                if turn_mode == "turn_qk" and args.bank_score == "relevance":
                    tokrel[0, :n_hist] = dbg[i, :n_hist].float()
                if want_attn_tok and base is None:
                    raise SystemExit("--bank-score relevance needs --select turn_attn or turn_qk")
                ev = set(queries[i].get("evidence_turns") or [ft])
                if turn_mode == "turn_evidence":  # upper bound: write exactly the turns that hold the evidence (labels)
                    sc = [1.0 if j in ev else 0.0 for j in range(len(spans))]
                elif turn_mode == "turn_oracle":
                    sc = [1.0 if j == ft else rng.random() * 0.5 for j in range(len(spans))]
                elif turn_mode == "turn_random":
                    sc = [rng.random() for _ in spans]
                elif turn_mode == "turn_qk":
                    sc = [float(dbg[i, s_:e_].float().topk(min(5, e_ - s_)).values.mean()) for s_, e_ in spans]
                else:
                    sc = []
                    for s_, e_ in spans:
                        ids = torch.tensor([enc.history_ids[s_:e_] + r["q_ids"]]).to(args.device)
                        attn_sel["k"] = torch.arange(0, e_ - s_, device=args.device)
                        attn_sel["q"] = torch.arange(e_ - s_, ids.shape[1], device=args.device)
                        attn_acc.clear()
                        with torch.no_grad():
                            base(input_ids=ids, logits_to_keep=1)
                        sc.append(float(attn_acc["sum"].sum()))
                        if want_attn_tok:
                            tokrel[0, s_:e_] = attn_acc["sum"].cpu()
                k_i = len(ev) if args.turn_k < 0 else args.turn_k  # --turn-k -1: as many turns as there are evidence turns
                if turn_mode == "turn_evidence":
                    top = set(ev)
                elif args.write_thresh is not None:
                    top = {j for j in range(len(spans)) if sc[j] >= args.write_thresh}
                else:
                    top = set(sorted(range(len(spans)), key=lambda j: -sc[j])[:k_i])
                written[0] += len(top); written[1] += len(spans)
                if args.dump_turn_scores:
                    with open(args.dump_turn_scores, "a") as fd:
                        fd.write(json.dumps({"dialogue": dlg["id"], "query_index": r["query_index"], "category": queries[i]["category"],
                                             "fact_turn": ft, "scores": sc}) + "\n")
                flags = [j in top for j in range(len(spans))] + list(batch.update_flags[len(spans):])
                ctx.layer_update_flags = None
                if turn_mode == "turn_qk" and args.per_layer != "off":
                    lf, nsel, nhit = {}, 0, 0
                    for l, t in dbg_all.items():
                        scl = torch.tensor([float(t[i, s_:e_].float().topk(min(5, e_ - s_)).values.mean()) for s_, e_ in spans])
                        z = float((scl.max() - scl.mean()) / (scl.std() + 1e-9))
                        if args.per_layer == "confident" and z < args.confident_z:
                            lf[l] = [True] * len(spans) + list(batch.update_flags[len(spans):])   # cannot tell: write all
                            continue
                        tl = set(scl.topk(min(args.turn_k, len(spans))).indices.tolist())
                        lf[l] = [j in tl for j in range(len(spans))] + list(batch.update_flags[len(spans):])
                        nsel += 1; nhit += ft in tl
                    ctx.layer_update_flags = lf
                    layer_stat[0] += nsel; layer_stat[1] += nhit; layer_stat[2] += len(dbg_all)
                    hit[0] += nhit > 0; hit[1] += 1
                else:
                    hit[0] += ft in top; hit[1] += 1
                ctx.layer_token_weights = None
                if args.key_select != "off":
                    ltw, total = {}, batch.input_ids.shape[1]
                    fa, fb = (fact_tok[i][0], fact_tok[i][1]) if fact_tok[i] is not None else (0, 0)
                    for l in (sorted(dbg_all) if dbg_all else range(n_layers)):
                        wl = torch.ones(1, total)
                        for s_, e_ in spans:
                            n_ = e_ - s_
                            m_ = max(1, int(round(args.key_frac * n_)))
                            if args.key_select == "random":
                                idx = torch.randperm(n_, generator=key_gen)[:m_]
                            else:
                                st = dbg_all[l][i, s_:e_].float()
                                if args.key_pool > 1:
                                    st = F.max_pool1d(st[None, None], args.key_pool, 1, args.key_pool // 2)[0, 0, :n_]
                                idx = st.topk(m_).indices
                            wl[0, s_:e_] = 0.0
                            wl[0, s_ + idx] = 1.0
                        ltw[l] = wl
                        if fb > fa:
                            key_stat[0] += float(wl[0, fa:fb].mean()); key_stat[1] += 1
                    ctx.layer_token_weights = ltw
                one = type(batch)(batch.input_ids[i : i + 1], None, batch.segments, flags, [r])
                qm1 = None
                if args.bank_score == "layer_qk":
                    ctx.bank_scores = {l: t[i : i + 1].float() for l, t in dbg_all.items()}
                else:
                    ctx.bank_scores = tokrel if args.bank_score == "relevance" else None
                if args.replay == "pseudo" and span_all is not None:
                    nq_i, hh = len(r["q_ids"]), next(iter(span_all.values())).shape[0] // len(batch.rows)
                    ctx.query_q = {l: t[i * hh : (i + 1) * hh, :nq_i] for l, t in span_all.items()}
                    qm1 = torch.ones(1, nq_i, dtype=torch.bool)
                elif args.replay == "pseudo":
                    # rehearsal probes = the question's read vectors (question alone, before any history)
                    qrow = torch.tensor([r["q_ids"]])
                    qm1 = torch.ones(1, qrow.shape[1], dtype=torch.bool)
                    ctx.query_q, ctx.capture_query = {}, True
                    ctx.set_segments([(0, qrow.shape[1])], [False])
                    with torch.no_grad(), autocast(args):
                        model(input_ids=qrow.to(args.device), logits_to_keep=1)
                    ctx.capture_query = False
                results.extend(run(one, n_hist, [fact_tok[i]], [turn_span[i]], [queries[i]], qm1, None))
                ctx.query_q, ctx.bank_scores, ctx.layer_update_flags, ctx.layer_token_weights = {}, None, None, None
            n_show = n_hist
        elif args.query_source == "prefix":
            # the question is put in front of every turn; one row at a time because rows differ in length
            results = []
            for i, r in enumerate(batch.rows):
                qi_ids, nq = r["q_ids"], len(r["q_ids"])
                ids, segs = [], []
                for (ts, te) in enc.turn_spans:
                    segs.append((len(ids), len(ids) + nq + te - ts))
                    ids.extend(qi_ids + enc.history_ids[ts:te])
                nh = len(ids)
                seq = ids + qi_ids + r["a_ids"]
                row = dict(r, answer_span=(nh + nq, nh + nq + len(r["a_ids"])))
                one = type(batch)(torch.tensor([seq]), None, segs + [(nh, len(seq))], [True] * len(segs) + [False], [row])
                t = queries[i]["fact_turn"]
                shift = segs[t][0] + nq - enc.turn_spans[t][0]
                ft = None if fact_tok[i] is None else tuple(x + shift for x in fact_tok[i])
                if ctx is not None:
                    ctx.query_prefix = nq
                oracle = None
                if sel.mode == "oracle":
                    oracle = torch.ones(1, len(seq))
                    if ft is not None:
                        oracle[0, segs[t][0] : segs[t][1]] = args.oracle_low
                        oracle[0, ft[0] : ft[1]] = args.oracle_high
                results.extend(run(one, nh, [ft], [(segs[t][0] + nq, segs[t][1])], [queries[i]], None, oracle))
            n_show = nh
        else:
            qmask = torch.zeros(len(batch.rows), total - n_hist, dtype=torch.bool)
            for i, r in enumerate(batch.rows):
                qmask[i, : len(r["q_ids"])] = True  # question tokens only: the answer must not steer the write
            if args.query_source == "alone" and ctx is not None and (sel.mode in ("query", "hybrid") or args.replay == "pseudo"):
                # the question alone, before any history: its read vectors are computed once and reused
                nq = max(len(r["q_ids"]) for r in batch.rows)
                qb = torch.full((len(batch.rows), nq), tok.pad_token_id, dtype=torch.long)
                qmask = torch.zeros(len(batch.rows), nq, dtype=torch.bool)
                for i, r in enumerate(batch.rows):
                    qb[i, : len(r["q_ids"])] = torch.tensor(r["q_ids"])
                    qmask[i, : len(r["q_ids"])] = True
                ctx.capture_query = True
                ctx.set_segments([(0, nq)], [False])
                with torch.no_grad(), autocast(args):
                    model(input_ids=qb.to(args.device), logits_to_keep=1)
                ctx.capture_query = False
            oracle = None
            if lexical:
                hist_t = torch.tensor(enc.history_ids)
                counts = corpus_counts if corpus_counts is not None else torch.bincount(hist_t, minlength=len(tok)).float()
                oracle = torch.ones(len(batch.rows), total)
                for i, r in enumerate(batch.rows):
                    rar = rarity(r["q_ids"], counts, float(counts.sum()))
                    oracle[i, :n_hist] = lexical_weights(hist_t, r["q_ids"], rar, args.boost, args.pool, args.pool_mode, args.min_rarity)
            elif sel.mode == "oracle":
                oracle = torch.ones(len(batch.rows), total)
                for i in range(len(batch.rows)):
                    if fact_tok[i] is not None:
                        oracle[i, turn_span[i][0] : turn_span[i][1]] = args.oracle_low
                        oracle[i, fact_tok[i][0] : fact_tok[i][1]] = args.oracle_high
            results = run(batch, n_hist, fact_tok, turn_span, queries, qmask, oracle)
            n_show = n_hist

        by_query = defaultdict(dict)
        for res in results:
            by_query[res["query_index"]][res["which"]] = res
        for qi, q in enumerate(dlg["queries"]):
            res = by_query[qi]
            ans = res["answer"]
            rec = {
                "dialogue": dlg["id"], "category": q["category"], "distance_tokens": ans["distance_tokens"],
                "fact_turn": q["fact_turn"], "num_turns": len(dlg["turns"]),
                "bucket": distance_bucket(ans["distance_tokens"], args.window),
                "exact": ans["exact"], "logprob": ans["logprob"],
                "enrichment": ans["enrichment"], "fact_gain": ans["fact_gain"], "venrich": ans["venrich"],
            }
            if "old_answer" in res:
                rec["prefers_old"] = res["old_answer"]["logprob"] > ans["logprob"]
            records.append(rec)
        print(f"[{di + 1}/{len(dialogues)}] history={n_show} tokens", flush=True)

    summary = {}
    groups = defaultdict(list)
    for r in records:
        groups[(r["category"], "all")].append(r)
        groups[(r["category"], r["bucket"])].append(r)
    for (cat, bucket), rs in sorted(groups.items()):
        e = {"n": len(rs), "accuracy": sum(r["exact"] for r in rs) / len(rs),
             "answer_logprob": sum(r["logprob"] for r in rs) / len(rs)}
        en = [r["enrichment"] for r in rs if r["enrichment"] is not None]
        if en:
            e["enrichment"] = sum(en) / len(en)
        ve = [r["venrich"] for r in rs if r["venrich"] is not None]
        if ve:
            e["venrich"] = sum(ve) / len(ve)
        fg = [r["fact_gain"] for r in rs if r["fact_gain"] is not None]
        if fg:
            e["fact_gain"] = sum(fg) / len(fg)
        olds = [r["prefers_old"] for r in rs if "prefers_old" in r]
        if olds:
            e["prefers_old_rate"] = sum(olds) / len(olds)
        summary[f"{cat}/{bucket}"] = e

    if turn_mode:
        print(f"\nturn selection {turn_mode} k={args.turn_k} thresh={args.write_thresh} layer={args.turn_layer}: fact turn written in {hit[0]}/{hit[1]} rows; "
              f"turns written {written[0] / max(written[1], 1):.3f} of all turns")
        if key_stat[1]:
            print(f"key selection {args.key_select} frac={args.key_frac} pool={args.key_pool}: a layer keeps {key_stat[0] / key_stat[1]:.2f} "
                  f"of the fact sentence's tokens (random would keep about {args.key_frac:.2f})")
        if layer_stat[2]:
            print(f"per-layer {args.per_layer}: {layer_stat[0] / layer_stat[2]:.2f} of the layers selected; "
                  f"a selecting layer kept the fact turn in {layer_stat[1] / max(layer_stat[0], 1):.2f} of the cases")
        args.select = turn_mode
    print(f"\nselect={args.select} boost={args.boost} idf={os.path.basename(args.idf)} source={args.query_source} tau={args.tau} cap={args.cap} seg_gain={args.seg_gain} "
          f"keep={args.keep} pool={args.pool}/{args.pool_mode} qlast={args.query_last} topk={args.topk_frac} replay={args.replay or 'from-config'} probes={args.probe_src} target={args.pseudo_target} "
          f"rkeep={args.replay_keep} samples={args.replay_samples} lr_scale={args.replay_lr_scale}")
    print(f"{'category/bucket':32s} {'n':>5s} {'acc':>7s} {'logprob':>9s} {'enrich':>7s} {'venr':>7s} {'gain':>6s} {'old':>6s}")
    for key, e in summary.items():
        f = lambda k: f"{e[k]:.3f}" if k in e else ""
        print(f"{key:32s} {e['n']:5d} {e['accuracy']:7.3f} {e['answer_logprob']:9.3f} "
              f"{f('enrichment'):>7s} {f('venrich'):>7s} {f('fact_gain'):>6s} {f('prefers_old_rate'):>6s}")
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as fo:
            json.dump({"args": vars(args), "summary": summary, "records": records}, fo, indent=2)


if __name__ == "__main__":
    main()
