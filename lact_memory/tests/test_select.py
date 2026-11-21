"""Selective writing and scored replay: shapes, invariants, and that every mode runs."""
import torch

from lact_qwen.replay import ReplayBank, ReplayConfig
from lact_qwen.select import SelectConfig, agnostic_scores, query_scores, scores_to_weights, select_weights


def _qk(b=2, h=3, n=40, d=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = torch.nn.functional.normalize(torch.randn(b * h, n, d, generator=g), dim=-1)
    k = torch.nn.functional.normalize(torch.randn(b * h, n, d, generator=g), dim=-1)
    return q, k


def test_weights_mean_and_cap():
    s = torch.randn(4, 50)
    w = scores_to_weights(s, SelectConfig(mode="query", tau=0.5, cap=10.0))
    assert w.shape == s.shape and (w >= 0).all() and w.max() <= 10.0 + 1e-6
    assert (w.mean(dim=-1) <= 1.0 + 1e-5).all()
    flat = scores_to_weights(torch.zeros(2, 20), SelectConfig(mode="query"))
    assert torch.allclose(flat, torch.ones(2, 20), atol=1e-4)  # no signal -> ordinary write


def test_query_scores_find_planted_key():
    b, h = 2, 3
    q, k = _qk(b, h)
    # history = tokens 0..29, question = tokens 30..39; plant the question direction at token 7
    k[:, 7] = q[:, 33]
    qmask = torch.ones(b, 10, dtype=torch.bool)
    s = query_scores(k[:, :30], q[:, 30:], qmask, b, h, topk_frac=0.1)
    assert s.shape == (b, 30) and (s.argmax(dim=-1) == 7).all()
    # masking the matching question token removes the hit
    qmask[:, 3] = False
    s2 = query_scores(k[:, :30], q[:, 30:], qmask, b, h, topk_frac=0.1)
    assert (s2[:, 7] < s[:, 7]).all()


def test_agnostic_scores_shape_and_last_token():
    b, h = 2, 3
    q, k = _qk(b, h, n=12)
    for red in ("max", "sum"):
        s = agnostic_scores(q, k, b, h, 0.05, red)
        assert s.shape == (b, 12) and (s[:, -1] == 0).all() and (s >= 0).all()


def test_select_weights_modes_and_gains():
    b, h = 2, 3
    q, k = _qk(b, h)
    segments = [(0, 10), (10, 20), (20, 30), (30, 40)]
    flags = [True, True, True, False]
    qmask = torch.ones(b, 10, dtype=torch.bool)
    for mode in ("query", "agnostic", "hybrid"):
        w, g = select_weights(SelectConfig(mode=mode, seg_gain=True), q, k, segments, flags, b, h, (30, 40), qmask)
        assert w.shape == (b, 40) and g.shape == (b, 4)
        assert torch.allclose(w[:, 30:], torch.ones(b, 10))  # read-only segment untouched
        assert (g[:, 3] == 1).all() and (g[:, :3].mean(dim=1) <= 1.0 + 1e-5).all()
    ow = torch.rand(b, 40)
    w, g = select_weights(SelectConfig(mode="oracle"), q, k, segments, flags, b, h, oracle_weight=ow)
    assert torch.equal(w, ow) and g is None


def test_scored_bank_keeps_best_and_replays_them():
    cfg = ReplayConfig(mode="scored_kv", keep_per_segment=4, bank_size=6, num_samples=5)
    bank = ReplayBank(cfg)
    bh, d = 3, 8
    for step in range(3):
        k, v, lr = torch.randn(bh, 20, d), torch.randn(bh, 20, d), torch.rand(bh, 20, 3)
        score = torch.rand(bh, 20) + step  # later segments score higher
        bank.observe(k, v, lr, score)
    assert bank.keys.shape == (bh, 6, d) and bank.scores.shape == (bh, 6)
    assert (bank.scores.min(dim=1).values >= 1.0).all()  # the first segment (scores < 1) was pushed out
    k, v, lr = torch.randn(bh, 7, d), torch.randn(bh, 7, d), torch.rand(bh, 7, 3)
    ka, va, la = bank.augment_write(k, v, lr)
    assert ka.shape == (bh, 12, d) and va.shape == (bh, 12, d) and la.shape == (bh, 12, 3)
    assert torch.equal(ka[:, :7], k)


def test_model_forward_all_modes():
    import argparse

    from lact_qwen.loading import add_model_args, load_model
    from lact_qwen.modeling import set_replay

    p = argparse.ArgumentParser()
    add_model_args(p)
    args = p.parse_args(["--tiny-random", "--memory", "new", "--window", "16", "--device", "cpu", "--dtype", "float32",
                         "--mem-heads", "2", "--mem-head-dim", "32"])
    torch.manual_seed(0)
    model, _ = load_model(args)
    ctx = model.lact_ctx
    ids = torch.randint(0, 1000, (2, 48))
    segs, flags = [(0, 12), (12, 24), (24, 36), (36, 48)], [True, True, True, False]
    ctx.set_segments(segs, flags)
    ctx.query_span, ctx.query_mask = (36, 48), torch.ones(2, 12, dtype=torch.bool)
    with torch.no_grad():
        base = model(input_ids=ids).logits
        outs = {}
        for mode in ("query", "agnostic", "hybrid"):
            ctx.select = SelectConfig(mode=mode, seg_gain=True)
            ctx.last_weights, ctx.last_gains = {}, {}
            outs[mode] = model(input_ids=ids).logits
            assert torch.isfinite(outs[mode]).all() and ctx.last_weights and ctx.last_gains
        ctx.select = SelectConfig(mode="oracle")
        ctx.oracle_weight = torch.ones(2, 48)
        same = model(input_ids=ids).logits
        assert torch.allclose(same, base, atol=1e-4)  # all-ones weights = ordinary write
        ctx.select = SelectConfig(mode="query", score_layer=0)
        ctx.last_weights, ctx.last_gains = {}, {}
        model(input_ids=ids)
        ws = list(ctx.last_weights.values())
        assert all(torch.equal(ws[0], w) for w in ws[1:])  # upper layers reuse layer 0's weights
        set_replay(model, mode="scored_kv", keep_per_segment=4, num_samples=8)
        ctx.select = SelectConfig(mode="query")
        rep = model(input_ids=ids).logits
        assert torch.isfinite(rep).all()


def test_question_alone_and_prefix_sources():
    b, h = 2, 3
    q, k = _qk(b, h, n=40)
    segments, flags = [(0, 10), (10, 20), (20, 30), (30, 40)], [True, True, True, False]
    # question alone: vectors come from outside the sequence
    qq = q[:, 30:].clone()
    k2 = k.clone()
    k2[:, 14] = qq[:, 2]
    w, _ = select_weights(SelectConfig(mode="query", tau=0.3), q, k2, segments, flags, b, h,
                          query_mask=torch.ones(b, 10, dtype=torch.bool), query_q=qq)
    assert (w[:, 10:20].argmax(dim=-1) == 4).all()
    # question as a prefix of every segment: 3 question tokens, they are not written
    k3 = k.clone()
    k3[:, 7] = q[:, 1]  # token 7 of segment 0 matches question token 1 of that segment
    w, g = select_weights(SelectConfig(mode="query", tau=0.3, seg_gain=True), q, k3, segments, flags, b, h, query_prefix=3)
    assert (w[:, 0:3] == 0).all() and (w[:, 10:13] == 0).all() and (w[:, 30:] == 1).all()
    assert (w[:, 3:10].argmax(dim=-1) == 4).all() and g.shape == (b, 4)


def test_capture_query_then_score():
    import argparse

    from lact_qwen.loading import add_model_args, load_model

    p = argparse.ArgumentParser()
    add_model_args(p)
    args = p.parse_args(["--tiny-random", "--memory", "new", "--window", "16", "--device", "cpu", "--dtype", "float32",
                         "--mem-heads", "2", "--mem-head-dim", "32"])
    torch.manual_seed(0)
    model, _ = load_model(args)
    ctx = model.lact_ctx
    qids = torch.randint(0, 1000, (2, 6))
    ctx.capture_query = True
    ctx.set_segments([(0, 6)], [False])
    with torch.no_grad():
        model(input_ids=qids)
    ctx.capture_query = False
    assert ctx.query_q and all(v.shape[1] == 6 for v in ctx.query_q.values())
    ids = torch.randint(0, 1000, (2, 48))
    ctx.set_segments([(0, 12), (12, 24), (24, 36), (36, 48)], [True, True, True, False])
    ctx.select = SelectConfig(mode="query")
    ctx.query_span, ctx.query_mask = None, torch.ones(2, 6, dtype=torch.bool)
    with torch.no_grad():
        out = model(input_ids=ids).logits
    assert torch.isfinite(out).all() and len(ctx.last_weights) == len(ctx.query_q)
    ctx.query_q, ctx.query_prefix = {}, 4
    with torch.no_grad():
        out = model(input_ids=ids).logits
    w0 = next(iter(ctx.last_weights.values()))
    assert torch.isfinite(out).all() and (w0[:, 0:4] == 0).all() and (w0[:, 12:16] == 0).all()


def test_hard_keep_snapkv_style():
    s = torch.zeros(1, 20)
    s[0, 10] = 5.0
    w = scores_to_weights(s, SelectConfig(mode="query", keep=0.25, pool=5, floor=0.0))
    assert w.sum() == 5 and (w[0, 8:13] == 1).all()  # the hit and its neighbours, nothing else


def test_pseudo_rehearsal_pairs():
    from lact_qwen.fast_weight import FastWeightState

    bh, d, dh = 2, 8, 8
    g = torch.Generator().manual_seed(0)
    w0, w1, w2 = (torch.randn(bh, dh, d, generator=g), torch.randn(bh, d, dh, generator=g), torch.randn(bh, dh, d, generator=g))
    init = FastWeightState(w0, w1, w2)
    moved = FastWeightState(w0 + 0.1, w1 - 0.1, w2 + 0.05)
    k, v, lr = torch.randn(bh, 6, d), torch.randn(bh, 6, d), torch.rand(bh, 6, 3)
    probes = torch.nn.functional.normalize(torch.randn(bh, 4, d), dim=-1)
    valid = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.float32)

    bank = ReplayBank(ReplayConfig(mode="pseudo", probe_src="query", pseudo_target="delta", kv_lr_scale=5.0))
    bank.query_q, bank.query_valid = probes, valid
    bank.init_weights = init.weights()
    same = bank.augment_write(k, v, lr, init)  # nothing written yet -> nothing to rehearse
    assert same[0].shape == k.shape
    ka, va, la = bank.augment_write(k, v, lr, moved)
    assert ka.shape == (bh, 10, d) and torch.equal(ka[:, 6:], probes)
    from lact_qwen.fast_weight import fw_apply
    want = fw_apply(*moved.weights(), probes) - fw_apply(*init.weights(), probes)
    assert torch.allclose(va[:, 6:], want, atol=1e-5)
    assert (la[0, 9] == 0).all() and (la[1, 8:] == 0).all()  # padded question tokens are not written
    assert torch.allclose(la[0, 6], lr[0].mean(dim=0) * 5.0, atol=1e-6)

    full = ReplayBank(ReplayConfig(mode="pseudo", probe_src="both", pseudo_target="full", keep_per_segment=2, num_samples=3))
    full.query_q, full.query_valid, full.init_weights = probes, valid, init.weights()
    full.observe(k, v, lr, torch.rand(bh, 6))
    full.observe(k, v, lr, torch.rand(bh, 6))
    assert full.keys.shape == (bh, 4, d) and full.values is None  # keys only, no stored values
    ka, va, la = full.augment_write(k, v, lr, moved)
    assert ka.shape == (bh, 6 + 4 + 3, d)


def test_model_forward_pseudo():
    import argparse

    from lact_qwen.loading import add_model_args, load_model
    from lact_qwen.modeling import set_replay

    p = argparse.ArgumentParser()
    add_model_args(p)
    args = p.parse_args(["--tiny-random", "--memory", "new", "--window", "16", "--device", "cpu", "--dtype", "float32",
                         "--mem-heads", "2", "--mem-head-dim", "32", "--mem-delta-readout", "--mem-gate-init", "0.1"])
    torch.manual_seed(0)
    model, _ = load_model(args)
    ctx = model.lact_ctx
    ids = torch.randint(0, 1000, (2, 48))
    ctx.set_segments([(0, 12), (12, 24), (24, 36), (36, 48)], [True, True, True, False])
    with torch.no_grad():
        base = model(input_ids=ids).logits
        ctx.capture_query = True
        ctx.set_segments([(0, 6)], [False])
        model(input_ids=ids[:, 36:42])
        ctx.capture_query = False
        ctx.set_segments([(0, 12), (12, 24), (24, 36), (36, 48)], [True, True, True, False])
        ctx.query_span, ctx.query_mask = None, torch.ones(2, 6, dtype=torch.bool)
        for src in ("query", "bank", "both"):
            set_replay(model, mode="pseudo", probe_src=src, kv_lr_scale=5.0, keep_per_segment=4, num_samples=8)
            ctx.select = SelectConfig(mode="query", keep=0.5, pool=3)
            out = model(input_ids=ids).logits
            assert torch.isfinite(out).all() and not torch.allclose(out, base)


def test_lexical_weights():
    from lact_qwen.select import lexical_weights, rarity

    hist = torch.tensor([5, 5, 5, 7, 5, 9, 11, 12, 5, 5, 5, 5, 5, 5, 5, 5])
    counts = torch.bincount(hist, minlength=20).float()
    question = [5, 9, 3]  # 5 is common in the history, 9 occurs once, 3 never
    rar = rarity(question, counts, float(counts.sum()))
    assert rar[0] < 0.5 and rar[1] > 0.5 and rar[2] == 1.0
    w = lexical_weights(hist, question, rar, boost=10.0, pool=3, pool_mode="forward", min_rarity=0.5)
    assert (w[:5] == 1).all() and (w[8:] == 1).all()  # the common token gives no boost
    assert w[5] > 5 and torch.equal(w[5:8], w[5].expand(3))  # the hit and the two tokens after it
