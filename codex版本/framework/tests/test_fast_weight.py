import torch

from lact_qwen.fast_weight import (
    FastWeightState,
    UpdateConfig,
    fw_apply,
    fw_descent_direction,
    l2_norm,
    segmented_lact,
)
from lact_qwen.replay import ReplayBank, ReplayConfig


def rand_weights(bh=3, d=8, dh=12, requires_grad=False):
    torch.manual_seed(0)
    ws = [torch.randn(bh, dh, d) / d**0.5, torch.randn(bh, d, dh) / dh**0.5, torch.randn(bh, dh, d) / d**0.5]
    return [w.requires_grad_(requires_grad) for w in ws]


def test_descent_direction_matches_autograd():
    w0, w1, w2 = rand_weights(requires_grad=True)
    k = torch.randn(3, 10, 8)
    target = torch.randn(3, 10, 8)
    lr = torch.full((3, 10, 1), 0.3)

    loss = -(lr * (fw_apply(w0, w1, w2, k) * target)).sum()
    g0, g1, g2 = torch.autograd.grad(loss, (w0, w1, w2))
    d0, d1, d2 = fw_descent_direction(w0.detach(), w1.detach(), w2.detach(), k, target, lr, lr, lr)

    for g, dd in ((g0, d0), (g1, d1), (g2, d2)):
        torch.testing.assert_close(dd, -g, rtol=1e-4, atol=1e-5)


def run_segments(q, k, v, segments, replay=None, muon=False):
    bh, length, _ = q.shape
    w0, w1, w2 = rand_weights(bh=bh, d=q.shape[-1])
    lr = torch.full((bh, length, 3), 0.1)
    norms = tuple(w.norm(dim=2, keepdim=True) for w in (w0, w1, w2))
    state = FastWeightState(w0, w1, w2, replay=replay)
    return segmented_lact(
        state, q, k, v, lr, None, segments, [True] * len(segments), norms, UpdateConfig(use_muon=muon, use_momentum=False)
    )


def test_segments_are_causal():
    torch.manual_seed(1)
    q, k, v = (l2_norm(torch.randn(3, 30, 8)) for _ in range(3))
    segments = [(0, 10), (10, 20), (20, 30)]
    out, _ = run_segments(q, k, v, segments)

    k2, v2 = k.clone(), v.clone()
    k2[:, 12] += 1.0
    v2[:, 15] -= 1.0
    out2, _ = run_segments(q, k2, v2, segments)

    torch.testing.assert_close(out[:, :20], out2[:, :20])
    assert (out[:, 20:] - out2[:, 20:]).abs().max() > 1e-4


def test_oracle_key_replay_reduces_drift_on_old_keys():
    torch.manual_seed(2)
    bh, d = 2, 8
    old_k = l2_norm(torch.randn(bh, 40, d))
    new_k = l2_norm(torch.randn(bh, 40, d))
    v = torch.randn(bh, 40, d)
    w0, w1, w2 = rand_weights(bh=bh)
    norms = tuple(w.norm(dim=2, keepdim=True) for w in (w0, w1, w2))
    prev = FastWeightState(w0, w1, w2)
    lr = torch.full((bh, 40, 1), 0.5)

    from lact_qwen.fast_weight import fw_update

    cand = fw_update(prev, new_k, v, lr, lr, lr, None, norms, UpdateConfig(use_muon=False, use_momentum=False))
    bank = ReplayBank(ReplayConfig(mode="oracle_key", num_samples=128, strength=1.0, clip_ratio=0.0, selection="uniform"))
    bank.observe(old_k, v, lr.expand(-1, -1, 3))
    fixed = bank.correct(prev, cand, new_k, norms)

    ref = fw_apply(*prev.weights(), old_k)
    drift_cand = (fw_apply(*cand.weights(), old_k) - ref).norm()
    drift_fixed = (fw_apply(*fixed.weights(), old_k) - ref).norm()
    assert drift_fixed < drift_cand


def test_all_replay_modes_run_and_stay_causal():
    torch.manual_seed(3)
    q, k, v = (l2_norm(torch.randn(2, 40, 8)) for _ in range(3))
    segments = [(0, 10), (10, 20), (20, 30), (30, 40)]
    for mode in ("oracle_kv", "oracle_key", "gmm"):
        cfg = ReplayConfig(mode=mode, num_samples=16, gmm_components=4)
        torch.manual_seed(0)
        out, state = run_segments(q, k, v, segments, replay=ReplayBank(cfg), muon=True)
        assert torch.isfinite(out).all()
        k2 = k.clone()
        k2[:, 35] += 1.0
        torch.manual_seed(0)
        out2, _ = run_segments(q, k2, v, segments, replay=ReplayBank(cfg), muon=True)
        torch.testing.assert_close(out[:, :30], out2[:, :30])


def test_gmm_tracks_cluster_means():
    torch.manual_seed(4)
    centers = l2_norm(torch.randn(1, 3, 16))
    pts = l2_norm(centers.repeat_interleave(200, dim=1) + 0.05 * torch.randn(1, 600, 16))
    bank = ReplayBank(ReplayConfig(mode="gmm", gmm_components=3, num_samples=64, selection="uniform"))
    for chunk in pts.split(100, dim=1):
        bank.observe(chunk, chunk, torch.ones(1, chunk.shape[1], 3))
    samples = bank.sample_keys(pts[:, :10])
    nearest = torch.einsum("bnd,bkd->bnk", samples, centers).max(dim=-1).values
    assert nearest.mean() > 0.8
