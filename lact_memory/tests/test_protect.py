import torch

from lact_qwen.fast_weight import FastWeightState, UpdateConfig, fw_apply, fw_update
from lact_qwen.replay import ReplayBank, ReplayConfig


def _state(bh=3, d=32, e=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    w = [torch.randn(bh, e, d, generator=g) / d**0.5, torch.randn(bh, d, e, generator=g) / e**0.5, torch.randn(bh, e, d, generator=g) / d**0.5]
    return FastWeightState(*w), [x.norm(dim=-1, keepdim=True) for x in w]


def test_protected_write_keeps_banked_readout_and_still_writes():
    torch.manual_seed(0)
    bh, d, n_bank, n_new = 3, 32, 6, 20
    prev, ref = _state(bh, d)
    bank = ReplayBank(ReplayConfig(mode="pseudo", probe_src="bank", protect=True, num_samples=64, protect_ridge=1e-8))
    bank.keys = torch.nn.functional.normalize(torch.randn(bh, n_bank, d), dim=-1)
    bank.scores = torch.rand(bh, n_bank)
    k = torch.nn.functional.normalize(torch.randn(bh, n_new, d), dim=-1)
    v = torch.randn(bh, n_new, d)
    lr = torch.full((bh, n_new, 1), 0.05)
    cand = fw_update(prev, k, v, lr, lr, lr, None, ref, UpdateConfig(use_muon=True, use_momentum=False))
    new = bank.project(prev, cand)
    before, plain, prot = fw_apply(*prev.weights(), bank.keys), fw_apply(*cand.weights(), bank.keys), fw_apply(*new.weights(), bank.keys)
    assert (plain - before).abs().max() > 1e-3            # the ordinary write does disturb the banked keys
    assert (prot - before).abs().max() < 1e-4             # the protected write does not
    # ... and the new content is still written: the read-out at the new keys moves towards v almost as much
    gain = lambda st: (fw_apply(*st.weights(), k) * v).sum() - (fw_apply(*prev.weights(), k) * v).sum()
    assert gain(new) > 0.5 * gain(cand) > 0


def test_protect_off_or_empty_bank_is_identity():
    prev, ref = _state()
    cand, _ = _state(seed=1)
    assert ReplayBank(ReplayConfig(mode="pseudo", probe_src="bank", protect=False)).project(prev, cand) is cand
    assert ReplayBank(ReplayConfig(mode="pseudo", probe_src="bank", protect=True)).project(prev, cand) is cand
