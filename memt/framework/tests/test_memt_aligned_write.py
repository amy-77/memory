"""MemT-1.0 alignment of the matrix memory: stable block write without a decay bound, MemT norms."""

import copy
import json
import os

import pytest
import torch

from conftest import tiny_qwen3
from lact_qwen import MemoryConfig, attach_memory, freeze_base, set_attention_window
from lact_qwen.matrix_memory import (
    AUTO_MAX_LOG_GROWTH, MAX_DECAY, delta_write, delta_write_auto, delta_write_reference, delta_write_stable)
from lact_qwen.memory_branch import memt_l2_norm

HERE = os.path.dirname(os.path.abspath(__file__))


def inputs(b=3, length=70, dk=16, dv=12, decay=MAX_DECAY, dtype=torch.float64, seed=0):
    g = torch.Generator().manual_seed(seed)
    k = torch.nn.functional.normalize(torch.randn(b, length, dk, generator=g, dtype=dtype), dim=-1)
    u = torch.rand(b, length, dk, generator=g, dtype=dtype).sqrt() * k
    z = torch.randn(b, length, dv, generator=g, dtype=dtype)
    lam = -torch.rand(b, length, dk, generator=g, dtype=dtype) * decay
    start = torch.randn(b, dk, dv, generator=g, dtype=dtype)
    return start, u, z, lam


@pytest.mark.parametrize("decay", [MAX_DECAY, 0.5, 3.0, 20.0])
@pytest.mark.parametrize("block,group", [(1, 1), (5, 10), (64, 256), (64, 64), (7, 1000)])
def test_stable_write_equals_token_by_token_for_any_decay(decay, block, group):
    start, u, z, lam = inputs(decay=decay)
    ref = delta_write_reference(start, u, z, lam)
    out = delta_write_stable(start, u, z, lam, block=block, group=group)
    torch.testing.assert_close(out, ref, rtol=1e-10, atol=1e-10)


def test_stable_write_fp32_close_to_fp64_reference():
    start, u, z, lam = inputs(length=300, decay=2.0)
    ref = delta_write_reference(start, u, z, lam)
    out = delta_write_stable(*(t.float() for t in (start, u, z, lam)))
    torch.testing.assert_close(out.double(), ref, rtol=1e-4, atol=1e-5)


def test_scaled_write_needs_the_decay_cap():
    # why the old path clamps the decay: 1 / P_i overflows inside a 256-token block
    start, u, z, lam = (t.float() for t in inputs(length=256, decay=2.0))
    ref = delta_write_reference(start.double(), u.double(), z.double(), lam.double())
    old = delta_write(start, u, z, lam, 256)
    assert not torch.isfinite(old).all() or (old.double() - ref).abs().max() > 1e-2
    torch.testing.assert_close(delta_write_stable(start, u, z, lam).double(), ref, rtol=1e-4, atol=1e-5)


def mixed_decay_inputs(length=600, dtype=torch.float64):
    # blocks of 256 tokens: small decays (fast path), then one channel with large decays (stable path), then small
    start, u, z, lam = inputs(length=length, decay=1e-3, dtype=dtype)
    lam[:, 256:300, 3] = -2.0
    return start, u, z, lam


@pytest.mark.parametrize("decay", [1e-3, MAX_DECAY, 0.5, 20.0])
def test_auto_write_equals_token_by_token(decay):
    start, u, z, lam = inputs(length=600, decay=decay)
    torch.testing.assert_close(delta_write_auto(start, u, z, lam), delta_write_reference(start, u, z, lam),
                               rtol=1e-9, atol=1e-9)
    start, u, z, lam = mixed_decay_inputs()
    torch.testing.assert_close(delta_write_auto(start, u, z, lam), delta_write_reference(start, u, z, lam),
                               rtol=1e-9, atol=1e-9)


def test_auto_fast_path_fp32_accuracy_at_its_threshold():
    # every channel of every block sits just below the threshold: the largest 1 / P_i the fast path accepts
    start, u, z, lam = inputs(length=512, decay=0.0)
    lam[:] = -0.999 * AUTO_MAX_LOG_GROWTH / 256
    ref = delta_write_reference(start, u, z, lam)
    out = delta_write_auto(*(t.float() for t in (start, u, z, lam)))
    torch.testing.assert_close(out.double(), ref, rtol=1e-4, atol=1e-4)


def test_auto_write_gradients_match_stable():
    start, u, z, lam = (t.float() for t in mixed_decay_inputs())
    probe = torch.randn(start.shape)
    grads = []
    for fn in (delta_write_auto, delta_write_stable):
        args = [t.clone().requires_grad_() for t in (start, u, z, lam)]
        (fn(*args) * probe).sum().backward()
        grads.append([a.grad for a in args])
    for a, b in zip(*grads):
        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-5)


def test_stable_write_gradcheck():
    start, u, z, lam = inputs(b=2, length=11, dk=4, dv=3, decay=1.0)
    args = tuple(t.requires_grad_() for t in (start, u, z, lam))
    for block, group in ((3, 6), (4, 100), (1, 1)):
        assert torch.autograd.gradcheck(
            lambda s, a, b, c: delta_write_stable(s, a, b, c, block=block, group=group), args, eps=1e-6, atol=1e-7
        )


def test_checkpoint_groups_do_not_change_gradients():
    start, u, z, lam = (t.float() for t in inputs(length=100, decay=0.5))
    probe = torch.randn(start.shape)
    grads = []
    for group in (100, 8):
        args = [t.clone().requires_grad_() for t in (start, u, z, lam)]
        (delta_write_stable(*args, block=4, group=group) * probe).sum().backward()
        grads.append([a.grad for a in args])
    for a, b in zip(*grads):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)


def test_memt_l2_norm_is_nano_normalize():
    x = torch.randn(4, 7, 32, dtype=torch.bfloat16)
    expected = x.float() * torch.rsqrt(x.float().square().sum(-1, keepdim=True) + 1e-6)  # MemTMemory._normalize
    torch.testing.assert_close(memt_l2_norm(x), expected, rtol=0, atol=0)


def test_defaults_are_the_trained_4b_settings():
    cfg = MemoryConfig()
    assert (cfg.decay_cap, cfg.write_impl, cfg.out_norm_eps, cfg.qk_norm, cfg.memt_rounding) == (
        MAX_DECAY, "scaled", 1e-5, "lact", False)
    with open(os.path.join(HERE, "..", "..", "train_run", "step_2160_memory_config.json")) as f:
        old = MemoryConfig.from_dict(json.load(f)["memory"])  # written before these fields existed
    assert old.write_impl == "scaled" and old.decay_cap == MAX_DECAY
    with pytest.raises(ValueError):
        MemoryConfig(decay_cap=None)  # the scaled write cannot run without the cap
    with open(os.path.join(HERE, "..", "..", "train_run", "memt_aligned.json")) as f:
        aligned = MemoryConfig.from_dict(json.load(f)["memory"])
    assert (aligned.decay_cap, aligned.write_impl, aligned.out_norm_eps, aligned.qk_norm, aligned.memt_rounding) == (
        None, "auto", 1e-6, "memt", True)


def build(base, **kw):
    cfg = dict(num_heads=2, head_dim=32, chunk_size=8, core="matrix", gate_init=0.5, write_block=5, forget_rank=4,
               qkv_init="attention", qkv_act="none", inject="pre_o_proj")
    cfg.update(kw)
    model = copy.deepcopy(base)
    attach_memory(model, MemoryConfig(**cfg))
    return model


def test_stable_and_scaled_writes_agree_in_the_model_when_the_cap_is_inactive():
    base = tiny_qwen3()
    set_attention_window(base, 8, segment=8)
    old = build(base)
    new = build(base, write_impl="stable")
    new.load_state_dict(old.state_dict())
    ids = torch.randint(0, 97, (2, 40))
    with torch.no_grad():
        torch.testing.assert_close(new(ids).logits, old(ids).logits, rtol=1e-4, atol=1e-4)


def test_aligned_model_trains_with_large_decays():
    base = tiny_qwen3()
    set_attention_window(base, 8, segment=8)
    model = build(base, decay_cap=None, write_impl="auto", out_norm_eps=1e-6, qk_norm="memt", memt_rounding=True)
    branch = model.model.layers[0].memory
    with torch.no_grad():  # decays of 1-5 per token, far beyond the old cap of 0.03
        branch.forget_up.bias.fill_(3.0)
    freeze_base(model)
    ids = torch.randint(0, 97, (2, 40))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        logits = model(ids).logits
    assert torch.isfinite(logits).all()
    logits.float().sum().backward()
    for name, p in branch.named_parameters():
        if name in ("forget_down.weight",):  # forget_up starts at zero, so its input factor has no gradient yet
            continue
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
    assert branch.forget_up.bias.grad.abs().sum() > 0
