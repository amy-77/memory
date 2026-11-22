import torch
import torch.nn.functional as F
import pytest

from lact_qwen.memory_branch import LaCTMemoryBranch, MemoryConfig, MemoryContext
from lact_qwen.numerics import clip_grad_norm_fp64


def test_saturated_erase_gate_preserves_finite_backward():
    # A real branch write reproduces NaNs in the legacy sqrt(sigmoid) backward.
    torch.manual_seed(4)
    cfg = MemoryConfig(core="matrix", num_heads=2, head_dim=4, forget_rank=2,
                       write_block=4, stable_matrix_gates=True)
    branch = LaCTMemoryBranch(8, cfg, 0, MemoryContext())
    with torch.no_grad():
        branch.erase_proj.weight.fill_(-15.0)
    x = torch.ones(1, 8, 8, requires_grad=True)
    state = torch.randn(2, 4, 4)
    out = branch._matrix_write(state, x)
    out.square().sum().backward()
    assert bool(torch.isfinite(out).all())
    assert bool(torch.isfinite(x.grad).all())
    assert branch.erase_proj.weight.grad is not None
    assert bool(torch.isfinite(branch.erase_proj.weight.grad).all())


def test_stable_gate_matches_double_precision_value_and_derivative():
    x = torch.tensor([-120., -100., -90., -20., 0., 20., 40.], requires_grad=True)
    actual = torch.exp(0.5 * F.logsigmoid(x))
    grad = torch.autograd.grad(actual.sum(), x)[0]
    xd = x.detach().double().requires_grad_()
    expected = torch.exp(0.5 * F.logsigmoid(xd))
    ref_grad = torch.autograd.grad(expected.sum(), xd)[0]
    torch.testing.assert_close(actual.double(), expected, rtol=2e-6, atol=1e-35)
    torch.testing.assert_close(grad.double(), ref_grad, rtol=2e-6, atol=1e-35)


def test_fp64_clip_handles_large_but_finite_gradients():
    p = torch.nn.Parameter(torch.zeros(16))
    p.grad = torch.full_like(p, 1e20)
    assert not bool(torch.isfinite(torch.linalg.vector_norm(p.grad)))
    norm = clip_grad_norm_fp64([p], 1.0)
    assert bool(torch.isfinite(norm))
    torch.testing.assert_close(p.grad, torch.full_like(p, 0.25))


def test_fp64_clip_still_rejects_actual_nonfinite_gradients():
    p = torch.nn.Parameter(torch.zeros(2))
    p.grad = torch.tensor([1.0, float("nan")])
    with pytest.raises(RuntimeError, match="Non-finite"):
        clip_grad_norm_fp64([p], 1.0)
