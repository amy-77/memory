"""SwiGLU fast-weight memory with segment-level apply/update.

Shapes follow the LaCT minimal implementation:
    w0, w2: [BH, dh, d]   w1: [BH, d, dh]
    f(x) = w1 @ (silu(w0 @ x) * (w2 @ x))
q, k, v are [BH, L, d] and the per-token learning rates are [BH, L, 1].

Each segment is first read with the weights accumulated from earlier segments
and then (optionally) written into them, so a token never reads memory written
by its own segment or any later one.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Sequence

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from .replay import ReplayBank


def compiled_on_cuda(fn):
    """Each segment launches hundreds of tiny kernels; on GPU they are fused with torch.compile."""
    compiled = torch.compile(fn, dynamic=True)

    @functools.wraps(fn)
    def wrapper(*args):
        return compiled(*args) if args[0].is_cuda else fn(*args)

    return wrapper


def l2_norm(x: torch.Tensor) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + 1e-5)


def silu_backprop(dy: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    sigma = torch.sigmoid(x)
    return dy * sigma * (1 + x * (1 - sigma))


@compiled_on_cuda
def zeropower_via_newtonschulz5(G: torch.Tensor) -> torch.Tensor:
    """Batched Muon orthogonalisation, G: [b, m, n]."""
    X = G.to(torch.bfloat16) if G.is_cuda else G.float()
    transposed = G.size(1) > G.size(2)
    if transposed:
        X = X.transpose(1, 2)
    X = X / (X.norm(dim=(1, 2), keepdim=True) + 1e-7)
    for a, b, c in [
        (4.0848, -6.8946, 2.9270),
        (3.9505, -6.3029, 2.6377),
        (3.7418, -5.5913, 2.3037),
        (2.8769, -3.1427, 1.2046),
        (2.8366, -3.0525, 1.2012),
    ]:
        A = X @ X.transpose(1, 2)
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed:
        X = X.transpose(1, 2)
    return X.to(G.dtype)


@dataclass
class FastWeightState:
    w0: torch.Tensor
    w1: torch.Tensor
    w2: torch.Tensor
    m0: Optional[torch.Tensor] = None
    m1: Optional[torch.Tensor] = None
    m2: Optional[torch.Tensor] = None
    replay: Optional["ReplayBank"] = field(default=None, repr=False)
    # Un-normalised sums of all writes, kept only when UpdateConfig.norm == "pre".
    raw: Optional[tuple] = field(default=None, repr=False)

    def detach(self) -> "FastWeightState":
        def d(t):
            return None if t is None else t.detach()

        raw = None if self.raw is None else tuple(d(t) for t in self.raw)
        return FastWeightState(
            d(self.w0), d(self.w1), d(self.w2), d(self.m0), d(self.m1), d(self.m2), self.replay, raw
        )

    def weights(self):
        return self.w0, self.w1, self.w2


@compiled_on_cuda
def fw_apply(w0: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """x: [BH, n, d] -> [BH, n, d]."""
    x = x.to(w0.dtype)
    gate = F.silu(torch.bmm(x, w0.transpose(1, 2)))
    hidden = torch.bmm(x, w2.transpose(1, 2))
    return torch.bmm(gate * hidden, w1.transpose(1, 2))


@compiled_on_cuda
def fw_descent_direction(
    w0: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    k: torch.Tensor,
    target: torch.Tensor,
    lr0: torch.Tensor,
    lr1: torch.Tensor,
    lr2: torch.Tensor,
):
    """Negative gradient of  -sum_i lr_i * <f(k_i), target_i>  w.r.t. (w0, w1, w2).

    With target = v this is the LaCT write rule. With target = (v_ref - f(k)) it is
    the descent direction of 0.5 * ||f(k) - v_ref||^2, which the replay step uses.
    """
    dt = w0.dtype
    k = k.to(dt)
    target = target.to(dt)
    gate_pre = torch.bmm(k, w0.transpose(1, 2))
    hidden_pre = torch.bmm(k, w2.transpose(1, 2))
    gate = F.silu(gate_pre)
    hidden = gate * hidden_pre

    dhidden = torch.bmm(target, w1)
    dhidden_pre = dhidden * gate
    dgate_pre = silu_backprop(dhidden * hidden_pre, gate_pre)

    dw1 = torch.bmm((target * lr1.to(dt)).transpose(1, 2), hidden)
    dw0 = torch.bmm(dgate_pre.transpose(1, 2), k * lr0.to(dt))
    dw2 = torch.bmm(dhidden_pre.transpose(1, 2), k * lr2.to(dt))
    return dw0, dw1, dw2


@compiled_on_cuda
def renorm_rows(w: torch.Tensor, ref_norm: torch.Tensor) -> torch.Tensor:
    return w / (w.norm(dim=2, keepdim=True) + 1e-5) * ref_norm


@dataclass
class UpdateConfig:
    use_muon: bool = True
    use_momentum: bool = True
    # "post": add the write to the row-normalised weights, then normalise again (LaCT).
    #         Every write then rescales what earlier writes left behind.
    # "pre":  add the write to an un-normalised running sum and read through its
    #         row-normalised copy, so earlier writes keep their relative size.
    norm: str = "post"


def fw_update(
    state: FastWeightState,
    k: torch.Tensor,
    target: torch.Tensor,
    lr0: torch.Tensor,
    lr1: torch.Tensor,
    lr2: torch.Tensor,
    momentum: Optional[torch.Tensor],
    ref_norms: Sequence[torch.Tensor],
    cfg: UpdateConfig,
    gain: Optional[torch.Tensor] = None,
) -> FastWeightState:
    """One large-chunk write. momentum: [BH, 1, 1] coefficient or None.
    gain: [BH] step-size multiplier applied after Muon (selective writing across segments)."""
    w0, w1, w2 = state.weights()
    dw0, dw1, dw2 = fw_descent_direction(w0, w1, w2, k, target, lr0, lr1, lr2)

    m0 = m1 = m2 = None
    if cfg.use_momentum and momentum is not None:
        coef = momentum.to(w0.dtype)
        if state.m0 is not None:
            dw0 = dw0 + state.m0 * coef
            dw1 = dw1 + state.m1 * coef
            dw2 = dw2 + state.m2 * coef
        m0, m1, m2 = dw0, dw1, dw2

    if cfg.use_muon:
        dw0 = zeropower_via_newtonschulz5(dw0)
        dw1 = zeropower_via_newtonschulz5(dw1)
        dw2 = zeropower_via_newtonschulz5(dw2)

    if gain is not None:
        gg = gain.to(dw0.dtype)[:, None, None]
        dw0, dw1, dw2 = dw0 * gg, dw1 * gg, dw2 * gg

    n0, n1, n2 = ref_norms
    raw = None
    if cfg.norm == "pre":
        b0, b1, b2 = state.raw if state.raw is not None else (w0, w1, w2)
        raw = (b0 + dw0, b1 + dw1, b2 + dw2)
        new = raw
    else:
        new = (w0 + dw0, w1 + dw1, w2 + dw2)
    return FastWeightState(
        renorm_rows(new[0], n0),
        renorm_rows(new[1], n1),
        renorm_rows(new[2], n2),
        m0,
        m1,
        m2,
        state.replay,
        raw,
    )


def segmented_lact(
    state: FastWeightState,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lr: torch.Tensor,
    momentum: Optional[torch.Tensor],
    segments: Sequence[tuple[int, int]],
    update_flags: Sequence[bool],
    ref_norms: Sequence[torch.Tensor],
    cfg: UpdateConfig,
    record: Optional[list] = None,
    borrowed: Optional[Sequence[FastWeightState]] = None,
    scores: Optional[torch.Tensor] = None,
    gains: Optional[torch.Tensor] = None,
):
    """Run apply-then-update over consecutive segments.

    q, k, v: [BH, L, d]; lr: [BH, L, 3]; momentum: [BH, L, 1] or None.
    record: if a list, the state each segment was read with is appended to it.
    borrowed: states recorded on another sequence; each segment is read with those and nothing is written.
    Returns (output [BH, L, d], final state).
    """
    outputs = []
    replay = state.replay
    if replay is not None and getattr(replay, "init_weights", 0) is None:
        replay.init_weights = tuple(w.detach() for w in state.weights())
    for i, ((s, e), do_update) in enumerate(zip(segments, update_flags)):
        if borrowed is not None:
            state = borrowed[i]
        if record is not None:
            record.append(state)
        outputs.append(fw_apply(*state.weights(), q[:, s:e]))
        if not do_update or borrowed is not None:
            continue

        ks, vs = k[:, s:e], v[:, s:e]
        lrs = lr[:, s:e]
        mom = None if momentum is None else momentum[:, s:e].mean(dim=1, keepdim=True)

        if replay is not None:
            ks_w, vs_w, lrs_w = replay.augment_write(ks, vs, lrs, state)
        else:
            ks_w, vs_w, lrs_w = ks, vs, lrs

        prev = state
        state = fw_update(
            state,
            ks_w,
            vs_w,
            lrs_w[..., 0:1],
            lrs_w[..., 1:2],
            lrs_w[..., 2:3],
            mom,
            ref_norms,
            cfg,
            None if gains is None else gains[:, i],
        )
        if replay is not None:
            state = replay.correct(prev, state, ks, ref_norms)
            replay.observe(ks, vs, lrs, None if scores is None else scores[:, s:e])

    return torch.cat(outputs, dim=1), state


def fixed_segments(length: int, chunk_size: int) -> list[tuple[int, int]]:
    return [(s, min(s + chunk_size, length)) for s in range(0, length, chunk_size)]
