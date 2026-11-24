"""Matrix memory: every token is written into one fixed-size matrix per head (gated delta rule).

The state S of a head is a [dk, dv] matrix, zero at the start of a sequence. A token with key k
(unit length), value v, erase gate beta, write gate omega (both in (0, 1), per channel) and log-decay
lam (<= 0, per key channel) changes it as

    u = sqrt(beta) * k
    S <- D S + u (omega * v - u^T D S)^T,        D = diag(exp(lam))

that is: let the old content fade a little (D), remove what the state currently returns for this key
(the u^T D S term) and store the new value there. Because |u| <= 1 the state cannot blow up. A read
with query q returns q^T S / sqrt(dk).

This is the update rule described in the MemT-1.0 technical report (arXiv 2610.26825, section 2.2.2,
built on Gated DeltaNet). The code below is written from the formulas in the report.

Tokens depend on each other, so a plain implementation is one small step per token. `delta_write`
gives the same result for a block of tokens with a few matrix products and one triangular solve:
with P_i = D_i ... D_1 and S_i = P_i T_i the rule becomes T_i = T_{i-1} + a_i e_i^T with
a_i = u_i / P_i, b_i = P_i u_i and e_i = z_i - T_{i-1}^T b_i, and the e_i of a block solve
(I + L) E = Z - B T_0, where L is the strictly lower triangle of B A^T.
"""

from __future__ import annotations

import torch
import torch.utils.checkpoint

# Largest per-token decay (-lam). 0.03 is a half-life of 23 tokens; the base model sees the last
# 2048 tokens anyway, so nothing needs to be forgotten faster. The bound also keeps 1 / P_i in
# `delta_write` small (exp(0.03 * 256) ~ 2000 for the default block of 256 tokens).
MAX_DECAY = 0.03


def delta_write_reference(state: torch.Tensor, u: torch.Tensor, z: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
    """One token at a time, straight from the definition. Slow; used by the tests."""
    for i in range(u.shape[1]):
        ui = u[:, i]
        decayed = state * torch.exp(lam[:, i])[:, :, None]
        returned = torch.einsum("bk,bkv->bv", ui, decayed)
        state = decayed + ui[:, :, None] * (z[:, i] - returned)[:, None, :]
    return state


def delta_write(
    state: torch.Tensor, u: torch.Tensor, z: torch.Tensor, lam: torch.Tensor, block: int = 256
) -> torch.Tensor:
    """Write a run of tokens into the state.

    state: [B, dk, dv]   u, lam: [B, L, dk]   z: [B, L, dv]   (all float32, lam in [-MAX_DECAY, 0])
    Returns the state after the last token.
    """
    for s in range(0, u.shape[1], block):
        us, zs = u[:, s : s + block], z[:, s : s + block]
        grow = torch.exp(-lam[:, s : s + block].cumsum(dim=1))  # 1 / P_i, >= 1
        a = us * grow
        b = us / grow
        rhs = zs - b @ state
        lower = torch.tril(b @ a.transpose(1, 2), diagonal=-1)
        e = torch.linalg.solve_triangular(lower, rhs, upper=False, unitriangular=True)
        state = (state + a.transpose(1, 2) @ e) / grow[:, -1, :, None]
    return state


# ---------------------------------------------------------------------------------------------------
# Stable block form (no bound on the decay). `delta_write` above multiplies by 1 / P_i, which grows
# like exp(-sum lam) inside a block and needs MAX_DECAY. Here only ratios P_t / P_s with s <= t appear,
# computed as exp(G_t - G_s) with G = cumsum(lam) (so every factor is <= 1). With G_t the cumulative
# log-decay inside the block (inclusive of t) and S_0 the state entering the block:
#
#     L[t, s] = sum_k u_tk u_sk exp(G_tk - G_sk)       (s < t, zero otherwise)
#     (I + L) E = Z - (exp(G) * U) S_0                 (unit lower triangular solve)
#     S_C = exp(G_C) * S_0 + (exp(G_C - G) * U)^T E
#
# which is the per-token rule exactly (MemT-1.0 report, section 4: "chunked WY"). The pairwise
# [C, C, dk] decay tensor bounds the block size; 64 keeps it small.
# ---------------------------------------------------------------------------------------------------


def delta_block_stable(state: torch.Tensor, u: torch.Tensor, z: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
    """One block of tokens. state [N, dk, dv]; u, lam [N, C, dk]; z [N, C, dv]; lam <= 0."""
    c = u.shape[1]
    g = lam.cumsum(dim=1)
    pair = g[:, :, None, :] - g[:, None, :, :]  # [N, t, s, dk] = G_t - G_s
    below = torch.ones(c, c, dtype=torch.bool, device=u.device).tril(-1)
    pair = pair.masked_fill(~below[None, :, :, None], float("-inf"))  # s >= t -> exp(-inf) = 0
    lower = torch.einsum("ntk,nsk,ntsk->nts", u, u, pair.exp())
    rhs = z - (u * g.exp()) @ state
    e = torch.linalg.solve_triangular(lower, rhs, upper=False, unitriangular=True)
    last = g[:, -1:, :]
    return state * last.transpose(1, 2).exp() + (u * (last - g).exp()).transpose(1, 2) @ e


def _write_group(state, u, z, lam, block):
    for s in range(0, u.shape[1], block):
        state = delta_block_stable(state, u[:, s : s + block], z[:, s : s + block], lam[:, s : s + block])
    return state


def delta_write_stable(
    state: torch.Tensor,
    u: torch.Tensor,
    z: torch.Tensor,
    lam: torch.Tensor,
    block: int = 64,
    group: int = 256,
) -> torch.Tensor:
    """Same result as `delta_write_reference` for any lam <= 0 (no MAX_DECAY needed).

    state: [B, dk, dv]   u, lam: [B, L, dk]   z: [B, L, dv]   (float32)
    With autograd on, every `group` tokens are recomputed in backward (activation checkpointing), so
    only the state entering each group is kept instead of the pairwise decay tensor of every block.
    """
    track = torch.is_grad_enabled() and any(t.requires_grad for t in (state, u, z, lam))
    for s in range(0, u.shape[1], group):
        args = (state, u[:, s : s + group], z[:, s : s + group], lam[:, s : s + group])
        if track:
            state = torch.utils.checkpoint.checkpoint(_write_group, *args, block, use_reentrant=False)
        else:
            state = _write_group(*args, block)
    return state


# Largest total decay of one 256-token block for which the scaled form is used in delta_write_auto. It is the
# range the scaled form has always run in (MAX_DECAY per token over 256 tokens): 1 / P_i <= exp(7.68) ~ 2200.
AUTO_MAX_LOG_GROWTH = MAX_DECAY * 256


def delta_write_auto(
    state: torch.Tensor,
    u: torch.Tensor,
    z: torch.Tensor,
    lam: torch.Tensor,
    block: int = 256,
    max_log_growth: float = AUTO_MAX_LOG_GROWTH,
    safe: list | None = None,
) -> torch.Tensor:
    """Exact for any lam <= 0, fast when decays are small. Each block of `block` tokens uses the scaled form
    (`delta_write`, ~9x faster) if its total decay sum(-lam) stays within `max_log_growth` in every channel,
    and the stable form otherwise. Both compute the same recurrence; nothing is clipped.
    `safe` (one bool per block) skips the GPU -> host check; a wrong plan only changes speed or overflows the
    scaled form, so it must come from the same lam (see LaCTMemoryBranch._auto_plan)."""
    length = u.shape[1]
    starts = list(range(0, length, block))
    if safe is None:  # one host sync for this call; callers that write many steps pass a precomputed plan
        with torch.no_grad():
            totals = [(-lam[:, s : s + block]).sum(dim=1).amax() for s in starts]
            safe = (torch.stack(totals) <= max_log_growth).tolist()
    if len(safe) != len(starts):
        raise ValueError(f"plan has {len(safe)} blocks, the write has {len(starts)}")
    for s, ok in zip(starts, safe):
        args = (state, u[:, s : s + block], z[:, s : s + block], lam[:, s : s + block])
        state = delta_write(*args, block) if ok else delta_write_stable(*args)
    return state
