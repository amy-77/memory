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
