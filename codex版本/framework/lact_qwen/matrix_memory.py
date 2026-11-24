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
# 稳定的分块形式(衰减没有上限)。上面的 `delta_write` 要乘 1 / P_i,它在块内按 exp(-sum lam) 增长,
# 所以需要 MAX_DECAY。这里只出现 s <= t 的比值 P_t / P_s,按 exp(G_t - G_s) 计算,其中 G = cumsum(lam),
# 因此每个因子都 <= 1。记 G_t 为块内到第 t 个 token(含)的累积 log 衰减,S_0 为进入这一块时的状态:
#
#     L[t, s] = sum_k u_tk u_sk exp(G_tk - G_sk)       (s < t,其余为零)
#     (I + L) E = Z - (exp(G) * U) S_0                 (单位下三角求解)
#     S_C = exp(G_C) * S_0 + (exp(G_C - G) * U)^T E
#
# 这与逐 token 规则完全等价(MemT-1.0 报告第 4 节:"chunked WY")。两两衰减张量 [C, C, dk]
# 限制了块的大小;取 64 可以让它保持较小。
# ---------------------------------------------------------------------------------------------------


def delta_block_stable(state: torch.Tensor, u: torch.Tensor, z: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
    """一个块的 token。state [N, dk, dv];u, lam [N, C, dk];z [N, C, dv];lam <= 0。"""
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
    """对任意 lam <= 0,结果与 `delta_write_reference` 相同(不需要 MAX_DECAY)。

    state: [B, dk, dv]   u, lam: [B, L, dk]   z: [B, L, dv]   (float32)
    开着 autograd 时,每 `group` 个 token 在反向时重算(激活 checkpoint),
    所以只保存进入每组时的状态,而不是每个块的两两衰减张量。"""
    track = torch.is_grad_enabled() and any(t.requires_grad for t in (state, u, z, lam))
    for s in range(0, u.shape[1], group):
        args = (state, u[:, s : s + group], z[:, s : s + group], lam[:, s : s + group])
        if track:
            state = torch.utils.checkpoint.checkpoint(_write_group, *args, block, use_reentrant=False)
        else:
            state = _write_group(*args, block)
    return state


# delta_write_auto 里,256-token 的块使用快公式所允许的最大总衰减。这是快公式一直运行的范围
# (每 token MAX_DECAY、共 256 个 token):1 / P_i <= exp(7.68) ~ 2200。
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
    """对任意 lam <= 0 都精确,衰减小时速度快。每 `block` 个 token 一块:若各通道的总衰减 sum(-lam)
    都不超过 `max_log_growth`,用快公式(`delta_write`,快约 9 倍),否则用稳定公式。
    两者计算的是同一个递推,没有任何截断。
    `safe`(每块一个布尔值)可以跳过 GPU -> CPU 的判断;错误的计划只会改变速度,或让快公式溢出,
    所以它必须由同一个 lam 算出(见 LaCTMemoryBranch._auto_plan)。"""
    length = u.shape[1]
    starts = list(range(0, length, block))
    if safe is None:  # 这次调用做一次 GPU -> CPU 同步;要写很多段的调用方会传入预先算好的计划
        with torch.no_grad():
            totals = [(-lam[:, s : s + block]).sum(dim=1).amax() for s in starts]
            safe = (torch.stack(totals) <= max_log_growth).tolist()
    if len(safe) != len(starts):
        raise ValueError(f"plan has {len(safe)} blocks, the write has {len(starts)}")
    for s, ok in zip(starts, safe):
        args = (state, u[:, s : s + block], z[:, s : s + block], lam[:, s : s + block])
        state = delta_write(*args, block) if ok else delta_write_stable(*args)
    return state
