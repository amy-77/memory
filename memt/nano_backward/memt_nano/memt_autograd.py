"""MemT-1.0-Nano 记忆分支(modeling_memt.py)的手写 backward。

三部分对应任务的三项:

1. ``GatedRMSNormFn``:对应 ``MemTRMSNormGated``(RMS 归一化、SiLU 输出门、第 51 行的 tanh 软上限)。
2. ``MemoryResidualFn``:对应 ``MemTAttention.forward`` 里的记忆部分(第 257-266 行)。记忆读出加到注意力输出上,
   更新后的状态交给 ``memory_call.results``;传入的读/写状态可能在另一张卡上,它们的梯度会送回那张卡。
3. ``MemTMemoryFn``(以及 ``memory_forward`` / ``memory_backward``):对应 ``MemTMemory.forward``(第 92-132 行),
   包括带 sqrt 擦除门的 delta 规则写入。两种写入算法计算的是同一个递推:
   - ``writer="block"``(默认):分块 WY 形式(MemT 训练时的做法,见报告第 4 节)。每 ``ckpt_every`` 个 token 一组:
     组内总衰减小时用快公式一次求解;否则用稳定公式,每 64 个 token 做一次矩阵乘法加三角求解,
     只出现衰减比 exp(G_t - G_s) <= 1,衰减多大都可以。前向保存每组开始时的 FP32 状态;
     反向从存档重算这一组,组内用 autograd 求导。
   - ``writer="token"``:官方的逐 token 循环,配手写的逐 token backward(保留作参照)。

除写入算法外,前向的每一步运算都照抄官方代码(同样的类型转换、同样的顺序):读出与 modeling_memt.py 逐位相同;
写入后的状态,block 算法与官方相差 FP32 舍入量级,token 算法逐位相同。反向在 FP32 下计算。
前向里的类型转换在反向时当作恒等(与 autograd 一致);不同的是,这里不会像 BF16 下的 autograd 那样
把中间梯度舍入到 BF16。

一步写入的递推(每个 batch、每个头;S 为 K x V,u、d 属于 R^K,w 属于 R^V):
    P_t = diag(d_t) S_{t-1}
    S_t = P_t + u_t (w_t - P_t^T u_t)^T
一步的反向,记 G = dL/dS_t:
    g     = G^T u_t                       -> dL/dw_t = g
    dL/du_t = G (w_t - P_t^T u_t) - P_t g
    dP    = G - u_t g^T
    dL/dd_t = rowsum(dP * S_{t-1})
    dL/dS_{t-1} = diag(d_t) dP"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

_L2_EPS = 1e-6  # 对应 MemTMemory._normalize


@dataclass(frozen=True)
class MemoryConfig:
    num_heads: int
    head_dim: int
    read_scale: float
    norm_eps: float
    softcap: bool
    limit: float
    ckpt_every: int = 256  # 保存写入状态的间隔(token 数)
    writer: str = "block"  # "block"(分块 WY)或 "token"(官方逐 token 循环,逐 token 反向)
    block: int = 64  # block 算法里稳定公式每次三角求解的 token 数


# --------------------------------------------------------------------------------------------------------------------
# 1. 带软上限的门控 RMS 归一化(MemTRMSNormGated,第 36-52 行)
# --------------------------------------------------------------------------------------------------------------------


def gated_rmsnorm_forward(value, gate, weight, eps, softcap, limit):
    """与 MemTRMSNormGated.forward 的算式相同。"""
    normalized = value.float() * torch.rsqrt(value.float().square().mean(-1, keepdim=True) + eps)
    normalized = weight * normalized.to(value.dtype)
    activated = F.silu(gate.float())
    if softcap:
        activated = limit * torch.tanh(activated / limit)
    return (normalized * activated).to(value.dtype)


def _gated_rmsnorm_grads(grad, value, gate, weight, eps, softcap, limit):
    """out = (weight * n) * a 的 FP32 梯度,其中
        n = v * rstd,  rstd = (mean(v^2) + eps)^(-1/2)
        a = c * tanh(silu(g) / c)      (没有软上限时 a = silu(g))
    d weight = 对所有行求和 grad * a * n
    d g      = grad * weight * n * (1 - tanh^2(silu(g) / c)) * sigmoid(g) * (1 + g * (1 - sigmoid(g)))
    d v      = rstd * (dn - n * mean(dn * n)),   dn = grad * a * weight"""
    grad = grad.float()
    v = value.float()
    rstd = torch.rsqrt(v.square().mean(-1, keepdim=True) + eps)
    n = v * rstd
    n_used = n.to(value.dtype).float()  # 前向乘的是舍入后的归一化值
    w = weight.float()
    g = gate.float()
    sig = torch.sigmoid(g)
    act = g * sig
    if softcap:
        t = torch.tanh(act / limit)
        a = limit * t
    else:
        a = act
    d_weight = (grad * a * n_used).reshape(-1, n.shape[-1]).sum(0)
    d_a = grad * w * n_used
    if softcap:
        d_a = d_a * (1 - t.square())
    d_gate = d_a * (sig * (1 + g * (1 - sig)))
    d_n = grad * a * w
    d_value = rstd * (d_n - n * (d_n * n).mean(-1, keepdim=True))
    return d_value, d_gate, d_weight


def gated_rmsnorm_backward(grad, value, gate, weight, eps, softcap, limit):
    d_value, d_gate, d_weight = _gated_rmsnorm_grads(grad, value, gate, weight, eps, softcap, limit)
    return d_value.to(value.dtype), d_gate.to(gate.dtype), d_weight.to(weight.dtype)


class GatedRMSNormFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, gate, weight, eps, softcap, limit):
        ctx.save_for_backward(value, gate, weight)
        ctx.args = (eps, softcap, limit)
        return gated_rmsnorm_forward(value, gate, weight, eps, softcap, limit)

    @staticmethod
    def backward(ctx, grad):
        value, gate, weight = ctx.saved_tensors
        d_value, d_gate, d_weight = gated_rmsnorm_backward(grad, value, gate, weight, *ctx.args)
        return d_value, d_gate, d_weight, None, None, None


# --------------------------------------------------------------------------------------------------------------------
# 3a. delta 规则写入(MemTMemory.forward,第 125-131 行)
# --------------------------------------------------------------------------------------------------------------------


def delta_rule_step(state, u, write_value, decay):
    """写入循环里的一个 token(第 128-131 行),运算相同。"""
    state = state * decay.unsqueeze(-1)
    erase = (u.unsqueeze(-1) * state).sum(-2)
    return state + u.unsqueeze(-1) * (write_value - erase).unsqueeze(-2)


def delta_rule_forward(state, key, write_value, decay, every):
    """key, decay: [B, L, H, K];write_value: [B, L, H, V];state: [B, H, K, V]。
    返回最终状态,以及进入第 0、every、2*every、... 个 token 时的状态。"""
    checkpoints = []
    for index in range(key.shape[1]):
        if index % every == 0:
            checkpoints.append(state)
        state = delta_rule_step(state, key[:, index], write_value[:, index], decay[:, index])
    return state, checkpoints


def delta_rule_backward(grad_state, checkpoints, key, write_value, decay, every):
    """返回 dL/d(初始状态)、dL/dkey、dL/dwrite_value、dL/ddecay(公式见模块开头的说明)。"""
    length = key.shape[1]
    d_key = torch.zeros_like(key)
    d_write = torch.zeros_like(write_value)
    d_decay = torch.zeros_like(decay)
    grad = grad_state
    for block in reversed(range(len(checkpoints))):
        start, stop = block * every, min(length, (block + 1) * every)
        states = [checkpoints[block]]
        for index in range(start, stop - 1):
            states.append(
                delta_rule_step(states[-1], key[:, index], write_value[:, index], decay[:, index])
            )
        for index in reversed(range(start, stop)):
            previous = states[index - start]
            u, w, d = key[:, index], write_value[:, index], decay[:, index]
            decayed = previous * d.unsqueeze(-1)
            erase = torch.einsum("bhk,bhkv->bhv", u, decayed)
            g = torch.einsum("bhk,bhkv->bhv", u, grad)
            d_write[:, index] = g
            d_key[:, index] = torch.einsum("bhkv,bhv->bhk", grad, w - erase) - torch.einsum(
                "bhkv,bhv->bhk", decayed, g
            )
            d_decayed = grad - u.unsqueeze(-1) * g.unsqueeze(-2)
            d_decay[:, index] = (d_decayed * previous).sum(-1)
            grad = d_decayed * d.unsqueeze(-1)
        del states
    return grad, d_key, d_write, d_decay


def delta_block_stable(state, u, z, lam):
    """写入的一个块,分块 WY 形式,对任意 lam <= 0 都精确。
    state [N, K, V];u, lam [N, C, K];z [N, C, V]。记 G = 块内 cumsum(lam):
        L[t, s] = sum_k u_tk u_sk exp(G_tk - G_sk)   (s < t)
        (I + L) E = Z - (exp(G) * U) S_0
        S_C = exp(G_C) * S_0 + (exp(G_C - G) * U)^T E
    只对非正数取 exp(衰减比 <= 1)。"""
    c = u.shape[1]
    g = lam.cumsum(dim=1)
    pair = g[:, :, None, :] - g[:, None, :, :]  # [N, t, s, K] = G_t - G_s
    below = torch.ones(c, c, dtype=torch.bool, device=u.device).tril(-1)
    pair = pair.masked_fill(~below[None, :, :, None], float("-inf"))
    lower = torch.einsum("ntk,nsk,ntsk->nts", u, u, pair.exp())
    rhs = z - (u * g.exp()) @ state
    e = torch.linalg.solve_triangular(lower, rhs, upper=False, unitriangular=True)
    last = g[:, -1:, :]
    return state * last.transpose(1, 2).exp() + (u * (last - g).exp()).transpose(1, 2) @ e


def delta_block_scaled(state, u, z, lam):
    """同一个块的快公式(带 1 / P_t 因子)。比稳定公式快约 9 倍,但 1 / P_t 会按 exp(块内总衰减) 增长,
    所以只在总衰减不超过 FAST_MAX_LOG_GROWTH 时使用。"""
    grow = torch.exp(-lam.cumsum(dim=1))  # 1 / P_t
    a, b = u * grow, u / grow
    lower = torch.tril(b @ a.transpose(1, 2), diagonal=-1)
    e = torch.linalg.solve_triangular(lower, z - b @ state, upper=False, unitriangular=True)
    return (state + a.transpose(1, 2) @ e) / grow[:, -1, :, None]


# 快公式允许的一组内最大总衰减:1 / P_t <= exp(7.68) ~ 2200(每 token 0.03、共 256 个 token,
# 即 4B 框架的快公式一直运行的范围;此范围内 FP32 误差低于 1e-4)。
FAST_MAX_LOG_GROWTH = 0.03 * 256


def _write_group(state, u, z, lam, block):
    """一组 token(最多 `ckpt_every` 个):各通道总衰减都小时,用快公式一次求解;
    否则用稳定公式,每 `block` 个 token 一块。两者都精确,没有任何截断。"""
    if u.shape[1] <= 256 and float((-lam).sum(dim=1).amax()) <= FAST_MAX_LOG_GROWTH:
        return delta_block_scaled(state, u, z, lam)
    for s in range(0, u.shape[1], block):
        state = delta_block_stable(state, u[:, s : s + block], z[:, s : s + block], lam[:, s : s + block])
    return state


def _heads_first(t):
    """[B, L, H, D] -> [B * H, L, D]"""
    b, length, h, d = t.shape
    return t.permute(0, 2, 1, 3).reshape(b * h, length, d)


def block_write_forward(state, key, write_value, lam, block, every):
    """state [B, H, K, V];key, lam [B, L, H, K];write_value [B, L, H, V](FP32)。
    返回最终状态 [B, H, K, V],以及进入第 0、every、2 * every、... 个 token 时的状态 [B * H, K, V]。"""
    b, h = state.shape[:2]
    u, z, g = _heads_first(key), _heads_first(write_value), _heads_first(lam)
    s = state.reshape(b * h, *state.shape[2:])
    checkpoints = []
    for start in range(0, u.shape[1], every):
        checkpoints.append(s)
        s = _write_group(s, u[:, start : start + every], z[:, start : start + every], g[:, start : start + every], block)
    return s.reshape(state.shape), checkpoints


def block_write_backward(grad_state, checkpoints, key, write_value, lam, block, every):
    """block_write_forward 的梯度:d state、d key、d write_value、d lam。每组 `every` 个 token
    从保存的状态重算,组内用 autograd 求导(MemT 报告:"reconstructing intermediate quantities
    during backward")。"""
    b, length, h, _ = key.shape
    u, z, g = _heads_first(key), _heads_first(write_value), _heads_first(lam)
    d_u, d_z, d_g = torch.zeros_like(u), torch.zeros_like(z), torch.zeros_like(g)
    grad = grad_state.reshape(b * h, *grad_state.shape[2:])
    for index in reversed(range(len(checkpoints))):
        start, stop = index * every, min(length, (index + 1) * every)
        with torch.enable_grad():
            inputs = [t.detach().requires_grad_() for t in
                      (checkpoints[index], u[:, start:stop], z[:, start:stop], g[:, start:stop])]
            out = _write_group(*inputs, block)
            grad, d_u[:, start:stop], d_z[:, start:stop], d_g[:, start:stop] = torch.autograd.grad(
                out, inputs, grad
            )

    def back(t, width):  # [B * H, L, D] -> [B, L, H, D]
        return t.reshape(b, h, length, width).permute(0, 2, 1, 3)

    return grad.reshape(grad_state.shape), back(d_u, u.shape[-1]), back(d_z, z.shape[-1]), back(d_g, g.shape[-1])


class BlockWriteFn(torch.autograd.Function):
    """只包含 block 写入(测试用;MemTMemoryFn 调用的是同一组函数)。"""

    @staticmethod
    def forward(ctx, state, key, write_value, lam, block, every):
        final, checkpoints = block_write_forward(state, key, write_value, lam, block, every)
        ctx.args = (block, every)
        ctx.save_for_backward(key, write_value, lam, *checkpoints)
        return final

    @staticmethod
    def backward(ctx, grad):
        key, write_value, lam, *checkpoints = ctx.saved_tensors
        d_state, d_key, d_write, d_lam = block_write_backward(grad, checkpoints, key, write_value, lam, *ctx.args)
        return d_state, d_key, d_write, d_lam, None, None


class DeltaRuleFn(torch.autograd.Function):
    """只包含逐 token 写入递推(测试用;MemTMemoryFn 调用的是同一组函数)。"""

    @staticmethod
    def forward(ctx, state, key, write_value, decay, every):
        final, checkpoints = delta_rule_forward(state, key, write_value, decay, every)
        ctx.every = every
        ctx.save_for_backward(key, write_value, decay, *checkpoints)
        return final

    @staticmethod
    def backward(ctx, grad):
        key, write_value, decay, *checkpoints = ctx.saved_tensors
        d_state, d_key, d_write, d_decay = delta_rule_backward(
            grad, checkpoints, key, write_value, decay, ctx.every
        )
        return d_state, d_key, d_write, d_decay, None


# --------------------------------------------------------------------------------------------------------------------
# 3b. 整个记忆模块(MemTMemory.forward,第 92-132 行)
# --------------------------------------------------------------------------------------------------------------------
# params = (in_proj.weight, f_down.weight, g_down.weight, g_down.bias 或 None, decay_bias, decay_scale_log,
#           o_norm.weight)。in_proj 的行依次是:q、k、v、b、w(各 value_dim 行),f、g(各 head_dim 行)。


def _normalize(value, rounded):
    result = value.float() * torch.rsqrt(value.float().square().sum(-1, keepdim=True) + _L2_EPS)
    return result.to(value.dtype) if rounded else result


def _l2norm_backward(d_normalized, normalized, rstd):
    """n = x * rstd, rstd = (sum(x^2) + eps)^(-1/2):  dx = rstd * (dn - n * sum(dn * n))."""
    return rstd * (d_normalized - normalized * (d_normalized * normalized).sum(-1, keepdim=True))


def memory_forward(hidden, params, snapshot, write_state, commit, cfg, with_checkpoints=False):
    """给定快照时的 MemTMemory.forward(state=None 且不写入的情况由调用方处理)。
    返回读出 [B, L, H*D]、更新后的状态(或 None)、写入的存档(或 None)。"""
    in_w, f_w, g_w, g_b, decay_bias, decay_scale_log, norm_w = params
    batch, length, _ = hidden.shape
    heads, dim = cfg.num_heads, cfg.head_dim
    width = heads * dim
    shape = (batch, length, heads, dim)
    projected = F.linear(hidden, in_w[: 3 * width if commit else width])
    query = _normalize(projected[..., :width].reshape(shape), commit)
    with torch.autocast(device_type=hidden.device.type, enabled=False):
        read = cfg.read_scale * torch.einsum("blhk,bhkv->blhv", query.float(), snapshot.float())
    gate = F.linear(F.linear(hidden, in_w[-dim:]), g_w, g_b).reshape(shape)
    read = gated_rmsnorm_forward(
        read.to(gate.dtype), gate, norm_w, cfg.norm_eps, cfg.softcap, cfg.limit
    ).reshape(batch, length, width)
    if not commit:
        return read, None, None
    key, value = (item.reshape(shape) for item in projected[..., width:].split(width, -1))
    key = _normalize(key, True)
    b, w, f = F.linear(hidden, in_w[3 * width : -dim]).split((width, width, dim), -1)
    sqrt_b = (0.5 * F.logsigmoid(b.float())).exp().to(b.dtype).reshape(shape)
    key = (key.float() * sqrt_b.float()).to(value.dtype).float()
    write_value = w.sigmoid().reshape(shape).float() * value.float()
    log_decay = (
        -decay_scale_log.float().exp().reshape(heads, 1)
        * F.softplus(F.linear(f, f_w).float() + decay_bias.float()).reshape(shape)
    )
    decay = log_decay.exp()  # 与官方表达式的运算相同
    start = snapshot.float() if write_state is None else write_state.float()
    if with_checkpoints and cfg.writer == "block":
        updated, checkpoints = block_write_forward(start, key, write_value, log_decay, cfg.block, cfg.ckpt_every)
        return read, updated, checkpoints
    if with_checkpoints:
        updated, checkpoints = delta_rule_forward(start, key, write_value, decay, cfg.ckpt_every)
        return read, updated, checkpoints
    updated = start
    for index in range(length):
        updated = delta_rule_step(updated, key[:, index], write_value[:, index], decay[:, index])
    return read, updated, None


def memory_backward(grad_read, grad_updated, hidden, params, snapshot, write_state, checkpoints, commit, cfg):
    """memory_forward 的梯度。grad_read / grad_updated 可以是 None(该输出没被用到)。
    返回 d_hidden、d_params(顺序同 params)、d_snapshot、d_write_state。"""
    in_w, f_w, g_w, g_b, decay_bias, decay_scale_log, norm_w = params
    batch, length, _ = hidden.shape
    heads, dim = cfg.num_heads, cfg.head_dim
    width = heads * dim
    shape = (batch, length, heads, dim)
    flat = (batch, length, width)
    x = hidden.float()
    snap = snapshot.float()

    # ---- 重算前向的各个量(投影很便宜;状态取自存档) ----
    projected = F.linear(hidden, in_w[: 3 * width if commit else width])
    q_raw = projected[..., :width].reshape(shape).float()
    q_rstd = torch.rsqrt(q_raw.square().sum(-1, keepdim=True) + _L2_EPS)
    q_hat = q_raw * q_rstd
    q_used = q_hat.to(projected.dtype).float() if commit else q_hat
    gate_in = F.linear(hidden, in_w[-dim:])
    gate = F.linear(gate_in, g_w, g_b).reshape(shape)

    # ---- 读路径:read = o_norm(scale * q_hat S_snapshot, gate) ----
    if grad_read is not None:
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            read_raw = cfg.read_scale * torch.einsum("blhk,bhkv->blhv", q_used, snap)
        d_read_raw, d_gate, d_norm_w = _gated_rmsnorm_grads(
            grad_read.reshape(shape), read_raw.to(gate.dtype), gate, norm_w, cfg.norm_eps, cfg.softcap, cfg.limit
        )
        d_q_used = cfg.read_scale * torch.einsum("blhv,bhkv->blhk", d_read_raw, snap)
        d_snapshot = cfg.read_scale * torch.einsum("blhk,blhv->bhkv", q_used, d_read_raw)
        d_q = _l2norm_backward(d_q_used, q_hat, q_rstd).reshape(flat)
        d_gate = d_gate.reshape(flat)
        d_g_w = torch.einsum("blo,bli->oi", d_gate, gate_in.float())
        d_g_b = d_gate.sum((0, 1)) if g_b is not None else None
        d_gate_in = d_gate @ g_w.float()
    else:  # 读出没被用到:只属于读路径的参数梯度为 None,与 autograd 一致
        d_snapshot = torch.zeros_like(snap)
        d_q = x.new_zeros(flat)
        d_norm_w = d_g_w = d_g_b = None
        d_gate_in = x.new_zeros(batch, length, dim)

    d_f_w = d_decay_bias = d_scale_log = None  # 只在写入时用到的参数:不写入时梯度为 None
    d_write_state = None
    x_2d = x.reshape(-1, x.shape[-1])
    if not commit:
        # 只用到了 in_proj 的 q 行和 g 行
        d_in_w = torch.zeros(in_w.shape, dtype=torch.float32, device=in_w.device)
        d_in_w[:width] = d_q.reshape(-1, width).t() @ x_2d
        d_in_w[-dim:] = d_gate_in.reshape(-1, dim).t() @ x_2d
        d_hidden = d_q @ in_w[:width].float() + d_gate_in @ in_w[-dim:].float()
    else:
        # ---- 写路径 ----
        k_raw = projected[..., width : 2 * width].reshape(shape)
        value = projected[..., 2 * width :].reshape(shape)
        k_rstd = torch.rsqrt(k_raw.float().square().sum(-1, keepdim=True) + _L2_EPS)
        k_hat = k_raw.float() * k_rstd
        k_used = k_hat.to(k_raw.dtype).float()
        b, w, f = F.linear(hidden, in_w[3 * width : -dim]).split((width, width, dim), -1)
        b32 = b.float()
        sqrt_b = (0.5 * F.logsigmoid(b32)).exp()
        sqrt_b_used = sqrt_b.to(b.dtype).float().reshape(shape)
        u = (k_used * sqrt_b_used).to(value.dtype).float()
        sig_w = w.sigmoid().reshape(shape).float()
        write_value = sig_w * value.float()
        z = F.linear(f, f_w).float() + decay_bias.float()
        softplus_z = F.softplus(z)
        scale = decay_scale_log.float().exp().reshape(heads, 1)
        log_decay = -scale * softplus_z.reshape(shape)
        decay = log_decay.exp()

        # d_exponent = dL / d(log 衰减)
        if grad_updated is None:
            d_start = torch.zeros_like(snap)
            d_u = torch.zeros_like(u)
            d_wv = torch.zeros_like(write_value)
            d_exponent = torch.zeros_like(decay)
        elif cfg.writer == "block":
            d_start, d_u, d_wv, d_exponent = block_write_backward(
                grad_updated.float(), checkpoints, u, write_value, log_decay, cfg.block, cfg.ckpt_every
            )
        else:
            d_start, d_u, d_wv, d_decay = delta_rule_backward(
                grad_updated.float(), checkpoints, u, write_value, decay, cfg.ckpt_every
            )
            d_exponent = d_decay * decay

        # u = k_hat * sqrt(sigmoid(b))
        d_k = _l2norm_backward(d_u * sqrt_b_used, k_hat, k_rstd).reshape(flat)
        d_b = (d_u * k_used).reshape(flat) * (0.5 * sqrt_b * torch.sigmoid(-b32))
        # write_value = sigmoid(w) * v
        d_v = (d_wv * sig_w).reshape(flat)
        d_w = (d_wv * value.float() * sig_w * (1 - sig_w)).reshape(flat)
        # decay = exp(-exp(scale_log) * softplus(f_down(f) + decay_bias))
        d_scale_log = -(d_exponent * scale * softplus_z.reshape(shape)).sum((0, 1, 3))
        softplus_grad = torch.where(z > 20, torch.ones_like(z), torch.sigmoid(z))  # F.softplus 的阈值 20
        d_z = (-(d_exponent * scale)).reshape(flat) * softplus_grad
        d_decay_bias = d_z.sum((0, 1))
        d_f_w = torch.einsum("blo,bli->oi", d_z, f.float())
        d_f = d_z @ f_w.float()
        if write_state is None:
            d_snapshot = d_snapshot + d_start
        else:
            d_write_state = d_start
        # ---- in_proj:行 = [q, k, v, b, w, f, g] ----
        d_rows = torch.cat([d_q, d_k, d_v, d_b, d_w, d_f, d_gate_in], -1)
        d_in_w = d_rows.reshape(-1, d_rows.shape[-1]).t() @ x_2d
        d_hidden = d_rows @ in_w.float()

    def cast(grad, like):
        return None if grad is None or like is None else grad.to(like.dtype)

    d_params = (
        cast(d_in_w, in_w),
        cast(d_f_w, f_w),
        cast(d_g_w, g_w),
        cast(d_g_b, g_b),
        cast(d_decay_bias, decay_bias),
        cast(d_scale_log, decay_scale_log),
        cast(d_norm_w, norm_w),
    )
    return (
        d_hidden.to(hidden.dtype),
        d_params,
        d_snapshot.to(snapshot.dtype),
        cast(d_write_state, write_state),
    )


class MemTMemoryFn(torch.autograd.Function):
    """给定快照时的 MemTMemory.forward。返回读出;写入时返回 (读出, 更新后的状态)。"""

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, hidden, snapshot, write_state, cfg, commit, *params):
        read, updated, checkpoints = memory_forward(
            hidden, params, snapshot, write_state, commit, cfg, with_checkpoints=commit
        )
        ctx.cfg, ctx.commit = cfg, commit
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(hidden, snapshot, write_state, *params, *(checkpoints or ()))
        return (read, updated) if commit else read

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, *grads):
        saved = ctx.saved_tensors
        hidden, snapshot, write_state = saved[:3]
        params, checkpoints = saved[3:10], saved[10:]
        grad_read = grads[0]
        grad_updated = grads[1] if ctx.commit else None
        if grad_read is None and grad_updated is None:
            return (None,) * (5 + len(params))
        d_hidden, d_params, d_snapshot, d_write_state = memory_backward(
            grad_read, grad_updated, hidden, params, snapshot, write_state, checkpoints, ctx.commit, ctx.cfg
        )
        return (d_hidden, d_snapshot, d_write_state, None, None, *d_params)


# --------------------------------------------------------------------------------------------------------------------
# 2. MemTAttention.forward 里的记忆部分(第 257-266 行)
# --------------------------------------------------------------------------------------------------------------------


class MemoryResidualFn(torch.autograd.Function):
    """输出 + 记忆读出,以及更新后的状态,合成一个节点:

        state  = memory_call.read[layer].to(device)        (没有且要写入时为全零)
        writer = memory_call.write[layer].to(device)        (没有时为 None)
        read, updated = memory(hidden, state, commit, write_state=writer)
        output = output + read.to(output.dtype)

    反向:d output 原样传回;d read = d output;memory_backward 给出 d hidden、d params,
    以及 state / writer 的梯度,这两个梯度会送回各自张量原来所在的设备。"""

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, output, hidden, state, writer, cfg, commit, *params):
        device = hidden.device
        ctx.state_device = None if state is None else state.device
        ctx.writer_device = None if writer is None else writer.device
        if state is None:
            snapshot = hidden.new_zeros(
                (hidden.shape[0], cfg.num_heads, cfg.head_dim, cfg.head_dim), dtype=torch.float32
            )
        else:
            snapshot = state.to(device)
        writer = None if writer is None else writer.to(device)
        read, updated, checkpoints = memory_forward(
            hidden, params, snapshot, writer, commit, cfg, with_checkpoints=commit
        )
        ctx.cfg, ctx.commit = cfg, commit
        ctx.read_dtype = read.dtype
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(hidden, snapshot, writer, *params, *(checkpoints or ()))
        result = output + read.to(output.dtype)
        return (result, updated) if commit else result

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, *grads):
        saved = ctx.saved_tensors
        hidden, snapshot, writer = saved[:3]
        params, checkpoints = saved[3:10], saved[10:]
        grad_output = grads[0]
        grad_updated = grads[1] if ctx.commit else None
        grad_read = None if grad_output is None else grad_output.to(ctx.read_dtype)
        if grad_read is None and grad_updated is None:
            return (None,) * (6 + len(params))
        d_hidden, d_params, d_snapshot, d_writer = memory_backward(
            grad_read, grad_updated, hidden, params, snapshot, writer, checkpoints, ctx.commit, ctx.cfg
        )
        d_state = None if ctx.state_device is None else d_snapshot.to(ctx.state_device)
        d_writer = None if ctx.writer_device is None or d_writer is None else d_writer.to(ctx.writer_device)
        return (grad_output, d_hidden, d_state, d_writer, None, None, *d_params)
