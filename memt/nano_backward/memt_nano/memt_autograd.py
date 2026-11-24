"""MemT-1.0-Nano 记忆分支(modeling_memt.py)的手写 backward。

三部分对应任务的三项:

1. ``GatedRMSNormFn``:对应 ``MemTRMSNormGated``(RMS 归一化、SiLU 输出门、第 51 行的 tanh 软上限)。
2. ``MemoryResidualFn``:对应 ``MemTAttention.forward`` 里的记忆部分(第 257-266 行)。记忆读出加到注意力输出上,
   更新后的状态交给 ``memory_call.results``;传入的读/写状态可能在另一张卡上,它们的梯度会送回那张卡。
3. ``MemTMemoryFn``(以及 ``memory_forward`` / ``memory_backward``):对应 ``MemTMemory.forward``(第 92-132 行),
   包括带 sqrt 擦除门的 delta 规则写入。两种写入算法计算的是同一个递推:
   - ``writer="block"``(默认):分块 WY 形式(MemT 报告 4.2 节)。每 ``block`` 个 token 一块,块内只有矩阵乘和
     一次三角求解;衰减因子都写成"不超过 1 的两个因子相乘",所以衰减多大都不会溢出,不需要衰减上限。
     前向保存每块开始时的 FP32 状态;反向从存档重算中间量,再用手写的块反向公式求梯度,不调用 autograd。
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
    dL/dS_{t-1} = diag(d_t) dP

一个块(C 个 token)的分块 WY 形式。记 lam_t = log d_t,G_t = lam_1 + ... + lam_t(块内累加),U、Z 的第 t 行是 u_t、w_t:
    L[t, s] = sum_k u_tk u_sk exp(G_tk - G_sk)   (s < t,严格下三角)
    Ut      = exp(G) * U                          (第 t 行乘 exp(G_t))
    E       = (I + L)^-1 (Z - Ut S_0)             (三角求解;E 的第 t 行就是 w_t - P_t^T u_t)
    W       = exp(G_C - G) * U
    S_C     = exp(G_C) * S_0 + W^T E
L 的算法(``wy_lower``):把块对半分,右半的 t 和左半的 s 之间取参考点 r(左半最后一个 token),
    exp(G_t - G_s) = exp(G_t - G_r) * exp(G_r - G_s),两个因子都不超过 1,
于是这一层的 L 子块 = (右半的 u * 第一个因子) 乘 (左半的 u * 第二个因子)^T,是普通矩阵乘;
再对左右两半各自递归,共 log2(C) 层,每层对所有块、所有节点一次批量矩阵乘。
一个块的反向,记 G' = dL/dS_C(K x V):
    dE   = W G'                 dW = E G'^T
    dR   = (I + L)^-T dE        (转置的三角求解)   -> dL/dZ = dR
    dL_  = -dR E^T                                 (L 的梯度,只用严格下三角部分)
    dUt  = -dR S_0^T
    dL/dS_0 = exp(G_C) * G' - Ut^T dR
    X[t] = sum_s dL_[t, s] exp(G_t - G_s) * u_s,   Y[s] = sum_t dL_[t, s] exp(G_t - G_s) * u_t
           (``wy_lower_backward``:与 L 同样的分层,每层两次矩阵乘)
    dL/dU   = X + Y + exp(G) * dUt + exp(G_C - G) * dW
    dL/dG   = U * (X - Y) + Ut * dUt - W * dW,     最后一行再加 rowsum(G' * S_C)
    dL/dlam_t = dL/dG_t + dL/dG_{t+1} + ... + dL/dG_C"""

from __future__ import annotations

import functools
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
    ckpt_every: int = 256  # token 算法保存写入状态的间隔(token 数)
    writer: str = "block"  # "block"(分块 WY)或 "token"(官方逐 token 循环,逐 token 反向)
    block: int = 256  # block 算法的块长(2 的幂):每块一次三角求解,并存一份块起始状态


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


def _halves(size):
    """块长 size(2 的幂)的各层半长:size / 2、size / 4、...、1。"""
    half = size // 2
    while half >= 1:
        yield half
        half //= 2


def _level_operands(u, g):
    """wy_lower 各层矩阵乘的操作数。u [M, C, K];g [M, C, K] 是块内 cumsum(lam)(FP64)。
    半长为 half 的一层把块切成长 2 * half 的节点,参考点 r 取节点左半的最后一个 token。每层返回
        right = 右半的 u * exp(G_t - G_r),   left = 左半的 u * exp(G_r - G_s),
        f_right = exp(G_t - G_r),            f_left = exp(G_r - G_s),
    形状都是 [M, 节点数, half, K]。两个指数都不超过 0;差在 FP64 下算,再转回 u 的精度。"""
    m, c, k = u.shape
    levels = []
    for half in _halves(c):
        uv = u.reshape(m, c // (2 * half), 2 * half, k)
        gv = g.reshape(m, c // (2 * half), 2 * half, k)
        ref = gv[:, :, half - 1 : half]
        f_right = (gv[:, :, half:] - ref).to(u.dtype).exp()
        f_left = (ref - gv[:, :, :half]).to(u.dtype).exp()
        levels.append((uv[:, :, half:] * f_right, uv[:, :, :half] * f_left, f_right, f_left))
    return levels


def _node_blocks(matrix, nodes, half):
    """[M, C, C] 矩阵里各节点的"右半行 x 左半列"子块,作为视图返回,形状 [M, half, half, 节点数]。"""
    m = matrix.shape[0]
    return matrix.view(m, nodes, 2 * half, nodes, 2 * half).diagonal(dim1=1, dim2=3)[:, half:, :half]


def wy_lower(u, g, levels=None):
    """L[t, s] = sum_k u_tk u_sk exp(G_tk - G_sk)(s < t),其余为 0。
    u [M, C, K];g [M, C, K] 是块内 cumsum(lam)(FP64);C 是 2 的幂。算法见模块开头"L 的算法"。"""
    m, c, _ = u.shape
    lower = u.new_zeros(m, c, c)
    for right, left, _, _ in levels or _level_operands(u, g):
        _node_blocks(lower, right.shape[1], right.shape[2])[:] = (right @ left.transpose(-1, -2)).permute(0, 2, 3, 1)
    return lower


def wy_lower_backward(d_lower, u, g, levels=None):
    """wy_lower 的手写反向。d_lower [M, C, C] 是对 L 的梯度(只读严格下三角部分)。
    返回 (X + Y, X - Y):L 对 u 的梯度是 X + Y,对 G 的梯度是 u * (X - Y)。"""
    m, c, k = u.shape
    x, y = torch.zeros_like(u), torch.zeros_like(u)  # 每层里,X 落在右半的行上,Y 落在左半的行上
    for right, left, f_right, f_left in levels or _level_operands(u, g):
        nodes, half = right.shape[1], right.shape[2]
        d_piece = _node_blocks(d_lower, nodes, half).permute(0, 3, 1, 2)  # [M, 节点数, half, half]
        x.view(m, nodes, 2 * half, k)[:, :, half:] += (d_piece @ left) * f_right
        y.view(m, nodes, 2 * half, k)[:, :, :half] += (d_piece.transpose(-1, -2) @ right) * f_left
    return x + y, x - y


def _block_size(length, block):
    """实际块长:不超过 block、且够装下 length 的最小的 2 的幂(block 本身必须是 2 的幂)。"""
    assert block >= 1 and block & (block - 1) == 0, f"write block must be a power of two, got {block}"
    size = 1
    while size < min(block, length):
        size *= 2
    return size


def _wy_terms(u, z, lam, size):
    """把序列切成长 size 的块(末尾补零:u = 0、z = 0、lam = 0 的 token 不改变状态)。
    返回每块的量,第一维 M = N * 块数:u、z、块内累加的 log 衰减 G(FP64)、
    exp(G_t)、exp(G_C - G_t)、exp(G_C)。三个指数都不超过 1。"""
    k = u.shape[-1]
    pad = (-u.shape[1]) % size
    if pad:
        u, z, lam = (F.pad(t, (0, 0, 0, pad)) for t in (u, z, lam))
    ub = u.reshape(-1, size, k)
    zb = z.reshape(-1, size, z.shape[-1])
    g = lam.reshape(-1, size, k).double().cumsum(1)
    last = g[:, -1:]
    return ub, zb, g, g.to(u.dtype).exp(), (last - g).to(u.dtype).exp(), last.squeeze(1).to(u.dtype).exp()


def _heads_first(t):
    """[B, L, H, D] -> [B * H, L, D]"""
    b, length, h, d = t.shape
    return t.permute(0, 2, 1, 3).reshape(b * h, length, d)


def _fp32_region(fn):
    """让函数在关闭 autocast 的环境下运行:里面的矩阵乘和三角求解必须保持 FP32,不能被 autocast 降到 BF16。"""

    @functools.wraps(fn)
    def wrapper(first, *args, **kwargs):
        with torch.autocast(device_type=first.device.type, enabled=False):
            return fn(first, *args, **kwargs)

    return wrapper


@_fp32_region
def block_write_forward(state, key, write_value, lam, block):
    """分块 WY 写入。state [B, H, K, V];key、lam [B, L, H, K];write_value [B, L, H, V](FP32)。
    返回最终状态 [B, H, K, V],以及每块开始时的状态 [B * H, K, V](反向用的存档)。"""
    u, z, lm = _heads_first(key), _heads_first(write_value), _heads_first(lam)
    n, length, k = u.shape
    size = _block_size(length, block)
    ub, zb, g, head, tail, keep = _wy_terms(u, z, lm, size)
    count = ub.shape[0] // n
    # 下面四个量不依赖状态,所有块一起算
    lower = wy_lower(ub, g).view(n, count, size, size)
    scaled_u, w = (ub * head).view(n, count, size, k), (ub * tail).view(n, count, size, k)
    zb, keep = zb.view(n, count, size, -1), keep.view(n, count, k)
    s = state.reshape(n, *state.shape[2:])
    starts = []
    for i in range(count):  # 只有这一步依赖状态,必须按块依次做
        starts.append(s)
        e = torch.linalg.solve_triangular(lower[:, i], zb[:, i] - scaled_u[:, i] @ s, upper=False, unitriangular=True)
        s = s * keep[:, i, :, None] + w[:, i].transpose(1, 2) @ e
    return s.reshape(state.shape), starts


@_fp32_region
def block_write_backward(grad_state, starts, key, write_value, lam, block):
    """block_write_forward 的手写反向:返回 d state、d key、d write_value、d lam。
    中间量(L、E 等)从存档的块起始状态重算,全程不调用 autograd
    (MemT 报告 4.2 节:"reconstructing intermediate quantities during backward")。"""
    b, length, h, _ = key.shape
    u, z, lm = _heads_first(key), _heads_first(write_value), _heads_first(lam)
    n, k = u.shape[0], u.shape[-1]
    size = _block_size(length, block)
    ub, zb, g, head, tail, keep = _wy_terms(u, z, lm, size)  # 第一维 M = N * 块数
    count = ub.shape[0] // n
    # ---- 重算前向的中间量:各块互不依赖(起始状态有存档),所有块一起算 ----
    levels = _level_operands(ub, g)
    lower = wy_lower(ub, g, levels)
    scaled_u, w = ub * head, ub * tail
    start = torch.stack(starts, 1).reshape(n * count, *starts[0].shape[1:])  # [M, K, V]
    e = torch.linalg.solve_triangular(lower, zb - scaled_u @ start, upper=False, unitriangular=True)  # [M, C, V]
    # ---- 状态的梯度沿块往回传:只有这一步必须依次做 ----
    upper = lower.transpose(1, 2).reshape(n, count, size, size)
    w_blocks, scaled_blocks = w.view(n, count, size, k), scaled_u.view(n, count, size, k)
    keep_blocks = keep.view(n, count, k)
    grads = torch.empty_like(start).view(n, count, *start.shape[1:])  # 每块末状态的梯度 G'
    d_rhs = torch.empty_like(zb).view(n, count, size, -1)
    grad = grad_state.reshape(n, *grad_state.shape[2:])
    for i in reversed(range(count)):
        grads[:, i] = grad
        d_rhs[:, i] = torch.linalg.solve_triangular(upper[:, i], w_blocks[:, i] @ grad, upper=True, unitriangular=True)
        grad = grad * keep_blocks[:, i, :, None] - scaled_blocks[:, i].transpose(1, 2) @ d_rhs[:, i]
    # ---- 其余梯度:各块互不依赖,所有块一起算 ----
    grads, d_rhs = grads.view(n * count, *start.shape[1:]), d_rhs.view(n * count, size, -1)
    d_w = e @ grads.transpose(1, 2)
    d_scaled_u = -(d_rhs @ start.transpose(1, 2))
    plus, minus = wy_lower_backward(-(d_rhs @ e.transpose(1, 2)), ub, g, levels)
    d_u = plus + d_scaled_u * head + d_w * tail
    d_cum = ub * (minus + d_scaled_u * head - d_w * tail)  # dL/dG_t
    d_cum[:, -1] += (grads * start).sum(-1) * keep + (d_w * w).sum(1)  # rowsum(G' * S_C)
    d_lam = d_cum.flip(1).cumsum(1).flip(1)

    def back(t):  # [N * 块数, size, D] -> [B, L, H, D]
        return t.reshape(b, h, count * size, t.shape[-1])[:, :, :length].permute(0, 2, 1, 3)

    return grad.reshape(grad_state.shape), back(d_u), back(d_rhs), back(d_lam)


class BlockWriteFn(torch.autograd.Function):
    """只包含 block 写入(测试用;MemTMemoryFn 调用的是同一组函数)。"""

    @staticmethod
    def forward(ctx, state, key, write_value, lam, block):
        final, starts = block_write_forward(state, key, write_value, lam, block)
        ctx.block = block
        ctx.save_for_backward(key, write_value, lam, *starts)
        return final

    @staticmethod
    def backward(ctx, grad):
        key, write_value, lam, *starts = ctx.saved_tensors
        d_state, d_key, d_write, d_lam = block_write_backward(grad, starts, key, write_value, lam, ctx.block)
        return d_state, d_key, d_write, d_lam, None


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
        updated, checkpoints = block_write_forward(start, key, write_value, log_decay, cfg.block)
        return read, updated, checkpoints
    with torch.autocast(device_type=hidden.device.type, enabled=False):
        if with_checkpoints:
            updated, checkpoints = delta_rule_forward(start, key, write_value, decay, cfg.ckpt_every)
            return read, updated, checkpoints
        updated = start
        for index in range(length):
            updated = delta_rule_step(updated, key[:, index], write_value[:, index], decay[:, index])
    return read, updated, None


# in_proj 反向的两次大矩阵乘(d in_proj = d_rows^T x,d hidden = d_rows in_proj)用什么精度算。
# None:与前向做这次线性变换时相同(BF16 模型就是 BF16,与 autograd 相同;累加仍在 FP32 下进行),整层反向最快。
# torch.float32:在 FP32 下算,2048 个 token 的 Nano 层要多花约 10 ms。其余梯度始终在 FP32 下算。
PROJECTION_BACKWARD_DTYPE = None


def memory_backward(grad_read, grad_updated, hidden, params, snapshot, write_state, checkpoints, commit, cfg):
    """memory_forward 的梯度。grad_read / grad_updated 可以是 None(该输出没被用到)。
    返回 d_hidden、d_params(顺序同 params)、d_snapshot、d_write_state。"""
    in_w, f_w = params[0], params[1]
    dim, width = cfg.head_dim, cfg.num_heads * cfg.head_dim
    # 先重算前向里的线性变换。它们要在与前向相同的 autocast 状态下算,才能得到与前向相同的值。
    projected = F.linear(hidden, in_w[: 3 * width if commit else width])
    gate_in = F.linear(hidden, in_w[-dim:])
    gate = F.linear(gate_in, params[2], params[3])
    gates = F.linear(hidden, in_w[3 * width : -dim]) if commit else None  # b、w、f
    decay_in = F.linear(gates[..., 2 * width :], f_w) if commit else None
    # 其余运算都显式指定了精度:关闭 autocast,避免 FP32 的矩阵乘被自动降到 BF16。
    with torch.autocast(device_type=hidden.device.type, enabled=False):
        return _memory_backward(
            grad_read, grad_updated, hidden, params, snapshot, write_state, checkpoints, commit, cfg,
            projected, gate_in, gate, gates, decay_in,
        )


def _memory_backward(
    grad_read, grad_updated, hidden, params, snapshot, write_state, checkpoints, commit, cfg,
    projected, gate_in, gate, gates, decay_in,
):
    """memory_backward 的主体。最后五个参数是已经重算好的线性变换输出。"""
    in_w, f_w, g_w, g_b, decay_bias, decay_scale_log, norm_w = params
    batch, length, _ = hidden.shape
    heads, dim = cfg.num_heads, cfg.head_dim
    width = heads * dim
    shape = (batch, length, heads, dim)
    flat = (batch, length, width)
    snap = snapshot.float()
    gate = gate.reshape(shape)

    # ---- 重算前向的其余量(状态取自存档) ----
    q_raw = projected[..., :width].reshape(shape).float()
    q_rstd = torch.rsqrt(q_raw.square().sum(-1, keepdim=True) + _L2_EPS)
    q_hat = q_raw * q_rstd
    q_used = q_hat.to(projected.dtype).float() if commit else q_hat

    # ---- 读路径:read = o_norm(scale * q_hat S_snapshot, gate) ----
    if grad_read is not None:
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
        d_q = snap.new_zeros(flat)
        d_norm_w = d_g_w = d_g_b = None
        d_gate_in = snap.new_zeros(batch, length, dim)

    d_f_w = d_decay_bias = d_scale_log = None  # 只在写入时用到的参数:不写入时梯度为 None
    d_write_state = None
    # in_proj 的反向是整层最大的两次矩阵乘,精度见 PROJECTION_BACKWARD_DTYPE 的说明
    mm_dtype = PROJECTION_BACKWARD_DTYPE or projected.dtype
    x_2d = hidden.reshape(-1, hidden.shape[-1]).to(mm_dtype)
    if not commit:
        # 只用到了 in_proj 的 q 行和 g 行
        d_q_mm, d_gate_in_mm = d_q.to(mm_dtype), d_gate_in.to(mm_dtype)
        d_in_w = torch.zeros(in_w.shape, dtype=mm_dtype, device=in_w.device)
        d_in_w[:width] = d_q_mm.reshape(-1, width).t() @ x_2d
        d_in_w[-dim:] = d_gate_in_mm.reshape(-1, dim).t() @ x_2d
        d_hidden = d_q_mm @ in_w[:width].to(mm_dtype) + d_gate_in_mm @ in_w[-dim:].to(mm_dtype)
    else:
        # ---- 写路径 ----
        k_raw = projected[..., width : 2 * width].reshape(shape)
        value = projected[..., 2 * width :].reshape(shape)
        k_rstd = torch.rsqrt(k_raw.float().square().sum(-1, keepdim=True) + _L2_EPS)
        k_hat = k_raw.float() * k_rstd
        k_used = k_hat.to(k_raw.dtype).float()
        b, w, f = gates.split((width, width, dim), -1)
        b32 = b.float()
        sqrt_b = (0.5 * F.logsigmoid(b32)).exp()
        sqrt_b_used = sqrt_b.to(b.dtype).float().reshape(shape)
        u = (k_used * sqrt_b_used).to(value.dtype).float()
        sig_w = w.sigmoid().reshape(shape).float()
        write_value = sig_w * value.float()
        z = decay_in.float() + decay_bias.float()
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
                grad_updated.float(), checkpoints, u, write_value, log_decay, cfg.block
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
        d_rows = torch.cat([d_q, d_k, d_v, d_b, d_w, d_f, d_gate_in], -1).to(mm_dtype)
        d_in_w = d_rows.reshape(-1, d_rows.shape[-1]).t() @ x_2d
        d_hidden = d_rows @ in_w.to(mm_dtype)

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
