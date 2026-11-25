"""一个记忆层(读 + 写)在 GPU 上的前向、反向耗时和显存峰值。只依赖 torch 和 memt_autograd.py,不需要 transformers。

    python tests/bench_memory_layer.py                                  # Nano 尺寸:hidden 2048、16 头 x 256、2048 个 token
    python tests/bench_memory_layer.py --hidden 2560 --heads 32 --dim 128   # 4B 框架的尺寸

比较三种做法(都是手写反向,模型权重和输入是 BF16,状态是 FP32):
    block          分块 WY 写入(默认)
    block, FP32 in_proj backward   同上,但 in_proj 的反向在 FP32 下算(PROJECTION_BACKWARD_DTYPE = float32)
    token          官方的逐 token 循环 + 逐 token 反向
可选 --old-stable:再测旧的"块内两两相乘"稳定公式的写入(前向 + autograd 反向),只算写入这一步。"""

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from memt_nano import memt_autograd as ma  # noqa: E402


def timed(fn, repeat):
    fn()  # 预热
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    best, out = float("inf"), None
    for _ in range(repeat):  # 取最小值:同一张卡上有别的任务时也比较稳
        torch.cuda.synchronize()
        start = time.time()
        out = fn()
        torch.cuda.synchronize()
        best = min(best, time.time() - start)
    return out, best * 1e3, (torch.cuda.max_memory_allocated() - base) / 2**30


def pairwise_stable_write(state, u, z, lam, block=64):
    """旧的稳定公式(只作速度参照):每 64 个 token 一块,块内直接算 exp(G_t - G_s) 的 [C, C, K] 张量。
    state [N, K, V];u、lam [N, L, K];z [N, L, V]。"""
    for start in range(0, u.shape[1], block):
        ub, zb = u[:, start : start + block], z[:, start : start + block]
        g = lam[:, start : start + block].cumsum(dim=1)
        c = ub.shape[1]
        below = torch.ones(c, c, dtype=torch.bool, device=u.device).tril(-1)
        pair = (g[:, :, None, :] - g[:, None, :, :]).masked_fill(~below[None, :, :, None], float("-inf"))
        lower = torch.einsum("ntk,nsk,ntsk->nts", ub, ub, pair.exp())
        e = torch.linalg.solve_triangular(lower, zb - (ub * g.exp()) @ state, upper=False, unitriangular=True)
        last = g[:, -1:, :]
        state = state * last.transpose(1, 2).exp() + (ub * (last - g).exp()).transpose(1, 2) @ e
    return state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--length", type=int, default=2048)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--no-token", action="store_true", help="skip the per-token writer (slow)")
    parser.add_argument("--old-stable", action="store_true", help="also time the old pairwise stable write formula")
    args = parser.parse_args()
    device = torch.device("cuda")
    hidden, heads, dim, length = args.hidden, args.heads, args.dim, args.length
    width = heads * dim
    torch.manual_seed(0)
    bf16 = torch.bfloat16

    def weight(rows, cols, scale):
        return (torch.randn(rows, cols, device=device) * scale).to(bf16)

    params = (
        weight(5 * width + 2 * dim, hidden, hidden**-0.5),  # in_proj:q、k、v、b、w、f、g
        weight(width, dim, dim**-0.5),  # f_down
        weight(width, dim, dim**-0.5),  # g_down
        None,  # g_down.bias
        torch.randn(width, device=device).to(bf16),  # decay_bias
        torch.zeros(heads, device=device).to(bf16),  # decay_scale_log
        torch.ones(dim, device=device).to(bf16),  # o_norm.weight
    )
    x = torch.randn(1, length, hidden, device=device).to(bf16)
    state = torch.randn(1, heads, dim, dim, device=device) * 0.02
    g_read = torch.randn(1, length, width, device=device).to(bf16)
    g_updated = torch.randn(1, heads, dim, dim, device=device)
    print(f"{torch.cuda.get_device_name(0)}, torch {torch.__version__}")
    print(f"memory layer: hidden {hidden}, {heads} heads x {dim}, {length} tokens, BF16 weights, FP32 state")

    def config(writer):
        return ma.MemoryConfig(num_heads=heads, head_dim=dim, read_scale=dim**-0.5, norm_eps=1e-6, softcap=True,
                               limit=2.0, ckpt_every=256, writer=writer, block=256)

    results = {}
    with torch.no_grad():
        for label, writer, proj in (("block", "block", None), ("block, FP32 in_proj backward", "block", torch.float32),
                                    ("token", "token", None)):
            if writer == "token" and args.no_token:
                continue
            cfg = config(writer)
            repeat = 2 if writer == "token" else args.repeat
            ma.PROJECTION_BACKWARD_DTYPE = proj
            (read, updated, saved), t_fwd, m_fwd = timed(
                lambda: ma.memory_forward(x, params, state, None, True, cfg, with_checkpoints=True), repeat)
            grads, t_bwd, m_bwd = timed(
                lambda: ma.memory_backward(g_read, g_updated, x, params, state, None, saved, True, cfg), repeat)
            ma.PROJECTION_BACKWARD_DTYPE = None
            results[label] = (updated, grads)
            print(f"   {label:30s} forward {t_fwd:7.1f} ms   backward {t_bwd:7.1f} ms   total {t_fwd + t_bwd:7.1f} ms   "
                  f"peak extra {max(m_fwd, m_bwd):5.2f} GiB")
        if "token" in results:
            a, b = results["block"][0].double(), results["token"][0].double()
            print(f"   written state, block vs token loop: rel diff {((a - b).norm() / b.norm()).item():.1e}")
        if args.old_stable:
            # 只算写入:新的分块 WY 与旧的两两公式,输入用同一组 u、z、lam
            u = torch.randn(heads, length, dim, device=device)
            u = torch.rand(heads, length, 1, device=device).sqrt() * u / u.norm(dim=-1, keepdim=True)
            z = torch.randn(heads, length, dim, device=device)
            lam = -torch.rand(heads, length, dim, device=device) * 0.5
            s0, gout = state[0], g_updated[0]
            key4, z4, lam4 = (t.reshape(1, heads, length, dim).permute(0, 2, 1, 3) for t in (u, z, lam))
            (_, starts), t_fwd, _ = timed(lambda: ma.block_write_forward(s0[None], key4, z4, lam4, 256), args.repeat)
            _, t_bwd, m_new = timed(lambda: ma.block_write_backward(gout[None], starts, key4, z4, lam4, 256), args.repeat)
            _, t_old_fwd, _ = timed(lambda: pairwise_stable_write(s0, u, z, lam), 3)
    if args.old_stable:
        def old_both():
            inputs = [t.detach().requires_grad_() for t in (s0, u, z, lam)]
            return torch.autograd.grad(pairwise_stable_write(*inputs), inputs, gout)

        _, t_old_both, m_old = timed(old_both, 3)
        print(f"   write only, new chunked WY     : forward {t_fwd:6.1f} ms   backward {t_bwd:6.1f} ms   peak {m_new:.2f} GiB")
        print(f"   write only, old pairwise stable: forward {t_old_fwd:6.1f} ms   backward {t_old_both - t_old_fwd:6.1f} ms"
              f" (autograd)   peak {m_old:.2f} GiB")


if __name__ == "__main__":
    main()
