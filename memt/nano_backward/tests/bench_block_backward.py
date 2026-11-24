"""分块 WY 写入算子在 GPU 上的速度、显存和精度(FP32,单卡,只算写入这一步)。

    python tests/bench_block_backward.py                              # Nano 尺寸:16 头 x 256,2048 个 token
    python tests/bench_block_backward.py --heads 32 --dim 128         # 4B 框架的尺寸

每种衰减强度下比较:手写反向、"对同一个前向调 autograd"、官方的逐 token 循环(前向 + 手写逐 token 反向)。
误差都是相对 FP64 真值(同一算法的 FP64 版本,并用 FP64 逐 token 循环核对)。"""

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from memt_nano import memt_autograd as ma  # noqa: E402


def rel(a, b):
    a, b = a.double(), b.double()
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()


def timed(fn, repeat):
    fn()  # 预热
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    start = time.time()
    for _ in range(repeat):
        out = fn()
    torch.cuda.synchronize()
    return out, (time.time() - start) / repeat * 1e3, (torch.cuda.max_memory_allocated() - base) / 2**30


def autograd_backward(grad_out, state, key, write, lam, block):
    """参照:对同一个分块前向调 autograd。"""
    with torch.enable_grad():
        inputs = [t.detach().requires_grad_() for t in (state, key, write, lam)]
        return torch.autograd.grad(ma.block_write_forward(*inputs, block)[0], inputs, grad_out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--length", type=int, default=2048)
    parser.add_argument("--blocks", type=int, nargs="+", default=[64, 256])
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--no-token", action="store_true", help="skip the per-token loop (slow)")
    args = parser.parse_args()
    device = torch.device("cuda")
    torch.manual_seed(0)
    h, d, length = args.heads, args.dim, args.length
    state = torch.randn(1, h, d, d, device=device) * 0.02
    key = torch.randn(1, length, h, d, device=device)
    key = torch.rand(1, length, h, 1, device=device).sqrt() * key / key.norm(dim=-1, keepdim=True)
    write = torch.randn(1, length, h, d, device=device)
    grad_out = torch.randn(1, h, d, d, device=device)
    print(f"{h} heads x {d}, {length} tokens, FP32")
    for scale in (0.02, 0.5, 5.0):
        lam = -torch.rand(1, length, h, d, device=device) * scale
        d64 = [t.double() for t in (state, key, write, lam)]
        with torch.no_grad():
            out64, starts64 = ma.block_write_forward(*d64, 64)
            truth = ma.block_write_backward(grad_out.double(), starts64, *d64[1:], 64)
            loop = d64[0]
            for t in range(length):
                loop = ma.delta_rule_step(loop, d64[1][:, t], d64[2][:, t], d64[3][:, t].exp())
        print(f"== per-token log decay up to {scale}   (FP64 block writer vs FP64 token loop: {rel(out64, loop):.1e})")
        for block in args.blocks:
            with torch.no_grad():
                (out, starts), t_fwd, _ = timed(lambda: ma.block_write_forward(state, key, write, lam, block), args.repeat)
                hand, t_hand, m_hand = timed(
                    lambda: ma.block_write_backward(grad_out, starts, key, write, lam, block), args.repeat)
            auto, t_auto, m_auto = timed(lambda: autograd_backward(grad_out, state, key, write, lam, block), args.repeat)
            print(f"   block {block:3d}: forward {t_fwd:5.1f} ms, state err {rel(out, out64):.1e}")
            print(f"      backward hand-written  {t_hand:6.1f} ms, peak extra {m_hand:5.2f} GiB, grad err "
                  + " ".join(f"{rel(a, b):.1e}" for a, b in zip(hand, truth)))
            print(f"      forward+backward via autograd {t_auto:6.1f} ms, peak extra {m_auto:5.2f} GiB, grad err "
                  + " ".join(f"{rel(a, b):.1e}" for a, b in zip(auto, truth)))
        if not args.no_token:
            with torch.no_grad():
                decay = lam.exp()
                (out, cks), t_fwd, _ = timed(lambda: ma.delta_rule_forward(state, key, write, decay, 256), 1)
                tok, t_tok, m_tok = timed(lambda: ma.delta_rule_backward(grad_out, cks, key, write, decay, 256), 1)
            tok = (tok[0], tok[1], tok[2], tok[3] * decay)  # d decay -> d log decay
            print(f"   token loop: forward {t_fwd:5.1f} ms, state err {rel(out, out64):.1e}; backward {t_tok:6.1f} ms, "
                  f"peak extra {m_tok:5.2f} GiB, grad err " + " ".join(f"{rel(a, b):.1e}" for a, b in zip(tok, truth)))


if __name__ == "__main__":
    main()
