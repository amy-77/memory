"""矩阵记忆几种写入算法在 GPU 上的速度和显存:delta_write(1 / P_i 技巧,需要衰减上限)、
delta_write_stable(任意衰减)和 delta_write_auto,测一个存储步的前向 + 反向,并比较结果是否一致。

    python scripts/bench_matrix_write.py --heads 32 --dim 128 --length 2048     # Qwen3-4B 的一层(batch 1)"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lact_qwen.matrix_memory import MAX_DECAY, delta_write, delta_write_auto, delta_write_stable  # noqa: E402


def run(fn, state, u, z, lam, probe):
    args = [t.clone().requires_grad_() for t in (state, u, z, lam)]
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    start = time.time()
    out = fn(*args)
    (out * probe).sum().backward()
    torch.cuda.synchronize()
    return out.detach(), [a.grad for a in args], time.time() - start, (torch.cuda.max_memory_allocated() - base) / 2**30


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--heads", type=int, default=32)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--length", type=int, default=2048)
    p.add_argument("--repeat", type=int, default=3)
    a = p.parse_args()
    torch.manual_seed(0)
    n, d, length = a.heads, a.dim, a.length
    k = torch.nn.functional.normalize(torch.randn(n, length, d, device="cuda"), dim=-1)
    u = torch.rand(n, length, d, device="cuda").sqrt() * k
    z = torch.randn(n, length, d, device="cuda")
    lam = -torch.rand(n, length, d, device="cuda") * MAX_DECAY  # 在上限以内,所以各种写法都适用
    state = torch.randn(n, d, d, device="cuda") * 0.1
    probe = torch.randn(n, d, d, device="cuda")
    writers = {"scaled (block 256)": lambda *t: delta_write(*t, 256),
               "stable (block 64, group 256)": lambda *t: delta_write_stable(*t, block=64, group=256),
               "auto (small decays)": lambda *t: delta_write_auto(*t)}
    results = {}
    for name, fn in writers.items():
        run(fn, state[:, :, :], u[:, :64], z[:, :64], lam[:, :64], probe)  # 预热
        times = []
        for _ in range(a.repeat):
            out, grads, sec, peak = run(fn, state, u, z, lam, probe)
            times.append(sec)
        results[name] = (out, grads)
        print(f"{name:30s} fwd+bwd {min(times) * 1000:8.1f} ms   peak extra {peak:6.2f} GiB")
    rel = lambda x, y: ((x - y).norm() / y.norm()).item()
    o1, g1 = results["scaled (block 256)"]
    for name in list(results)[1:]:
        o2, g2 = results[name]
        print(f"{name} vs scaled: state rel diff {rel(o2, o1):.2e}; grad rel diff state/u/z/lam: "
              + " ".join(f"{rel(x, y):.2e}" for x, y in zip(g2, g1)))
    big = lam.clone()
    big[:, 300:340, 5] = -2.0  # 一个通道在某个 256-token 的块里遗忘得很快:这个块会走稳定公式
    for name, fn in (("auto (one large-decay block)", lambda *t: delta_write_auto(*t)),
                     ("stable, same input", lambda *t: delta_write_stable(*t))):
        run(fn, state, u[:, :64], z[:, :64], big[:, :64], probe)
        sec = min(run(fn, state, u, z, big, probe)[2] for _ in range(a.repeat))
        print(f"{name:30s} fwd+bwd {sec * 1000:8.1f} ms")


if __name__ == "__main__":
    main()
