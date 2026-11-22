"""Validate CUDA KDA states + BPTT gradients, then benchmark writes (not full-model throughput)."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

# Set before importing Triton/PyTorch. The production adapter requires this explicitly.
os.environ.setdefault("TRITON_F32_DEFAULT", "ieee")
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lact_qwen.kda_backend import kda_write
from lact_qwen.matrix_memory import delta_write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--length", type=int, default=2048)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    torch.manual_seed(42)
    h, t, d = args.heads, args.length, args.dim
    state = torch.randn(h, d, d, device="cuda") * .1
    u = F.normalize(torch.randn(h, t, d, device="cuda"), dim=-1) * torch.rand(h, t, d, device="cuda").sqrt()
    z = torch.randn(h, t, d, device="cuda") * .1
    lam = -torch.rand(h, t, d, device="cuda") * .03
    inputs = [v.requires_grad_() for v in (state, u, z, lam)]
    probe = torch.randn_like(state)

    def run(function):
        # Splitting into two calls exercises gradients through the state boundary.
        mid = t // 2
        out = function(state, u[:, :mid], z[:, :mid], lam[:, :mid])
        out = function(out, u[:, mid:], z[:, mid:], lam[:, mid:])
        grad = torch.autograd.grad((out * probe).sum(), inputs)
        return out, grad

    print("Checking reference PyTorch recurrence", flush=True)
    ref, ref_grad = run(delta_write)
    print("Checking FLA KDA recurrence (first call may compile kernels)", flush=True)
    actual, actual_grad = run(kda_write)
    def error(a, b):
        diff = (a - b).float()
        return {"max_abs": diff.abs().max().item(),
                "relative_rms": (diff.square().mean().sqrt() / b.float().square().mean().sqrt().clamp_min(1e-12)).item()}
    errors = {"state": error(actual, ref), **{
        name: error(a, b) for name, a, b in zip(("initial_state_grad", "u_grad", "z_grad", "decay_grad"), actual_grad, ref_grad)}}
    print(json.dumps({"device": torch.cuda.get_device_name(), "torch": torch.__version__,
                       "shape": vars(args), "errors": errors}), flush=True)
    for name, item in errors.items():
        if item["relative_rms"] > 1e-3 or not torch.isfinite(torch.tensor(item["max_abs"])):
            raise AssertionError(f"KDA equivalence failed for {name}: {item}")
    timing = {}
    for name, function in (("torch", delta_write), ("fla_kda", kda_write)):
        run(function)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        baseline = torch.cuda.memory_allocated()
        begin = time.perf_counter()
        for _ in range(args.repeats):
            run(function)
        torch.cuda.synchronize()
        timing[name] = {"forward_backward_ms": (time.perf_counter() - begin) * 1000 / args.repeats,
                         "incremental_peak_mib": (torch.cuda.max_memory_allocated() - baseline) / 2**20}
    print(json.dumps({"timing": timing}), flush=True)


if __name__ == "__main__":
    main()
