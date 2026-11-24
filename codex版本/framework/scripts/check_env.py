"""Environment check for the MemT-style 4B training (and for planning a Megatron port).

    python scripts/check_env.py                 # everything, about a minute
    python scripts/check_env.py --no-network    # skip the package-index queries

Prints one line per check and a JSON summary at the end; send the whole output back when something fails.
Nothing is installed or changed. Compile caches go to TORCHINDUCTOR_CACHE_DIR / TMPDIR if set.
"""
import argparse
import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time

REPORT = {}


def record(name, ok, detail=""):
    REPORT[name] = {"ok": bool(ok), "detail": str(detail)[:400]}
    print(f"{'ok  ' if ok else 'FAIL'} {name}: {detail}", flush=True)


def version_of(module):
    try:
        mod = importlib.import_module(module)
        return getattr(mod, "__version__", "installed")
    except Exception as error:  # noqa: BLE001 - report any import failure
        return f"missing ({type(error).__name__})"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-network", action="store_true")
    args = parser.parse_args()

    record("python", sys.version_info >= (3, 10), f"{sys.version.split()[0]} on {platform.platform()}")
    import torch

    major, minor = (int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
    record("torch", (2, 11) <= (major, minor) < (2, 15),
           f"{torch.__version__}, cuda build {torch.version.cuda}, cudnn {torch.backends.cudnn.version()}")
    for name, wanted in (("transformers", "5.20.2"), ("datasets", "5.1.0"), ("einops", None)):
        found = version_of(name)
        record(name, "missing" not in found and (wanted is None or found == wanted), f"{found} (tested with {wanted})")

    cuda = torch.cuda.is_available()
    record("cuda available", cuda, f"{torch.cuda.device_count()} device(s), CUDA_VISIBLE_DEVICES="
           f"{os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
    if cuda:
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            free, total = torch.cuda.mem_get_info(index)
            record(f"gpu {index}", total >= 60 * 2**30,
                   f"{props.name}, {total / 2**30:.0f} GiB total, {free / 2**30:.0f} GiB free, "
                   f"sm_{props.major}{props.minor} (one 32K sequence needs about 55 GiB)")
        try:
            record("nccl", True, ".".join(str(x) for x in torch.cuda.nccl.version()))
        except Exception as error:  # noqa: BLE001
            record("nccl", False, error)

    for key in ("TMPDIR", "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR"):
        path = os.environ.get(key) or (tempfile.gettempdir() if key == "TMPDIR" else None)
        if path is None:
            record(f"cache {key}", True, "unset (defaults under the temp dir / home)")
            continue
        try:
            os.makedirs(path, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=path):
                pass
            record(f"cache {key}", True, f"{path}, {shutil.disk_usage(path).free / 2**30:.0f} GiB free")
        except OSError as error:
            record(f"cache {key}", False, f"{path}: {error} (set it to a writable directory with free space)")

    if cuda:
        device = torch.device("cuda")
        try:  # BF16 autocast + FP32 triangular solve: what the memory write uses
            with torch.autocast("cuda", dtype=torch.bfloat16):
                y = torch.randn(64, 64, device=device) @ torch.randn(64, 64, device=device)
            lower = torch.randn(8, 64, 64, device=device).tril(-1)
            torch.linalg.solve_triangular(lower, torch.randn(8, 64, 16, device=device), upper=False, unitriangular=True)
            record("bf16 autocast + triangular solve", y.dtype == torch.bfloat16, str(y.dtype))
        except Exception as error:  # noqa: BLE001
            record("bf16 autocast + triangular solve", False, error)
        try:  # flex_attention with a block mask compiles kernels the first time: needs a working compiler stack
            from torch.nn.attention.flex_attention import create_block_mask, flex_attention

            def same_segment(batch, head, q_idx, kv_idx):
                return (q_idx // 2048 == kv_idx // 2048) & (kv_idx <= q_idx)

            start = time.time()
            mask = create_block_mask(same_segment, None, None, 4096, 4096, device=device)
            q = torch.randn(1, 4, 4096, 64, device=device, dtype=torch.bfloat16)
            out = torch.compile(flex_attention)(q, q, q, block_mask=mask)
            torch.cuda.synchronize()
            record("flex_attention (segment mask, compiled)", torch.isfinite(out.float()).all(),
                   f"{time.time() - start:.1f} s for the first call")
        except Exception as error:  # noqa: BLE001
            record("flex_attention (segment mask, compiled)", False, f"{type(error).__name__}: {error}")
        try:
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from lact_qwen.matrix_memory import delta_write_auto, delta_write_reference

            torch.manual_seed(0)
            u = torch.nn.functional.normalize(torch.randn(4, 300, 32, device=device), dim=-1) * 0.8
            z, lam = torch.randn(4, 300, 32, device=device), -torch.rand(4, 300, 32, device=device) * 0.05
            state = torch.randn(4, 32, 32, device=device)
            diff = (delta_write_auto(state, u, z, lam) - delta_write_reference(state, u, z, lam)).abs().max().item()
            record("matrix write on GPU vs token loop", diff < 1e-3, f"max abs diff {diff:.1e}")
        except Exception as error:  # noqa: BLE001
            record("matrix write on GPU vs token loop", False, f"{type(error).__name__}: {error}")

    # Megatron stack: informational, not needed for train_aligned.sh
    for name in ("megatron.core", "megatron.bridge", "transformer_engine", "flash_attn", "apex", "triton", "fla"):
        found = version_of(name)
        REPORT[f"optional {name}"] = {"ok": True, "detail": found}
        print(f"info optional {name}: {found}", flush=True)
    for tool in ("gcc", "g++", "cmake", "ninja", "nvcc", "mpirun", "git"):
        path = shutil.which(tool)
        REPORT[f"tool {tool}"] = {"ok": True, "detail": path or "not found"}
        print(f"info tool {tool}: {path or 'not found'}", flush=True)
    if not args.no_network:
        for package in ("megatron-core", "transformer-engine", "flash-attn"):
            try:
                result = subprocess.run([sys.executable, "-m", "pip", "index", "versions", package],
                                        capture_output=True, text=True, timeout=60)
                line = (result.stdout.strip().splitlines() or result.stderr.strip().splitlines() or ["no output"])[0]
            except Exception as error:  # noqa: BLE001
                line = f"{type(error).__name__}: {error}"
            REPORT[f"index {package}"] = {"ok": True, "detail": line[:300]}
            print(f"info package index {package}: {line[:200]}", flush=True)

    failed = [name for name, item in REPORT.items() if not item["ok"]]
    print("\nSUMMARY", "all required checks passed" if not failed else f"FAILED: {failed}")
    print("JSON " + json.dumps(REPORT, ensure_ascii=False))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
