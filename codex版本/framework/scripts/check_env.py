"""MemT 式 4B 训练的环境检查(也用于规划 Megatron 版)。

    python scripts/check_env.py                 # 全部检查,约一分钟
    python scripts/check_env.py --no-network    # 跳过包索引的查询
    python scripts/check_env.py --allow-cpu     # 纯 CPU 任务:只检查镜像、依赖包和包索引

每项检查打印一行,最后打印一段 JSON 汇总;有检查失败时,请把完整输出发回来。
不安装、不修改任何东西。设置了 TORCHINDUCTOR_CACHE_DIR / TMPDIR 时,编译缓存写到那里。"""
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
    except Exception as error:  # noqa: BLE001 - 任何导入失败都要报告
        return f"missing ({type(error).__name__})"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-network", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true", help="do not fail when no GPU is visible")
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
    record("cuda available", cuda or args.allow_cpu,
           f"{torch.cuda.device_count()} device(s), CUDA_VISIBLE_DEVICES="
           f"{os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}"
           + ("" if cuda else " -- GPU checks skipped" if args.allow_cpu else ""))
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
        try:  # BF16 autocast + FP32 三角求解:记忆写入用到的运算
            with torch.autocast("cuda", dtype=torch.bfloat16):
                y = torch.randn(64, 64, device=device) @ torch.randn(64, 64, device=device)
            lower = torch.randn(8, 64, 64, device=device).tril(-1)
            torch.linalg.solve_triangular(lower, torch.randn(8, 64, 16, device=device), upper=False, unitriangular=True)
            record("bf16 autocast + triangular solve", y.dtype == torch.bfloat16, str(y.dtype))
        except Exception as error:  # noqa: BLE001
            record("bf16 autocast + triangular solve", False, error)
        try:  # 带块 mask 的 flex_attention 第一次调用要编译 kernel:需要可用的编译环境
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

    # Megatron 相关:仅供参考,train_aligned.sh 不需要
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
