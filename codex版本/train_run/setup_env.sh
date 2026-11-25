#!/usr/bin/env bash
# 给 train_aligned.sh 准备环境,用在已经带 CUDA 版 PyTorch 的平台镜像上
#(目标镜像:Python 3.12、torch 2.12.1+cu129;已在系统 torch 为 2.11.0+cu130 的机器上实测)。
# 创建一个复用镜像自带 torch 的 venv,只安装额外的几个包,镜像自带的 transformers / vLLM 不受影响。
#
#   bash setup_env.sh                      # venv 建在 train_run 的上一级目录:venv_memt
#   VENV=/path/venv PIP_INDEX_URL=https://... bash setup_env.sh
#   source "$VENV/bin/activate"; export PYTHON=python
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BASE_PYTHON="${BASE_PYTHON:-python3}"
VENV="${VENV:-$SCRIPT_DIR/../venv_memt}"
"$BASE_PYTHON" - <<'PY'
import sys
import torch
major, minor = (int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
print(f"base python {sys.version.split()[0]}, torch {torch.__version__}, cuda build {torch.version.cuda}")
if (major, minor) < (2, 11) or (major, minor) >= (2, 15):
    sys.exit("torch must be >= 2.11 and < 2.15 for this code (flex_attention, tested range)")
PY
"$BASE_PYTHON" -m venv --system-site-packages "$VENV"
"$VENV/bin/python" -m pip install --upgrade pip
# 这里故意不装 torch:它通过 --system-site-packages 来自镜像
"$VENV/bin/python" -m pip install "transformers==5.20.2" "datasets==5.1.0" "einops==0.8.2" "pytest>=8,<10"
"$VENV/bin/python" - <<'PY'
import datasets, einops, torch, transformers
print("venv ok: torch", torch.__version__, "| transformers", transformers.__version__,
      "| datasets", datasets.__version__, "| einops", einops.__version__)
if not torch.cuda.is_available():
    print("note: no GPU visible in this shell (fine for a CPU-only check)")
PY
echo "next: source $VENV/bin/activate && python $SCRIPT_DIR/../framework/scripts/check_env.py"
