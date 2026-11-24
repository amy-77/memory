#!/usr/bin/env bash
# Environment for train_aligned.sh on a platform image that already has a CUDA build of PyTorch
# (tested layout: Python 3.12, torch 2.12.1+cu129). Creates a venv that reuses the image's torch and
# installs only the extra packages, so the image's own transformers / vLLM stay untouched.
#
#   bash setup_env.sh                      # venv at ./venv_memt next to this repository copy
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
# torch is deliberately absent: it comes from the image through --system-site-packages
"$VENV/bin/python" -m pip install "transformers==5.20.2" "datasets==5.1.0" "einops==0.8.2" "pytest>=8,<10"
"$VENV/bin/python" - <<'PY'
import datasets, einops, torch, transformers
print("venv ok: torch", torch.__version__, "| transformers", transformers.__version__,
      "| datasets", datasets.__version__, "| einops", einops.__version__)
assert torch.cuda.is_available() or print("WARNING: CUDA not available in this shell")
PY
echo "next: source $VENV/bin/activate && python $SCRIPT_DIR/../framework/scripts/check_env.py"
