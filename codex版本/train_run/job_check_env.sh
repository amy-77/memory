#!/usr/bin/env bash
# One non-interactive command for a submitted job: build the venv (setup_env.sh) and run check_env.py.
# Everything goes to stdout; the job exits 0 even when a check fails, so the log is always available.
#
#   bash train_run/job_check_env.sh                 # GPU job
#   bash train_run/job_check_env.sh --allow-cpu     # CPU-only job: image, packages and index access
#
# Environment: VENV (default: venv_memt next to train_run), PIP_INDEX_URL, WORK_DIR (writable directory for
# compile caches and pip's cache; default: the directory holding the venv).
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export VENV="${VENV:-$SCRIPT_DIR/../venv_memt}"
WORK_DIR="${WORK_DIR:-$(dirname "$VENV")}"
export TMPDIR="${TMPDIR:-$WORK_DIR/.cache_memt/tmp}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$WORK_DIR/.cache_memt/inductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$WORK_DIR/.cache_memt/triton}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$WORK_DIR/.cache_memt/pip}"
mkdir -p "$TMPDIR" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$PIP_CACHE_DIR"
echo "===== job_check_env $(date '+%F %T') host $(hostname)"
echo "===== image"; (cat /etc/os-release 2>/dev/null | head -2; python3 -V; nvidia-smi -L 2>&1 | head -20; df -h "$WORK_DIR" / /tmp 2>/dev/null)
echo "===== setup_env.sh"
if [[ -x "$VENV/bin/python" ]] && "$VENV/bin/python" -c "import transformers, datasets, einops" 2>/dev/null; then
  echo "venv already present: $VENV"
else
  bash "$SCRIPT_DIR/setup_env.sh"; echo "setup_env exit code $?"
fi
echo "===== check_env.py"
PY="$VENV/bin/python"; [[ -x "$PY" ]] || PY=python3
"$PY" "$SCRIPT_DIR/../framework/scripts/check_env.py" "$@"; echo "check_env exit code $?"
echo "===== job_check_env done $(date '+%F %T')"
exit 0
