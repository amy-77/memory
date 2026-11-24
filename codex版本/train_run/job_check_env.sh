#!/usr/bin/env bash
# 给提交式任务用的一条非交互命令:建好 venv(setup_env.sh),再运行 check_env.py。
# 所有输出都打到标准输出;即使有检查失败,任务也以 0 退出,保证日志能拿到。
#
#   bash train_run/job_check_env.sh                 # GPU 任务
#   bash train_run/job_check_env.sh --allow-cpu     # 纯 CPU 任务:只检查镜像、依赖包和包索引
#
# 环境变量:VENV(默认:train_run 上一级目录里的 venv_memt)、PIP_INDEX_URL、
# WORK_DIR(可写目录,放编译缓存和 pip 缓存;默认:venv 所在的目录)。
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
