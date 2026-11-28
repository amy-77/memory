#!/bin/bash
# h20-1 上跑 oracle 诊断。用法：bash run_h20_1.sh prepare | pilot | run | decay | summarize
# /data 已满：输出写到 179 的挂载上，缓存和临时文件放 /dev/shm。
set -euo pipefail
W=/data/tmp1124/store_179/livemem_oracle_diag_20261128
CODE=$W/code
L=/data/tmp1124/store_179/livemem_repro
REPO=$L/LiveMem
CKPT=$L/models/LiveMem-SFT
DATA=/data/tmp1124/memt_bench_20261124/data/longmemeval-cleaned/longmemeval_s_cleaned.json
S=/dev/shm/livemem_oracle_diag_1128
mkdir -p $S/tmp $S/triton $S/xdg $S/inductor $S/hf $W/logs
export TMPDIR=$S/tmp TRITON_CACHE_DIR=$S/triton XDG_CACHE_HOME=$S/xdg TORCHINDUCTOR_CACHE_DIR=$S/inductor
export HF_HOME=$S/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
source /data/tmp1124/livemem_env/venv/bin/activate
export PYTHONPATH=$REPO
# smi 0、1 两张卡，按 UUID 指定（h20-1 上 smi 序号和 CUDA 序号可能不一致）
GPU0=GPU-5ed09c40-99b9-ce1f-af3e-b9e3622db86e
GPU1=GPU-73953b94-52b1-e129-29ac-9b1f9f0be497
COMMON="--run $W --repo $REPO --checkpoint $CKPT --data $DATA"

case "$1" in
  prepare)
    python $CODE/h200_experiment1.py prepare $COMMON 2>&1 | tee $W/logs/prepare.log
    python $CODE/oracle_diag.py turns $COMMON 2>&1 | tee $W/logs/turns.log ;;
  pilot)   # 每张卡 1 题，量时间、跑检查
    CUDA_VISIBLE_DEVICES=$GPU0 python $CODE/oracle_diag.py run $COMMON --limit 2 --rank 0 --world 2 > $W/logs/pilot_0.log 2>&1 &
    CUDA_VISIBLE_DEVICES=$GPU1 python $CODE/oracle_diag.py run $COMMON --limit 2 --rank 1 --world 2 > $W/logs/pilot_1.log 2>&1 &
    wait ;;
  run)     # 已有结果会跳过，可以反复启动
    CUDA_VISIBLE_DEVICES=$GPU0 python $CODE/oracle_diag.py run $COMMON --rank 0 --world 2 >> $W/logs/run_0.log 2>&1 &
    CUDA_VISIBLE_DEVICES=$GPU1 python $CODE/oracle_diag.py run $COMMON --rank 1 --world 2 >> $W/logs/run_1.log 2>&1 &
    wait ;;
  decay)
    CUDA_VISIBLE_DEVICES=${2:-$GPU0} python $CODE/oracle_diag.py decay $COMMON --limit ${3:-24} >> $W/logs/decay.log 2>&1 ;;
  full)    # 卡 0：先测 16 题的衰减；两张卡先跑第一批（最关键的 5 组），再跑第二批
    P1=write_all,turn_oracle,turn_random,turn_oracle_nodecay,turn_random_nodecay
    P2=write_nothing,session_oracle,memory_off,full_attention
    ( export CUDA_VISIBLE_DEVICES=$GPU0
      python $CODE/oracle_diag.py decay $COMMON --limit 16 >> $W/logs/decay.log 2>&1
      python $CODE/oracle_diag.py run $COMMON --rank 0 --world 2 --arms $P1 >> $W/logs/run_0.log 2>&1
      python $CODE/oracle_diag.py run $COMMON --rank 0 --world 2 --arms $P2 >> $W/logs/run_0.log 2>&1 ) &
    ( export CUDA_VISIBLE_DEVICES=$GPU1
      python $CODE/oracle_diag.py run $COMMON --rank 1 --world 2 --arms $P1 >> $W/logs/run_1.log 2>&1
      python $CODE/oracle_diag.py run $COMMON --rank 1 --world 2 --arms $P2 >> $W/logs/run_1.log 2>&1 ) &
    wait ;;
  summarize)
    python $CODE/oracle_diag.py summarize --run $W ;;
esac
