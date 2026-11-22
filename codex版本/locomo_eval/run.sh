#!/usr/bin/bash
# h20-1: LoCoMo gate test for the 4B MemT checkpoint step_2160 (stopped run, CE on LDC long text only).
# usage: bash run.sh smoke   -> first 2 rows, one GPU per mode
#        bash run.sh full    -> all 387 rows, each mode sharded over the GPUs in $GPUS
source /data/qyl/lact-insight-env/env.sh >/dev/null 2>&1
W=/data/qyl/memt_locomo_20261122; cd $W; export HF_HUB_OFFLINE=1 TMPDIR=/data/qyl/tmp PYTHONDONTWRITEBYTECODE=1
CK=/mnt/airs-business/memory/memt4b_20261120/step_2160
BASE="--model /data/models/Qwen/Qwen3-4B --data data/locomo2_g4.jsonl --attn-impl sdpa"
MEM="--memory $CK --window 2048 --sinks 0 --attn-segment 2048"
NOMEM="--memory none --window 0"
declare -A ARGS=( [A_full]="--mode full $NOMEM" [B_last]="--mode last $NOMEM" [D_nowrite]="--mode nowrite $MEM"
                  [C_write]="--mode write $MEM" [F_evidence]="--mode evidence $MEM" [G_random]="--mode random $MEM" [S_qk]="--mode qk $MEM" [W_wrong]="--mode wrong $MEM" )
G=(${GPUS:-0 1 2 3 4 5})
if [ "$1" = smoke ]; then
  mkdir -p out/smoke; i=0
  for n in A_full B_last D_nowrite C_write F_evidence G_random; do
    CUDA_VISIBLE_DEVICES=${G[$i]} python eval_locomo_memt.py $BASE ${ARGS[$n]} --limit 2 --out out/smoke/$n.s0.json > logs/smoke_$n.log 2>&1 &
    i=$((i+1))
  done; wait
  python summary_memt.py out/smoke; echo SMOKE_DONE
else
  mkdir -p out/full
  for n in B_last C_write W_wrong F_evidence G_random S_qk A_full; do
    ns=${#G[@]}
    for s in $(seq 0 $((ns-1))); do
      [ -f out/full/$n.s$s.json ] && continue
      CUDA_VISIBLE_DEVICES=${G[$s]} python eval_locomo_memt.py $BASE ${ARGS[$n]} --shard $s --nshards $ns --out out/full/$n.s$s.json > logs/full_${n}_s$s.log 2>&1 &
    done; wait
    echo "== $n $(grep -hE 'Traceback|Error' logs/full_${n}_s*.log | tail -1 | cut -c1-150)"
  done
  python summary_memt.py out/full; echo FULL_DONE
fi
