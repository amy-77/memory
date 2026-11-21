#!/usr/bin/bash
# Selective-write evaluation (no training). Usage: CKPT=<dir with memory_weights.pt> bash run_select.sh <gpu> <tag> <data.jsonl> <limit> -- <eval_select args>
R=$(dirname "$(readlink -f "$0")"); cd $R
GPU=$1; TAG=$2; DATA=$3; LIMIT=$4; shift 5
OUT=${OUTDIR:-$R/out}; mkdir -p $OUT
CUDA_VISIBLE_DEVICES=$GPU python scripts/eval_select.py --model ${MODEL:-Qwen/Qwen3-0.6B} --window 2048 --sinks 4 \
  --memory ${CKPT:-$R/ckpt/dialog-inattn-turn/final} --data $DATA --limit $LIMIT --out $OUT/$TAG.json "$@" > $OUT/$TAG.log 2>&1
echo "== $TAG rc=$?"; grep -A14 "^select=" $OUT/$TAG.log | grep -E "^select=|category|retention/"
