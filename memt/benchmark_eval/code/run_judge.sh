#!/usr/bin/bash
# usage: JUDGE_GPUS=6,7 bash run_judge.sh <work dir> <answers dir>
# Judges every answer file <answers dir>/<bench>/<format>_<mode>.s*.jsonl with the local judge Qwen3.6-35B-A3B
# (vLLM 0.31 in the python of $PY), then scores on CPU and writes <answers dir>/scores.json and report.md.
# Verdicts are cached in <answers dir>/judge_cache.json, so a rerun only judges new answers.
# MemT-1.0-Nano answers go in as <bench>/nano_<mode>.s*.jsonl (run "nano" in the report).
W=$1; OUT=$2
PY=${PY:-python}
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES=${JUDGE_GPUS:-0,1} HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
ARGS="$OUT --judge $W/models/Qwen3.6-35B-A3B --tp ${JUDGE_TP:-2} --mem ${JUDGE_MEM:-0.6} --entity2id $W/data/MemoryAgentBench/entity2id.json"
$PY judge_bench.py $ARGS --judge-only
CUDA_VISIBLE_DEVICES= $PY judge_bench.py $ARGS --no-llm
$PY report.py $OUT/scores.json --md $OUT/report.md
echo JUDGE_EXIT $?
