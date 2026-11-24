#!/usr/bin/bash
# One extra Nano shard that skips qids already answered elsewhere:
#   bash nano_shard.sh <work dir> <env.sh> <gpu> <bench> <mode> <shard> <nshards> <skip file>
W=$1; ENV=$2; GPU=$3; BENCH=$4; MODE=$5; SHARD=$6; NSH=$7; SKIP=$8
source $ENV >/dev/null 2>&1
cd $W/bench
export CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1 TMPDIR=$W/tmp PYTHONDONTWRITEBYTECODE=1 HF_MODULES_CACHE=$W/tmp/hf_modules
mkdir -p $W/out_nano/$BENCH $W/logs/nano_jobs
python bench_nano_hf.py --bench $BENCH --data $W/data --model $W/models/MemT-1.0-Nano --mode $MODE --shard $SHARD \
  --nshards $NSH --skip $SKIP --out $W/out_nano/$BENCH/chat_${MODE}.s${SHARD}of${NSH}.jsonl \
  > $W/logs/nano_jobs/x_${BENCH}_${MODE}_s${SHARD}of${NSH}.log 2>&1
