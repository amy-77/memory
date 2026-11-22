#!/usr/bin/bash
# Supervisor v2: train the 4B MemT run to 4768 steps in legs; between legs (GPUs free) evaluate the
# new checkpoints, then resume from the newest checkpoint on the object store. Restarts after crashes.
source /data/qyl/lact-insight-env/env.sh >/dev/null 2>&1
export NCCL_NVLS_ENABLE=0 PYTHONDONTWRITEBYTECODE=1
W=/data/qyl/memt4b_20261120
R=/data/qyl/lact-insight/overleaf_6af80380/memory_research_20261119
M=/data/models/Qwen/Qwen3-4B
STOPS="480 1440 2400 3600"
cd $R/frameworks/memt
for attempt in $(seq 1 60); do
  while pgrep -f "scripts/train_memory.py" > /dev/null; do sleep 60; done
  # let the mover finish any checkpoint that still has its big files locally (up to 60 min)
  for i in $(seq 1 60); do [ -z "$(ls $W/ckpt/step_*/optimizer.pt $W/ckpt/final/memory_weights.pt 2>/dev/null)" ] && break; sleep 60; done
  bash $W/eval_ckpts.sh >> $W/eval_ckpts.out 2>&1
  [ -e $W/ckpt/final ] && { echo "$(date) final exists, done" >> $W/supervisor.log; exit 0; }
  LAST=$(ls $W/ckpt/step_*/MOVED.txt 2>/dev/null | sed -E 's|.*/step_([0-9]+)/MOVED.txt|\1|' | sort -n | tail -1)
  if [ -z "$LAST" ]; then echo "$(date) no checkpoint to resume from, stopping" >> $W/supervisor.log; exit 1; fi
  RES=$(cat $W/ckpt/step_$LAST/MOVED.txt)
  [ -f $RES/optimizer.pt ] || { echo "$(date) $RES has no optimizer.pt, stopping" >> $W/supervisor.log; exit 1; }
  STOP=(); for s in $STOPS; do [ $s -gt $LAST ] && { STOP=(--stop-at $s); break; }; done
  echo "$(date) attempt $attempt: resume from $RES ${STOP[*]}" >> $W/supervisor.log
  python -m torch.distributed.run --nproc_per_node 8 --master_addr 127.0.0.1 --master_port 29577 \
    scripts/train_memory.py --model $M --data "text:/data/qyl/lact-insight-env/datasets/ldc/train/data-*.arrow" \
    --seq-len 32768 --batch-size 1 --grad-accum 4 --steps 4768 "${STOP[@]}" \
    --window 2048 --sinks 0 --attn-segment 2048 --attn-impl flex_attention \
    --memory $RES --resume $RES --exact-text-resume \
    --lr 1e-3 --warmup 256 --weight-decay .1 --adam-eps 1e-15 --clip 1 --qkv-lr-scale .1 \
    --grad-ckpt --seed 42 --log-every 1 --save-every 240 --keep-last 0 --min-doc-tokens 16384 \
    --ce-weight 1 --distill-weight 0 --fail-on-nonfinite --stable-grad-norm \
    --progress-file $W/progress.json --out $W/ckpt >> $W/train.log 2>&1
  echo "LEG_EXIT $? $(date)" >> $W/train.log
  sleep 120
done
