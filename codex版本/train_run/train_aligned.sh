#!/usr/bin/env bash
# Qwen3-4B MemT-style CPT with the memory aligned to MemT-1.0 (no decay cap, chunked WY write, MemT norms and
# BF16 rounding). Same recipe as train_8gpu.sh otherwise. One node (8 GPUs) or several nodes (e.g. 2 x 8 = 16).
#
#   RUN_DIR=/shared/run DATA_SPEC='/data/ldc/train/data-*.arrow' MODEL_PATH=/models/Qwen3-4B bash train_aligned.sh
#   # 16 GPUs: run on both nodes with NNODES=2 NODE_RANK=0|1 MASTER_ADDR=<node 0 address> (RUN_DIR on shared storage)
#
# Environment:
#   RUN_DIR, DATA_SPEC (required); MODEL_PATH (Qwen/Qwen3-4B); PYTHON (python)
#   NNODES (1), NODE_RANK (0), NPROC_PER_NODE (8), MASTER_ADDR (node 0; required when NNODES > 1), MASTER_PORT (29577)
#   GLOBAL_BATCH (32 sequences of 32K per update; grad accumulation = GLOBAL_BATCH / (NNODES * NPROC_PER_NODE))
#   STEPS (4768), SAVE_EVERY (240), MEMT_ALIGNED (1; 0 = the unaligned train_8gpu.sh memory, for A/B runs)
#   RESUME / INIT_MEMORY as in train_8gpu.sh. Extra arguments go to train_memory.py (e.g. --stop-at 10).
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
FRAMEWORK="$(cd -- "$SCRIPT_DIR/../framework" && pwd)"
PYTHON="${PYTHON:-python}"
: "${RUN_DIR:?Set RUN_DIR to the training output directory (shared storage for several nodes)}"
: "${DATA_SPEC:?Set DATA_SPEC to the LDC arrow glob, e.g. /data/ldc/train/data-*.arrow}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
GLOBAL_BATCH="${GLOBAL_BATCH:-32}"
WORLD=$((NNODES * NPROC_PER_NODE))
if (( GLOBAL_BATCH % WORLD != 0 )); then
  echo "GLOBAL_BATCH=$GLOBAL_BATCH is not a multiple of $WORLD GPUs" >&2
  exit 1
fi
GRAD_ACCUM=$((GLOBAL_BATCH / WORLD))
export PYTHONDONTWRITEBYTECODE=1
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
mkdir -p "$RUN_DIR"
if [[ -f "$RUN_DIR/ckpt/final/COMPLETE" ]]; then
  echo "Completed run already exists: $RUN_DIR/ckpt/final"
  exit 0
fi
LAST="${RESUME:-$("$PYTHON" "$FRAMEWORK/scripts/latest_checkpoint.py" "$RUN_DIR/ckpt")}"
MEMORY_ARGS=(--memory "${INIT_MEMORY:-new}")
if [[ -n "$LAST" ]]; then
  MEMORY_ARGS=(--resume "$LAST" --exact-text-resume)
elif [[ ! -e "$RUN_DIR/initial" ]]; then
  MEMORY_ARGS+=(--save-initial "$RUN_DIR/initial")
else
  echo 'No resumable checkpoint found in an existing run; set RESUME or choose a new RUN_DIR.' >&2
  exit 1
fi
ALIGNED=()
if [[ "${MEMT_ALIGNED:-1}" == 1 ]]; then
  ALIGNED=(--mem-decay-cap none --mem-write-impl auto --mem-out-norm-eps 1e-6 --mem-qk-norm memt --mem-memt-rounding)
fi
if (( NNODES > 1 )); then
  LAUNCH=(--nnodes "$NNODES" --node_rank "$NODE_RANK" --master_addr "${MASTER_ADDR:?Set MASTER_ADDR to node 0}"
          --master_port "${MASTER_PORT:-29577}")
else
  LAUNCH=(--standalone)
fi
LOG="$RUN_DIR/train.log"
if (( NNODES > 1 )); then LOG="$RUN_DIR/train.node$NODE_RANK.log"; fi
echo "$(date '+%F %T') node $NODE_RANK/$NNODES, $WORLD GPUs, grad accum $GRAD_ACCUM, aligned=${MEMT_ALIGNED:-1}" \
  | tee -a "$RUN_DIR/launch.log"
"$PYTHON" -m torch.distributed.run "${LAUNCH[@]}" --nproc_per_node "$NPROC_PER_NODE" \
  "$FRAMEWORK/scripts/train_memory.py" --model "$MODEL_PATH" --data "text:$DATA_SPEC" \
  --seq-len 32768 --batch-size 1 --grad-accum "$GRAD_ACCUM" --steps "${STEPS:-4768}" \
  --window 2048 --sinks 0 --attn-segment 2048 --attn-impl "${ATTN_IMPL:-flex_attention}" \
  "${MEMORY_ARGS[@]}" --mem-heads 32 --mem-head-dim 128 --mem-layers all \
  --mem-core matrix --mem-chunk 2048 --mem-write-block 256 --mem-stable-matrix-gates \
  --mem-qkv-init attention --mem-qkv-act none --mem-inject pre_o_proj --mem-out-scale memt \
  --mem-output-gate-rank 128 --mem-output-gate-softcap 2 --mem-forget-rank 128 "${ALIGNED[@]}" \
  --lr 1e-3 --warmup 256 --min-lr-ratio .05 --weight-decay .1 --adam-eps 1e-8 \
  --clip 1 --qkv-lr-scale .1 --grad-ckpt --seed 42 --log-every 1 --save-every "${SAVE_EVERY:-240}" \
  --min-doc-tokens 16384 --ce-weight 1 --distill-weight 0 --loss-chunk-size 128 \
  --fail-on-nonfinite --stable-grad-norm --progress-file "$RUN_DIR/progress.json" \
  --out "$RUN_DIR/ckpt" "$@" 2>&1 | tee -a "$LOG"
