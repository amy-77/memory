#!/usr/bin/env bash
# Portable Qwen3-4B CPT entry point. Use a new RUN_DIR for the corrected recipe.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
FRAMEWORK="$(cd -- "$SCRIPT_DIR/../framework" && pwd)"
PYTHON="${PYTHON:-python}"
: "${RUN_DIR:?Set RUN_DIR to the training output directory}"
: "${DATA_SPEC:?Set DATA_SPEC to a dataset path, arrow glob, or hf: specification}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B}"
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
COMPAT=()
if [[ "${LEGACY_RESUME:-0}" == 1 ]]; then
  COMPAT=(--text-packing legacy --legacy-cosine --adam-eps 1e-15)
fi
"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node "${NPROC_PER_NODE:-8}" \
  "$FRAMEWORK/scripts/train_memory.py" --model "$MODEL_PATH" --data "text:$DATA_SPEC" \
  --seq-len 32768 --batch-size 1 --grad-accum 4 --steps 4768 \
  --window 2048 --sinks 0 --attn-segment 2048 --attn-impl "${ATTN_IMPL:-flex_attention}" \
  "${MEMORY_ARGS[@]}" --mem-heads 32 --mem-head-dim 128 --mem-layers all \
  --mem-core matrix --mem-chunk 2048 --mem-write-block 256 --mem-stable-matrix-gates \
  --mem-qkv-init attention --mem-qkv-act none --mem-inject pre_o_proj --mem-out-scale memt \
  --mem-output-gate-rank 128 --mem-output-gate-softcap 2 --mem-forget-rank 128 \
  --lr 1e-3 --warmup 256 --min-lr-ratio .05 --weight-decay .1 --adam-eps 1e-8 \
  --clip 1 --qkv-lr-scale .1 --grad-ckpt --seed 42 --log-every 1 --save-every 240 \
  --min-doc-tokens 16384 --ce-weight 1 --distill-weight 0 --loss-chunk-size 128 \
  --fail-on-nonfinite --stable-grad-norm --progress-file "$RUN_DIR/progress.json" \
  --out "$RUN_DIR/ckpt" "${COMPAT[@]}" "$@" 2>&1 | tee -a "$RUN_DIR/train.log"
