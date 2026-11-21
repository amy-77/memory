#!/usr/bin/bash
# Stage 2: train the memory branches on synthetic multi-turn dialogues (loss on the answers), then
# score recall on unseen dialogues with writes on and off.
#   TRAIN=data/generated/dlg_train.jsonl TEST=data/generated/dlg_test.jsonl \
#   INIT=/path/to/stage1/step_N OUT=/path/to/ckpt NGPU=8 bash run_dialog.sh
# INIT=new starts from a fresh branch (then EXTRA carries the --mem-* layout flags).
# One optimizer step sees NGPU * ACCUM dialogues. Keep STEPS * NGPU * ACCUM <= number of training
# dialogues so that every dialogue is seen at most once: repeated dialogues get memorised
# (16 dialogues x 37 passes: 0.94 on those 16, 0.00 on new ones).
# Optional: WINDOW (2048; must match the checkpoint), SINKS (4), MIN_WRITE (0 = one write per turn),
# STEPS (1200), ACCUM (1), LR (1e-3), WARMUP (100), LIMIT (test dialogues, 200), MODEL, EXTRA.
set -u
export NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE:-0}
[ -n "${ENV_SH:-}" ] && source "$ENV_SH"
cd "$(dirname "$(readlink -f "$0")")"
: "${TRAIN:?set TRAIN to the training jsonl}"; : "${TEST:?set TEST to the test jsonl}"
: "${INIT:?set INIT to a Stage 1 checkpoint directory, or 'new'}"; : "${OUT:?set OUT to the checkpoint directory}"
NGPU=${NGPU:-1}
COMMON=(--model "${MODEL:-Qwen/Qwen3-0.6B}" --window "${WINDOW:-2048}" --sinks "${SINKS:-4}" --max-segment 4096
        --min-write-tokens "${MIN_WRITE:-0}")
[ -n "${EXTRA:-}" ] && COMMON+=($EXTRA)
if [ "$NGPU" -gt 1 ]; then RUN=(torchrun --nproc_per_node "$NGPU" --master_port "${MASTER_PORT:-29500}"); else RUN=(python); fi
mkdir -p "$OUT"
if [ ! -f "$OUT/final/memory_weights.pt" ]; then
  "${RUN[@]}" scripts/train_memory.py "${COMMON[@]}" --attn-impl flex_attention --data "synth:$TRAIN" --memory "$INIT" \
    --steps "${STEPS:-1200}" --grad-accum "${ACCUM:-1}" --lr "${LR:-1e-3}" --warmup "${WARMUP:-100}" --grad-ckpt \
    --seed "${SEED:-1}" --log-every 10 --save-every 200 --keep-last 2 --out "$OUT" || { echo TRAIN_FAILED; exit 1; }
fi
export CUDA_VISIBLE_DEVICES=${EVAL_GPU:-0}
for mode in write nowrite; do
  FLAG=(); [ $mode = nowrite ] && FLAG=(--no-write)
  python scripts/eval_synth.py "${COMMON[@]}" --data "$TEST" --limit "${LIMIT:-200}" --memory "$OUT/final" "${FLAG[@]}" \
    --out "$OUT/eval_$mode.json" > "$OUT/eval_$mode.log" 2>&1 || { echo "eval failed, see $OUT/eval_$mode.log"; exit 1; }
  echo "== $mode"; grep -E "^(retention|correction|recent)/" "$OUT/eval_$mode.log"
done
echo DIALOG_DONE
