#!/usr/bin/bash
# Stage 1 (continued pre-training, cross-entropy) of a MemT-1.0-style memory on frozen Qwen3-4B,
# with the optimisation recipe of the MemT-1.0 report for Nano (Sec. 3.6, App. B) instead of the LaCT one.
#
#   ENV=/data/qyl/lact-insight-env/env.sh MODEL=/path/Qwen3-4B TOK=/path/ldc_tok W=/path/run GPUS=5,6,7 \
#     bash train_run/train_cpt_cc.sh
#
# TOK is a directory written by framework/scripts/pretokenize.py. Rerunning the script resumes from the
# newest step_* checkpoint in $W/ckpt (the indexed data stream makes the resume exact).
set -u
: "${MODEL:?}" "${TOK:?}" "${W:?}" "${GPUS:?}"
[ -n "${ENV:-}" ] && source "$ENV" >/dev/null 2>&1
HERE=$(cd "$(dirname "$0")/.." && pwd)
SEQ=${SEQ:-32768}          # report: packs of up to 64K; 32K keeps a 4B run within one H20 card per rank
ACCUM=${ACCUM:-8}          # sequences per optimizer step = number of GPUs x ACCUM (report: 64)
STEPS=${STEPS:-4000}
LAYERS=${LAYERS:-every:4}  # report: one memory block every 4th layer (3, 7, ...), not every layer
N=$(echo "$GPUS" | tr ',' '\n' | wc -l)
export CUDA_VISIBLE_DEVICES=$GPUS HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 NCCL_NVLS_ENABLE=0
mkdir -p "$W"; cd "$HERE/framework"
LAST=$(ls -d "$W"/ckpt/step_* 2>/dev/null | grep -E '/step_[0-9]+$' | sed 's/.*step_//' | sort -n | tail -1)
if [ -n "$LAST" ]; then INIT=(--memory "$W/ckpt/step_$LAST" --resume "$W/ckpt/step_$LAST"); else
  HEADS=$(python - <<PY
import json; c = json.load(open("$MODEL/config.json")); print(c["num_attention_heads"] * c["head_dim"] // 128)
PY
)
  # 128-dim heads = the backbone's own attention heads, so q/k/v can be copied from them (App. B.1)
  INIT=(--memory new --mem-core matrix --mem-heads "$HEADS" --mem-head-dim 128 --mem-layers "$LAYERS"
        --mem-qkv-init attention --mem-qkv-act none --mem-inject pre_o_proj --mem-out-scale memt
        --mem-output-gate-rank 256 --mem-forget-rank 256 --mem-stable-matrix-gates --mem-chunk 2048)
fi
python -m torch.distributed.run --nproc_per_node "$N" --master_addr 127.0.0.1 --master_port "${PORT:-29631}" \
  scripts/train_memory.py --model "$MODEL" --data "tok:$TOK" \
  --seq-len "$SEQ" --batch-size 1 --grad-accum "$ACCUM" --steps "$STEPS" \
  --window 2048 --sinks 0 --attn-segment 2048 --attn-impl flex_attention "${INIT[@]}" \
  --lr 1e-4 --lr-schedule wsd --decay-steps 0 --warmup 600 \
  --adam-betas 0.9 0.999 --adam-eps 1e-8 --weight-decay 0.01 --clip 1 \
  --grad-ckpt --seed 42 --log-every 1 --save-every "${SAVE:-200}" --keep-last "${KEEP:-3}" \
  --ce-weight 1 --distill-weight 0 --fail-on-nonfinite --stable-grad-norm \
  --progress-file "$W/progress.json" --out "$W/ckpt" >> "$W/train.log" 2>&1
echo "TRAIN_EXIT $? $(date)" >> "$W/train.log"
