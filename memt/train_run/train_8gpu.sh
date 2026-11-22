#!/usr/bin/bash
# MemT-style memory on frozen Qwen3-4B, recipe of memory_research_20261119/scripts/train.sh memt (heads 16 -> 32 for 4B).
source /data/qyl/lact-insight-env/env.sh >/dev/null 2>&1
export NCCL_NVLS_ENABLE=0 PYTHONDONTWRITEBYTECODE=1
W=/data/qyl/memt4b_20261120
R=/data/qyl/lact-insight/overleaf_6af80380/memory_research_20261119
M=/data/models/Qwen/Qwen3-4B
mkdir -p $W; cd $R/frameworks/memt
if [ ! -f $W/initial/memory_weights.pt ] && [ ! -d $W/ckpt ]; then
CUDA_VISIBLE_DEVICES=0 python - <<PY
import json, sys, torch
from transformers import AutoModelForCausalLM
sys.path.insert(0, "$R/frameworks/memt")
from lact_qwen import MemoryConfig, attach_memory, save_memory, set_attention_window
c = json.load(open("$R/configs/memt.json")); torch.manual_seed(42)
m = AutoModelForCausalLM.from_pretrained("$M", dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True)
c["memory"]["num_heads"] = m.model.layers[0].self_attn.o_proj.in_features // c["memory"]["head_dim"]
set_attention_window(m, c["attention_window"], c["attention_sinks"], c["attention_segment"])
attach_memory(m, MemoryConfig.from_dict(c["memory"]))
save_memory(m, "$W/initial", extra={"step": 0, "initialization_seed": 42, "model": "$M"})
print("initialized", c["memory"]["num_heads"], "heads")
PY
fi
while pgrep -f eval_synth.py > /dev/null; do sleep 20; done
RES=(); LAST=$(ls -d $W/ckpt/step_* 2>/dev/null | sort -t_ -k2 -n | tail -1)
if [ -n "$LAST" ]; then INIT=$LAST; RES=(--resume $LAST); else INIT=$W/initial; fi
( until grep -q "^step 2 " $W/train.log 2>/dev/null; do sleep 30; done; rm -rf $W/initial ) &
python -m torch.distributed.run --nproc_per_node 8 --master_addr 127.0.0.1 --master_port 29577 \
  scripts/train_memory.py --model $M --data "text:/data/qyl/lact-insight-env/datasets/ldc/train/data-*.arrow" \
  --seq-len 32768 --batch-size 1 --grad-accum 4 --steps 4768 --stop-at 480 \
  --window 2048 --sinks 0 --attn-segment 2048 --attn-impl flex_attention \
  --memory $INIT "${RES[@]}" \
  --lr 1e-3 --warmup 256 --weight-decay .1 --adam-eps 1e-15 --clip 1 --qkv-lr-scale .1 \
  --grad-ckpt --seed 42 --log-every 1 --save-every 240 --keep-last 0 --min-doc-tokens 16384 \
  --ce-weight 1 --distill-weight 0 --fail-on-nonfinite --stable-grad-norm \
  --progress-file $W/progress.json --out $W/ckpt >> $W/train.log 2>&1
echo "TRAIN_EXIT $?" >> $W/train.log
