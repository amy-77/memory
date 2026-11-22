#!/usr/bin/bash
# After the selective-write fine-tune: export step 120, start the control fine-tune (same data and steps, no
# selection) on the same four GPUs, and evaluate the checkpoints on GPU 4.
source /data0/qyl/livemem_repro_20261121/env.sh
cd $W/LiveMem; export PYTHONPATH=. NCCL_NVLS_ENABLE=0 WANDB_MODE=disabled; unset http_proxy https_proxy; export no_proxy=localhost,127.0.0.1 PY=python
until grep -q "FT_sel rc" $W/logs/train_sel3.log; do sleep 60; done
grep "FT_sel rc" $W/logs/train_sel3.log
for p in $(ps -eo pid,cmd | grep "bash train_sel3.sh" | grep -v grep | awk '{print $1}'); do kill $p 2>/dev/null; done
sleep 20
( CUDA_VISIBLE_DEVICES=0,1,3,6 torchrun --nproc-per-node=4 --master-port=29614 train/sft/train_sft.py --config $W/sel/ft_ctrl.yaml > $W/logs/ft_ctrl.log 2>&1; echo "FT_ctrl rc=$?" ) &
export_ck() { # run step
  [ -f $W/models/ft_$1_s$2/config.json ] && return 0
  BASE_MERGED=$W/models/LiveMem-SFT-vllm QWEN3=$BASE_MODEL STEP=$W/outputs/sft/ft_$1/step_$2 OUT=$W/models/ft_$1_s$2 DEVICE=cpu \
    python tools/export_checkpoint.py > $W/logs/export_$1_s$2.log 2>&1
  for f in tokenizer.json tokenizer_config.json chat_template.jinja generation_config.json; do [ -e $W/models/ft_$1_s$2/$f ] || cp $W/models/LiveMem-SFT/$f $W/models/ft_$1_s$2/; done
  [ -f $W/models/ft_$1_s$2/config.json ]
}
export_ck sel 120 || echo "export failed sel 120"
until grep -qE "NOPACK_DONE|Traceback" $W/logs/nopack.log; do sleep 60; done
# the final selective checkpoint: all policies on LongMemEval, the two basic ones on the multi-hop packs too
bash $W/sel/eval_model.sh sel120 $W/models/ft_sel_s120 4 9600 > $W/logs/evm_sel120.log 2>&1
POLICIES="bm25sub_10_0 attn_10_0 attn_5_0" DATASETS=longmemeval_s_sub SKIP_TRUNC=1 bash $W/sel/eval_model.sh sel120 $W/models/ft_sel_s120 4 9620 > $W/logs/evm_sel120b.log 2>&1
SKIP_TRUNC=1 bash $W/sel/eval_model.sh sel80 $W/models/ft_sel_s80 4 9640 > $W/logs/evm_sel80.log 2>&1
until grep -q "FT_ctrl rc" $W/logs/phase2.log; do sleep 60; done
export_ck ctrl 120 && bash $W/sel/eval_model.sh ctrl120 $W/models/ft_ctrl_s120 4 9660 > $W/logs/evm_ctrl120.log 2>&1
wait
echo PHASE2_DONE
