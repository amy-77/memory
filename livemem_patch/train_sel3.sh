#!/usr/bin/bash
# Restart after two problems: GPU 2 is now used by another user (training ran out of memory there), and the
# multi-hop test sets exist as test_single (no pre-sampled 2000-question files). Training: GPUs 0,1,3,6.
source /data0/qyl/livemem_repro_20261121/env.sh
cd $W/LiveMem; export PYTHONPATH=. NCCL_NVLS_ENABLE=0 WANDB_MODE=disabled; unset http_proxy https_proxy; export no_proxy=localhost,127.0.0.1 PY=python
for p in $(ps -eo pid,cmd | grep -E "bash train_sel2.sh|eval_model.sh" | grep -v grep | awk '{print $1}'); do kill $p 2>/dev/null; done
for p in $(ps -eo pid,cmd | grep -E "eval.pipeline --model livemem|vllm.entrypoints.openai|train_sft.py" | grep -v grep | grep -v "served-model-name judge" | awk '{print $1}'); do
  readlink /proc/$p/cwd 2>/dev/null | grep -q livemem_repro_20261121 && kill $p 2>/dev/null; done
sleep 30; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr "\n" ";"; echo
( CUDA_VISIBLE_DEVICES=0,1,3,6 torchrun --nproc-per-node=4 --master-port=29613 train/sft/train_sft.py --config $W/sel/ft_sel.yaml > $W/logs/ft_sel.log 2>&1; echo "FT_sel rc=$?" ) &
for ds in 2wiki_single musique_single hotpotqa_single; do
  for pol in ones oracle:0; do python tools/build_select_masks.py --dataset $ds --limit 200 --out $W/masks/$(echo $pol | tr ':' '_') --policy $pol 2>&1 | tail -1; done
done
grep -q "Application startup complete" $W/logs/sub_judge.log || echo "JUDGE NOT UP"
bash $W/sel/eval_model.sh sft $W/models/LiveMem-SFT-vllm 4 9100 > $W/logs/evm_sft.log 2>&1 &
bash $W/sel/eval_model.sh rl $W/models/LiveMem-RL-vllm 5 9200 > $W/logs/evm_rl.log 2>&1 &
for step in 40 80 120; do
  until [ -f $W/outputs/sft/ft_sel/step_$step/trainable.pt ] || grep -q "FT_sel rc" $W/logs/train_sel3.log; do sleep 60; done
  [ -f $W/outputs/sft/ft_sel/step_$step/trainable.pt ] || break
  sleep 30
  BASE_MERGED=$W/models/LiveMem-SFT-vllm QWEN3=$BASE_MODEL STEP=$W/outputs/sft/ft_sel/step_$step OUT=$W/models/ft_sel_s$step DEVICE=cpu \
    python tools/export_checkpoint.py > $W/logs/export_sel_s$step.log 2>&1
  [ -f $W/models/ft_sel_s$step/config.json ] || { echo "export failed step $step"; continue; }
  for f in tokenizer.json tokenizer_config.json chat_template.jinja generation_config.json; do [ -e $W/models/ft_sel_s$step/$f ] || cp $W/models/LiveMem-SFT/$f $W/models/ft_sel_s$step/; done
  until grep -q "EVAL_MODEL_DONE sft" $W/logs/evm_sft.log && ! pgrep -f "eval_model.sh sel" >/dev/null; do sleep 60; done   # GPU 4 is free again
  bash $W/sel/eval_model.sh sel$step $W/models/ft_sel_s$step 4 $((9300 + step)) > $W/logs/evm_sel$step.log 2>&1
done
wait
echo TRAIN_SEL3_DONE
