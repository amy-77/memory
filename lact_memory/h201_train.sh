#!/usr/bin/bash
# h20-1: rebuild the 0.6B dialogue model used for the selective-write tests (same data seeds and settings as
# on h20-9-5; 8 GPUs x 1 dialogue instead of 4 GPUs x 2), so the follow-up experiments can run on these GPUs.
source /data/qyl/lact-insight-env/env.sh >/dev/null 2>&1
D=/data/qyl/sw_20261122; cd $D; mkdir -p data ckpt out logs
export HF_HUB_OFFLINE=1 NCCL_NVLS_ENABLE=0 TMPDIR=/data/qyl/tmp
gen() { [ -s data/$1.jsonl ] || python scripts/make_synth.py --out data/$1.jsonl --n $2 --num-turns $3 --num-facts 8 --words-per-turn 200 --queries-per-category 4 --seed $4; }
gen v8f_train 10000 16 101
for t in 16 32 64 128; do gen rc_t${t}_f8 100 $t 7; done
md5sum data/rc_t32_f8.jsonl data/v8f_train.jsonl
LAYOUT="--mem-heads 4 --mem-head-dim 512 --mem-chunk 2048 --mem-delta-readout --mem-gate-init 0.1 --mem-qkv-init attention --mem-qkv-act none --mem-inject pre_o_proj --mem-out-ref attention"
MODEL=/data/models/Qwen/Qwen3-0.6B CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 EVAL_GPU=0 MASTER_PORT=29701 NGPU=8 ACCUM=1 STEPS=1200 TRAIN=data/v8f_train.jsonl TEST=data/rc_t16_f8.jsonl LIMIT=100 \
    INIT=new EXTRA="$LAYOUT" OUT=$D/ckpt/dlg16_new bash run_dialog.sh > logs/train_dlg16.log 2>&1
echo H201_TRAIN_DONE; tail -8 logs/train_dlg16.log
