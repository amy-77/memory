#!/usr/bin/bash
# h20-9-5: train two 0.6B dialogue models from a fresh branch (fixed 16 turns / mixed 12-48 turns),
# then compare selective writing + scored replay on both. Frees the GPUs used by the LiveMem servers first.
source /data0/qyl/lact-env/env.sh >/dev/null 2>&1
D=/data0/qyl/select_write_20261121; cd $D; mkdir -p data ckpt out logs
export HF_HUB_OFFLINE=1 NCCL_NVLS_ENABLE=0
for p in $(ps -eo pid,cmd | grep "livemem_repro_20261121" | grep -E "eval.pipeline|vllm.entrypoints|serve_|run_eval|more_servers|serve.sh" | grep -v grep | awk '{print $1}'); do kill $p 2>/dev/null; done
sleep 30; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr "\n" ";"; echo
gen() { [ -s data/$1.jsonl ] || python scripts/make_synth.py --out data/$1.jsonl --n $2 --num-turns $3 --num-facts 8 --words-per-turn 200 --queries-per-category 4 --seed $4; }
gen v8f_train 10000 16 101
for t in 16 32 64 128; do gen rc_t${t}_f8 100 $t 7; done
s=301; for t in 12 16 24 32 48; do gen var_t$t 2000 $t $s; s=$((s + 1)); done
[ -s data/var_train.jsonl ] || python - <<PY
import random
rows = [l for t in (12, 16, 24, 32, 48) for l in open(f"data/var_t{t}.jsonl")]
random.Random(0).shuffle(rows); open("data/var_train.jsonl", "w").writelines(rows); print("var_train", len(rows))
PY
LAYOUT="--mem-heads 4 --mem-head-dim 512 --mem-chunk 2048 --mem-delta-readout --mem-gate-init 0.1 --mem-qkv-init attention --mem-qkv-act none --mem-inject pre_o_proj --mem-out-ref attention"
( CUDA_VISIBLE_DEVICES=0,1,2,3 EVAL_GPU=0 MASTER_PORT=29601 NGPU=4 ACCUM=2 STEPS=1200 TRAIN=data/v8f_train.jsonl TEST=data/rc_t16_f8.jsonl LIMIT=100 \
    INIT=new EXTRA="$LAYOUT" OUT=$D/ckpt/dlg16_new bash run_dialog.sh > logs/train_dlg16.log 2>&1 ) &
( CUDA_VISIBLE_DEVICES=4,5,6,7 EVAL_GPU=4 MASTER_PORT=29602 NGPU=4 ACCUM=2 STEPS=1200 TRAIN=data/var_train.jsonl TEST=data/rc_t32_f8.jsonl LIMIT=100 \
    INIT=new EXTRA="$LAYOUT" OUT=$D/ckpt/dlgvar_new bash run_dialog.sh > logs/train_dlgvar.log 2>&1 ) &
wait
echo TRAIN_DONE; tail -4 logs/train_dlg16.log logs/train_dlgvar.log
for m in dlg16_new dlgvar_new; do
  [ -f ckpt/$m/final/memory_weights.pt ] || continue
  for t in 32 64; do
    CKPT=$D/ckpt/$m/final OUTDIR=$D/out/$m bash batch_select.sh data/rc_t${t}_f8.jsonl t$t 100 "0 1 2 3 4 5 6 7" > logs/select_${m}_t$t.txt 2>&1
  done
done
echo PIPELINE_DONE
