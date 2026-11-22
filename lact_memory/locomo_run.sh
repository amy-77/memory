#!/usr/bin/bash
# h20-1: LoCoMo baseline for the 0.6B LaCT dialogue model (trained on synthetic dialogues only).
source /data/qyl/lact-insight-env/env.sh >/dev/null 2>&1
D=/data/qyl/sw_protect_20261122; cd $D; export HF_HUB_OFFLINE=1 TMPDIR=/data/qyl/tmp; mkdir -p out/locomo logs/locomo
md5sum data_locomo/locomo_compact.jsonl
python locomo_expand.py data_locomo/locomo_compact.jsonl data_locomo/locomo_g4.jsonl 4
B="--model /data/models/Qwen/Qwen3-0.6B --data data_locomo/locomo_g4.jsonl --max-segment 4096"
run() { g=$1; n=$2; shift 2
  CUDA_VISIBLE_DEVICES=$g python scripts/eval_select.py $B --out out/locomo/$n.json "$@" > logs/locomo/$n.log 2>&1
  echo "== $n $(grep -E 'Traceback|Error' logs/locomo/$n.log | tail -1 | cut -c1-120)"; }
G=($GPUS)
run ${G[0]} A_full_attention --window 0 --memory none &
run ${G[1]} B_window2048 --window 2048 --sinks 4 --memory none &
run ${G[2]} C_mem_write_per_session --window 2048 --sinks 4 --memory ckpt/dlg16_new/final --select none &
run ${G[3]} D_mem_no_write --window 2048 --sinks 4 --memory ckpt/dlg16_new/final --select turn_random --turn-k 0 &
wait
run ${G[0]} E_mem_write_per_256tok --window 2048 --sinks 4 --memory ckpt/dlg16_new/final --select none --max-segment 256
python locomo_summary.py out/locomo
echo LOCOMO_DONE
