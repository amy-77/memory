#!/usr/bin/bash
# h20-1, E14: does selective writing help on LoCoMo AT ALL? Upper bound: write only the sessions that hold the
# evidence of the question (labels). Control: as many randomly chosen sessions. Same model as E13.
source /data/qyl/lact-insight-env/env.sh >/dev/null 2>&1
D=/data/qyl/sw_protect_20261122; cd $D; export HF_HUB_OFFLINE=1 TMPDIR=/data/qyl/tmp; mkdir -p out/locomo logs/locomo
gunzip -c locomo_compact2.jsonl.gz > data_locomo/locomo_compact2.jsonl
python locomo_expand.py data_locomo/locomo_compact2.jsonl data_locomo/locomo2_g4.jsonl 4
B="--model /data/models/Qwen/Qwen3-0.6B --data data_locomo/locomo2_g4.jsonl --max-segment 4096 --window 2048 --sinks 4 --memory ckpt/dlg16_new/final"
run() { g=$1; n=$2; shift 2
  CUDA_VISIBLE_DEVICES=$g python scripts/eval_select.py $B --out out/locomo/$n.json "$@" > logs/locomo/$n.log 2>&1
  echo "== $n $(grep -E 'turn selection' logs/locomo/$n.log | cut -c1-160) $(grep -E 'Traceback|Error' logs/locomo/$n.log | tail -1 | cut -c1-120)"; }
run 2 F_mem_write_evidence_only --select turn_evidence &
run 5 G_mem_write_random_same_count --select turn_random --turn-k -1 &
wait
python locomo_summary.py out/locomo
echo E14_DONE
