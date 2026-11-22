#!/usr/bin/bash
# h20-1, seventh grid: STREAMING (causal) selection. Every turn is decided when it arrives, from that turn and the
# question only: write iff its attention-mass score >= a threshold calibrated on a separate DEV set (seed 8).
# Also: q.k scores computed causally (question appended to each turn alone) instead of from a full-dialogue pass.
source /data/qyl/lact-insight-env/env.sh >/dev/null 2>&1
D=/data/qyl/sw_protect_20261122; cd $D; export HF_HUB_OFFLINE=1 TMPDIR=/data/qyl/tmp; mkdir -p out/grid7 logs/grid7 data_dev
M="--model /data/models/Qwen/Qwen3-0.6B --window 2048 --sinks 4 --memory ckpt/dlg16_new/final"
[ -s data_dev/dev_t32_f8.jsonl ] || python scripts/make_synth.py --out data_dev/dev_t32_f8.jsonl --n 40 --num-turns 32 --num-facts 8 --words-per-turn 200 --queries-per-category 4 --seed 8
G=${GPUS:-"6 7"}; set -- $G; G1=$1; G2=$2
# 1) dev: dump the per-turn attention scores (and q.k layer 19 / 21 scores, causal) for calibration
rm -f out/grid7/dev_attn.jsonl out/grid7/dev_qk19.jsonl
CUDA_VISIBLE_DEVICES=$G1 python scripts/eval_select.py $M --data data_dev/dev_t32_f8.jsonl --limit 40 --select turn_attn --turn-k 999 \
  --dump-turn-scores out/grid7/dev_attn.jsonl > logs/grid7/dev_attn.log 2>&1 &
CUDA_VISIBLE_DEVICES=$G2 python scripts/eval_select.py $M --data data_dev/dev_t32_f8.jsonl --limit 40 --select turn_qk --qk-source turn --qk-tokens all \
  --turn-layer 19 --turn-k 999 --dump-turn-scores out/grid7/dev_qk19.jsonl > logs/grid7/dev_qk19.log 2>&1 &
wait
echo "## calibration on dev (attention)"; python scripts/calibrate_thresh.py out/grid7/dev_attn.jsonl 0.05 | tee logs/grid7/calib_attn.txt
echo "## calibration on dev (q.k layer 19, causal)"; python scripts/calibrate_thresh.py out/grid7/dev_qk19.jsonl 0.05 | tee logs/grid7/calib_qk19.txt
TA=$(grep CHOSEN logs/grid7/calib_attn.txt | awk '{print $2}'); TQ=$(grep CHOSEN logs/grid7/calib_qk19.txt | awk '{print $2}')
TA02=$(grep "q=0.02" logs/grid7/calib_attn.txt | awk '{print $3}'); TA10=$(grep "q=0.10" logs/grid7/calib_attn.txt | awk '{print $3}')
J=logs/grid7/jobs.txt; : > $J
for d in rc_t32_f8 rc_t64_f8; do
  echo "$d attn_stream05 --select turn_attn --write-thresh $TA" >> $J
  echo "$d attn_stream02 --select turn_attn --write-thresh $TA02" >> $J
  echo "$d attn_stream10 --select turn_attn --write-thresh $TA10" >> $J
  echo "$d qk19_stream05 --select turn_qk --qk-source turn --qk-tokens all --turn-layer 19 --write-thresh $TQ" >> $J
  echo "$d qk19_causal_k8 --select turn_qk --qk-source turn --qk-tokens all --turn-layer 19 --turn-k 8" >> $J
done
worker() { g=$1; slot=$2; i=0
  while read -r d n args; do
    if [ $((i % 2)) = $slot ] && [ ! -s out/grid7/${d}_$n.json ]; then
      CUDA_VISIBLE_DEVICES=$g python scripts/eval_select.py $M --data data/$d.jsonl --limit 40 --out out/grid7/${d}_$n.json $args > logs/grid7/${d}_$n.log 2>&1
      echo "== $d $n | $(grep -E 'turn selection' logs/grid7/${d}_$n.log | sed 's/.*fact turn written in //') | retention $(grep -E 'retention/beyond' logs/grid7/${d}_$n.log | awk '{print $2, $3, $4}') | correction $(grep -E 'correction/beyond' logs/grid7/${d}_$n.log | awk '{print $2, $3, $NF}') $(grep -E 'Traceback|Error' logs/grid7/${d}_$n.log | tail -1 | cut -c1-90)"
    fi
    i=$((i + 1))
  done < $J; }
worker $G1 0 & worker $G2 1 & wait
echo GRID7_DONE
