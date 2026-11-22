#!/usr/bin/bash
# h20-1, eighth grid: Robins rehearsal at high interference (16 turns written), 100 dialogues, strength 20:
# in-context question probes (offline: taken from a full-dialogue pass) versus question-alone probes (causal).
source /data/qyl/lact-insight-env/env.sh >/dev/null 2>&1
D=/data/qyl/sw_20261122; cd $D; export HF_HUB_OFFLINE=1 TMPDIR=/data/qyl/tmp; mkdir -p out/grid8 logs/grid8
cp /data/qyl/sw_protect_20261122/scripts/eval_select.py scripts/eval_select_g8.py 2>/dev/null; cp -r /data/qyl/sw_protect_20261122/lact_qwen lact_qwen_g8 2>/dev/null
M="--model /data/models/Qwen/Qwen3-0.6B --window 2048 --sinks 4 --memory ckpt/dlg16_new/final"
R="--query-source alone --replay pseudo --probe-src query --replay-lr-scale 20"
J=logs/grid8/jobs.txt; : > $J
echo "rc_t32_f8 a16_alone_s20 --select turn_attn --turn-k 16 $R" >> $J
echo "rc_t64_f8 a16_base --select turn_attn --turn-k 16" >> $J
echo "rc_t64_f8 a16_ctxq_s20 --select turn_attn --turn-k 16 $R --probe-ctx" >> $J
echo "rc_t64_f8 a16_alone_s20 --select turn_attn --turn-k 16 $R" >> $J
echo "rc_t32_f8 a8_alone_s20 --select turn_attn --turn-k 8 $R" >> $J
echo "rc_t32_f8 a8_ctxq_s20 --select turn_attn --turn-k 8 $R --probe-ctx" >> $J
worker() { g=$1; slot=$2; i=0
  while read -r d n args; do
    if [ $((i % 2)) = $slot ] && [ ! -s out/grid8/${d}_$n.json ]; then
      CUDA_VISIBLE_DEVICES=$g python scripts/eval_select.py $M --data data/$d.jsonl --limit 100 --out out/grid8/${d}_$n.json $args > logs/grid8/${d}_$n.log 2>&1
      echo "== $d $n | retention $(grep -E 'retention/beyond' logs/grid8/${d}_$n.log | awk '{print $2, $3, $4}') | correction $(grep -E 'correction/beyond' logs/grid8/${d}_$n.log | awk '{print $2, $3, $NF}') $(grep -E 'Traceback|Error' logs/grid8/${d}_$n.log | tail -1 | cut -c1-90)"
    fi
    i=$((i + 1))
  done < $J; }
worker 4 0 & worker 5 1 & wait
echo GRID8_DONE
