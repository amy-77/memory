#!/usr/bin/bash
# h20-1, third grid: PROTECTED WRITE. Keys of the written turns that the question scores highest go into a bank;
# every later write is projected so that the read-out at the banked keys does not change. 100 dialogues.
source /data/qyl/lact-insight-env/env.sh >/dev/null 2>&1
D=/data/qyl/sw_protect_20261122; cd $D; export HF_HUB_OFFLINE=1 TMPDIR=/data/qyl/tmp; mkdir -p out/grid3 logs/grid3
[ -e data ] || ln -s /data/qyl/sw_20261122/data data; [ -e ckpt ] || ln -s /data/qyl/sw_20261122/ckpt ckpt
M="--model /data/models/Qwen/Qwen3-0.6B --window 2048 --sinks 4 --memory ckpt/dlg16_new/final"
PR="--query-source alone --replay pseudo --probe-src bank --replay-lr-scale 0 --protect --replay-samples 512"
J=$D/logs/grid3/jobs.txt; : > $J
for d in rc_t64_f8 rc_t32_f8; do
  for k in 16 8; do A="--select turn_attn --turn-k $k"
    echo "$d a${k}_base $A" >> $J
    echo "$d a${k}_protect_rel8 $A $PR --bank-score relevance --replay-keep 8" >> $J
    echo "$d a${k}_protect_rel16 $A $PR --bank-score relevance --replay-keep 16" >> $J
    echo "$d a${k}_protect_lr8 $A $PR --bank-score lr --replay-keep 8" >> $J
  done
  echo "$d q8_base --select turn_qk --turn-k 8" >> $J
  echo "$d q8_protect_rel8 --select turn_qk --turn-k 8 $PR --bank-score relevance --replay-keep 8" >> $J
done
echo "rc_t32_f8 all32_base --select turn_attn --turn-k 32" >> $J
echo "rc_t32_f8 all32_protect_rel8 --select turn_attn --turn-k 32 $PR --bank-score relevance --replay-keep 8" >> $J
echo "rc_t32_f8 all32_protect_rel4 --select turn_attn --turn-k 32 $PR --bank-score relevance --replay-keep 4" >> $J
worker() { g=$1; i=0
  while read -r d n args; do
    if [ $((i % 8)) = $g ] && [ ! -s out/grid3/${d}_$n.json ]; then
      CUDA_VISIBLE_DEVICES=$g python scripts/eval_select.py $M --data data/$d.jsonl --limit 100 --out out/grid3/${d}_$n.json $args > logs/grid3/${d}_$n.log 2>&1
      echo "== $d $n | $(grep -E 'turn selection' logs/grid3/${d}_$n.log | sed 's/.*fact turn written in //') | $(grep -E 'retention/beyond' logs/grid3/${d}_$n.log | awk '{print $2, $3, $4}') $(grep -E 'Traceback|Error' logs/grid3/${d}_$n.log | tail -1 | cut -c1-90)"
    fi
    i=$((i + 1))
  done < $J; }
for g in 0 1 2 3 4 5 6 7; do worker $g & done; wait
echo GRID3_DONE
