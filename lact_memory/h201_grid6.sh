#!/usr/bin/bash
# h20-1, sixth grid: ERASE-THEN-WRITE at inference (no training): the write target is v minus what the memory already
# returns for the key, so a later fact about the same key replaces the old value instead of being added to it.
source /data/qyl/lact-insight-env/env.sh >/dev/null 2>&1
D=/data/qyl/sw_protect_20261122; cd $D; export HF_HUB_OFFLINE=1 TMPDIR=/data/qyl/tmp; mkdir -p out/grid6 logs/grid6
M="--model /data/models/Qwen/Qwen3-0.6B --window 2048 --sinks 4 --memory ckpt/dlg16_new/final"
CUDA_VISIBLE_DEVICES=5 python scripts/eval_select.py $M --data data/rc_t16_f8.jsonl --limit 2 --select none --delta-erase 1.0 > logs/grid6/smoke.log 2>&1
echo "smoke: $(grep -E 'retention/all|Error' logs/grid6/smoke.log | tail -1 | cut -c1-100)"; grep -q Error logs/grid6/smoke.log && exit 1
J=$D/logs/grid6/jobs.txt; : > $J
for d in rc_t16_f8 rc_t32_f8 rc_t64_f8; do for a in 0.5 1.0; do
  echo "$d none_erase$a --select none --delta-erase $a" >> $J
  [ $d != rc_t16_f8 ] && echo "$d a8_erase$a --select turn_attn --turn-k 8 --delta-erase $a" >> $J
done; done
echo "rc_t16_f8 none_erase0 --select none" >> $J
worker() { g=$1; i=0
  while read -r d n args; do
    if [ $((i % 4)) = $g ] && [ ! -s out/grid6/${d}_$n.json ]; then
      CUDA_VISIBLE_DEVICES=$((g + 4)) python scripts/eval_select.py $M --data data/$d.jsonl --limit 40 --out out/grid6/${d}_$n.json $args > logs/grid6/${d}_$n.log 2>&1
      echo "== $d $n | retention $(grep -E 'retention/beyond' logs/grid6/${d}_$n.log | awk '{print $2, $3, $4}') | correction $(grep -E 'correction/beyond' logs/grid6/${d}_$n.log | awk '{print $2, $3, $4, $NF}') $(grep -E 'Traceback|Error' logs/grid6/${d}_$n.log | tail -1 | cut -c1-90)"
    fi
    i=$((i + 1))
  done < $J; }
for g in 0 1 2 3; do worker $g & done; wait
echo GRID6_DONE
