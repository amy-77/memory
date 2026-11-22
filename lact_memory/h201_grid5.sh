#!/usr/bin/bash
# h20-1, fifth grid: PER-LAYER KEY SELECTION. Inside every written turn each layer writes only its own best keys,
# scored by its own in-context question read vectors (all question tokens, best match), SnapKV-style pooling.
source /data/qyl/lact-insight-env/env.sh >/dev/null 2>&1
D=/data/qyl/sw_protect_20261122; cd $D; export HF_HUB_OFFLINE=1 TMPDIR=/data/qyl/tmp; mkdir -p out/grid5 logs/grid5
M="--model /data/models/Qwen/Qwen3-0.6B --window 2048 --sinks 4 --memory ckpt/dlg16_new/final"
K="--qk-tokens all --key-pool 7"
PR="--query-source alone --replay pseudo --probe-src bank --replay-lr-scale 0 --protect --replay-samples 512 --replay-keep 8"
J=$D/logs/grid5/jobs.txt; : > $J
for d in rc_t32_f8 rc_t64_f8; do
  ALL="--select turn_random --turn-k 999"          # every turn written
  echo "$d allturns_keylayer25 $ALL $K --key-select layer --key-frac 0.25" >> $J
  echo "$d allturns_keyrand25 $ALL $K --key-select random --key-frac 0.25" >> $J
  echo "$d allturns_keylayer10 $ALL $K --key-select layer --key-frac 0.10" >> $J
  A="--select turn_attn --turn-k 8"
  echo "$d a8_keylayer25 $A $K --key-select layer --key-frac 0.25" >> $J
  echo "$d a8_keyrand25 $A $K --key-select random --key-frac 0.25" >> $J
  echo "$d a8_keylayer10 $A $K --key-select layer --key-frac 0.10" >> $J
  echo "$d own8_keylayer25 --select turn_qk --turn-k 8 --per-layer own $K --key-select layer --key-frac 0.25" >> $J
  echo "$d a8_keylayer25_protect $A $K --key-select layer --key-frac 0.25 $PR --bank-score layer_qk" >> $J
done
CUDA_VISIBLE_DEVICES=0 python scripts/eval_select.py $M --data data/rc_t32_f8.jsonl --limit 2 --select turn_attn --turn-k 8 $K --key-select layer --key-frac 0.25 $PR --bank-score layer_qk > logs/grid5/smoke.log 2>&1
echo "smoke: $(grep -E 'key selection|retention/beyond|Error' logs/grid5/smoke.log | tr '\n' ' ' | cut -c1-260)"; grep -q Error logs/grid5/smoke.log && exit 1
worker() { g=$1; i=0
  while read -r d n args; do
    if [ $((i % 8)) = $g ] && [ ! -s out/grid5/${d}_$n.json ]; then
      CUDA_VISIBLE_DEVICES=$g python scripts/eval_select.py $M --data data/$d.jsonl --limit 40 --out out/grid5/${d}_$n.json $args > logs/grid5/${d}_$n.log 2>&1
      echo "== $d $n | $(grep -E 'turn selection' logs/grid5/${d}_$n.log | sed 's/.*fact turn written in //') | $(grep -E 'key selection' logs/grid5/${d}_$n.log | sed 's/.*: a layer keeps/keeps/' | cut -c1-40) | $(grep -E 'retention/beyond' logs/grid5/${d}_$n.log | awk '{print $2, $3, $4}') $(grep -E 'Traceback|Error' logs/grid5/${d}_$n.log | tail -1 | cut -c1-90)"
    fi
    i=$((i + 1))
  done < $J; }
for g in 0 1 2 3 4 5 6 7; do worker $g & done; wait
echo GRID5_DONE
