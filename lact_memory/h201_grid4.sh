#!/usr/bin/bash
# h20-1, fourth grid: every layer chooses its own turns with its own q.k (no layer picked with labels, no merging).
source /data/qyl/lact-insight-env/env.sh >/dev/null 2>&1
D=/data/qyl/sw_protect_20261122; cd $D; export HF_HUB_OFFLINE=1 TMPDIR=/data/qyl/tmp; mkdir -p out/grid4 logs/grid4
M="--model /data/models/Qwen/Qwen3-0.6B --window 2048 --sinks 4 --memory ckpt/dlg16_new/final"
J=$D/logs/grid4/jobs.txt; : > $J
for d in rc_t32_f8 rc_t64_f8; do for k in 4 8; do
  Q="--select turn_qk --qk-tokens all --turn-k $k"
  echo "$d own_k$k $Q --per-layer own" >> $J
  echo "$d conf25_k$k $Q --per-layer confident --confident-z 2.5" >> $J
  echo "$d conf35_k$k $Q --per-layer confident --confident-z 3.5" >> $J
done; done
worker() { g=$1; i=0
  while read -r d n args; do
    if [ $((i % 6)) = $g ] && [ ! -s out/grid4/${d}_$n.json ]; then
      CUDA_VISIBLE_DEVICES=$((g + 1)) python scripts/eval_select.py $M --data data/$d.jsonl --limit 40 --out out/grid4/${d}_$n.json $args > logs/grid4/${d}_$n.log 2>&1
      echo "== $d $n | $(grep -E 'turn selection' logs/grid4/${d}_$n.log | sed 's/.*fact turn written in //') | $(grep -E 'per-layer' logs/grid4/${d}_$n.log | cut -c1-110) | $(grep -E 'retention/beyond' logs/grid4/${d}_$n.log | awk '{print $2, $3, $4}') $(grep -E 'Traceback|Error' logs/grid4/${d}_$n.log | tail -1 | cut -c1-90)"
    fi
    i=$((i + 1))
  done < $J; }
CUDA_VISIBLE_DEVICES=1 python scripts/eval_select.py $M --data data/rc_t32_f8.jsonl --limit 2 --select turn_qk --qk-tokens all --turn-k 8 --per-layer confident > logs/grid4/smoke.log 2>&1
echo "smoke: $(grep -E 'per-layer|Error' logs/grid4/smoke.log | tail -1 | cut -c1-120)"; grep -q Error logs/grid4/smoke.log && exit 1
for g in 0 1 2 3 4 5; do worker $g & done; wait
echo GRID4_DONE
