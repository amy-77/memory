#!/usr/bin/bash
# rerun of the protected-write jobs of grid 3 (they failed on a dtype error under autocast; fixed)
source /data/qyl/lact-insight-env/env.sh >/dev/null 2>&1
D=/data/qyl/sw_protect_20261122; cd $D; export HF_HUB_OFFLINE=1 TMPDIR=/data/qyl/tmp
M="--model /data/models/Qwen/Qwen3-0.6B --window 2048 --sinks 4 --memory ckpt/dlg16_new/final"
grep protect logs/grid3/jobs.txt > logs/grid3/jobs_b.txt
# quick check on GPU with autocast before the long runs
CUDA_VISIBLE_DEVICES=1 python scripts/eval_select.py $M --data data/rc_t32_f8.jsonl --limit 2 --select turn_attn --turn-k 8 --query-source alone --replay pseudo --probe-src bank --replay-lr-scale 0 --protect --replay-samples 512 --bank-score relevance --replay-keep 8 > logs/grid3/smoke.log 2>&1
echo "smoke: $(grep -E 'retention/beyond|Error' logs/grid3/smoke.log | tail -1 | cut -c1-100)"
grep -q "Error" logs/grid3/smoke.log && exit 1
worker() { g=$1; i=0
  while read -r d n args; do
    if [ $((i % 8)) = $g ] && [ ! -s out/grid3/${d}_$n.json ]; then
      CUDA_VISIBLE_DEVICES=$g python scripts/eval_select.py $M --data data/$d.jsonl --limit 100 --out out/grid3/${d}_$n.json $args > logs/grid3/${d}_$n.log 2>&1
      echo "== $d $n | $(grep -E 'turn selection' logs/grid3/${d}_$n.log | sed 's/.*fact turn written in //') | $(grep -E 'retention/beyond' logs/grid3/${d}_$n.log | awk '{print $2, $3, $4}') $(grep -E 'Traceback|Error' logs/grid3/${d}_$n.log | tail -1 | cut -c1-90)"
    fi
    i=$((i + 1))
  done < logs/grid3/jobs_b.txt; }
for g in 0 1 2 3 4 5 6 7; do worker $g & done; wait
echo GRID3B_DONE
