#!/usr/bin/bash
# h20-1: selective writing + replay grid on the re-trained 0.6B dialogue model, 8 GPUs in parallel.
source /data/qyl/lact-insight-env/env.sh >/dev/null 2>&1
D=/data/qyl/sw_20261122; cd $D; export HF_HUB_OFFLINE=1 TMPDIR=/data/qyl/tmp; mkdir -p out/grid logs/grid
until grep -q H201_TRAIN_DONE logs/h201_train.log; do sleep 30; done
[ -f ckpt/dlg16_new/final/memory_weights.pt ] || { echo "no checkpoint"; exit 1; }
R="--query-source alone --replay pseudo --replay-lr-scale 5"
J=$D/logs/grid/jobs.txt; : > $J
for d in rc_t32_f8 rc_t64_f8; do
  echo "$d none --select none" >> $J
  for k in 4 8 16; do echo "$d attn_k$k --select turn_attn --turn-k $k" >> $J; done
  echo "$d oracle_k8 --select turn_oracle --turn-k 8" >> $J
  echo "$d qk_k8 --select turn_qk --turn-k 8" >> $J
  for k in 8 16; do A="--select turn_attn --turn-k $k"
    echo "$d a${k}_robins_alone $A $R --probe-src query" >> $J
    echo "$d a${k}_bank_lr $A $R --probe-src bank --replay-keep 32 --replay-samples 256" >> $J
    echo "$d a${k}_bank_rel32 $A $R --probe-src bank --bank-score relevance --replay-keep 32 --replay-samples 256" >> $J
    echo "$d a${k}_bank_rel8 $A $R --probe-src bank --bank-score relevance --replay-keep 8 --replay-samples 64" >> $J
    echo "$d a${k}_ctxq $A $R --probe-src query --probe-ctx" >> $J
    echo "$d a${k}_ctxq_rel8 $A $R --probe-src both --probe-ctx --bank-score relevance --replay-keep 8 --replay-samples 64" >> $J
  done
  Q="--select turn_qk --turn-k 8"
  echo "$d q8_bank_rel8 $Q $R --probe-src bank --bank-score relevance --replay-keep 8 --replay-samples 64" >> $J
  echo "$d q8_ctxq $Q $R --probe-src query --probe-ctx" >> $J
  echo "$d q8_ctxq_rel8 $Q $R --probe-src both --probe-ctx --bank-score relevance --replay-keep 8 --replay-samples 64" >> $J
done
worker() { g=$1; i=0
  while read -r d n args; do
    if [ $((i % 8)) = $g ] && [ ! -s out/grid/${d}_$n.json ]; then
      CUDA_VISIBLE_DEVICES=$g python scripts/eval_select.py --model /data/models/Qwen/Qwen3-0.6B --window 2048 --sinks 4 --memory ckpt/dlg16_new/final \
        --data data/$d.jsonl --limit 40 --out out/grid/${d}_$n.json $args > logs/grid/${d}_$n.log 2>&1
      echo "== $d $n | $(grep -E 'turn selection' logs/grid/${d}_$n.log | sed 's/.*fact turn written in //') | $(grep -E 'retention/beyond' logs/grid/${d}_$n.log | awk '{print $2, $3, $4}') $(grep -E 'Traceback|Error' logs/grid/${d}_$n.log | tail -1 | cut -c1-90)"
    fi
    i=$((i + 1))
  done < $J; }
for g in 0 1 2 3 4 5 6 7; do worker $g & done; wait
echo GRID_DONE
