#!/usr/bin/bash
# h20-1, second grid: (a) scoring with all question tokens read in context, (b) replay with more questions
# (100 dialogues) and stronger rehearsal, in the settings where interference is largest.
source /data/qyl/lact-insight-env/env.sh >/dev/null 2>&1
D=/data/qyl/sw_20261122; cd $D; export HF_HUB_OFFLINE=1 TMPDIR=/data/qyl/tmp; mkdir -p out/grid2 logs/grid2
M="--model /data/models/Qwen/Qwen3-0.6B --window 2048 --sinks 4 --memory ckpt/dlg16_new/final"
( CUDA_VISIBLE_DEVICES=0 python scripts/audit_scores.py $M --data data/rc_t16_f8.jsonl --limit 24 --no-causal --out out/grid2/audit_t16.json > logs/grid2/audit_t16.log 2>&1
  CUDA_VISIBLE_DEVICES=0 python scripts/audit_scores.py $M --data data/rc_t32_f8.jsonl --limit 24 --no-causal --out out/grid2/audit_t32.json > logs/grid2/audit_t32.log 2>&1
  for t in 16 32; do echo "== audit t$t"; grep -E "all_layers_z|best layer|control|attention" logs/grid2/audit_t$t.log | cut -c1-80; done ) &
R="--query-source alone --replay pseudo"
J=$D/logs/grid2/jobs.txt; : > $J
for d in rc_t32_f8 rc_t64_f8; do
  for tk in all mean; do for k in 4 8; do echo "$d 40 qk${tk}_k$k --select turn_qk --qk-tokens $tk --turn-k $k" >> $J; done; done
  for k in 16; do A="--select turn_attn --turn-k $k"
    echo "$d 100 a${k}_n100 $A" >> $J
    for sc in 5 20; do
      echo "$d 100 a${k}_ctxq_s${sc}_n100 $A $R --replay-lr-scale $sc --probe-src query --probe-ctx" >> $J
      echo "$d 100 a${k}_ctxqrel8_s${sc}_n100 $A $R --replay-lr-scale $sc --probe-src both --probe-ctx --bank-score relevance --replay-keep 8 --replay-samples 64" >> $J
    done
    echo "$d 100 a${k}_ctxq_full_s5_n100 $A $R --replay-lr-scale 5 --probe-src query --probe-ctx --pseudo-target full" >> $J
  done
done
# every turn written (32 of 32) with and without rehearsal: the plain "does replay protect" test
A="--select turn_attn --turn-k 32"
echo "rc_t32_f8 100 all32_n100 $A" >> $J
for sc in 5 20; do
  echo "rc_t32_f8 100 all32_ctxq_s${sc}_n100 $A $R --replay-lr-scale $sc --probe-src query --probe-ctx" >> $J
  echo "rc_t32_f8 100 all32_ctxqrel8_s${sc}_n100 $A $R --replay-lr-scale $sc --probe-src both --probe-ctx --bank-score relevance --replay-keep 8 --replay-samples 64" >> $J
done
worker() { g=$1; i=0
  while read -r d lim n args; do
    if [ $((i % 8)) = $g ] && [ ! -s out/grid2/${d}_$n.json ]; then
      CUDA_VISIBLE_DEVICES=$g python scripts/eval_select.py $M --data data/$d.jsonl --limit $lim --out out/grid2/${d}_$n.json $args > logs/grid2/${d}_$n.log 2>&1
      echo "== $d $n | $(grep -E 'turn selection' logs/grid2/${d}_$n.log | sed 's/.*fact turn written in //') | $(grep -E 'retention/beyond' logs/grid2/${d}_$n.log | awk '{print $2, $3, $4}') $(grep -E 'Traceback|Error' logs/grid2/${d}_$n.log | tail -1 | cut -c1-90)"
    fi
    i=$((i + 1))
  done < $J; }
for g in 0 1 2 3 4 5 6 7; do worker $g & done; wait
echo GRID2_DONE
