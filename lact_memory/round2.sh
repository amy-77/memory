#!/usr/bin/bash
# Round 2 (step-200 snapshot, 32 turns, 50 dialogues): why does scoring with the whole question hurt?
# Try: only the last question tokens as the query, wider / forward pooling, boost-only weights.
source /data0/qyl/lact-env/env.sh >/dev/null 2>&1
D=/data0/qyl/select_write_20261121; cd $D; export HF_HUB_OFFLINE=1
until grep -q EARLY_ROBINS_DONE logs/early_robins.txt; do sleep 30; done
CUDA_VISIBLE_DEVICES= python -m pytest -q tests/test_select.py -x 2>&1 | tail -2
export CKPT=$D/ckpt/early_snapshot OUTDIR=$D/out/early2
QA="--select query --query-source alone"
CFG=(
"ql1_soft|$QA --query-last 1 --tau 0.5"
"ql2_soft|$QA --query-last 2 --tau 0.5"
"ql4_soft|$QA --query-last 4 --tau 0.5"
"ql2_soft_floor1|$QA --query-last 2 --tau 0.5 --floor 1.0"
"all_soft_floor1|$QA --tau 0.5 --floor 1.0"
"ql2_hard20_c7|$QA --query-last 2 --keep 0.2 --pool 7"
"ql2_hard20_f15|$QA --query-last 2 --keep 0.2 --pool 15 --pool-mode forward"
"ql2_hard40_f15|$QA --query-last 2 --keep 0.4 --pool 15 --pool-mode forward"
"all_hard20_f15|$QA --keep 0.2 --pool 15 --pool-mode forward"
"all_hard20_f31|$QA --keep 0.2 --pool 31 --pool-mode forward"
"all_hard40_f31|$QA --keep 0.4 --pool 31 --pool-mode forward"
"all_hard20_c31|$QA --keep 0.2 --pool 31"
"tail_ql2_soft|--select query --query-source tail --query-last 2 --tau 0.5"
"tail_all_soft|--select query --query-source tail --tau 0.5"
"oracle_boost|--select oracle --oracle-low 1.0 --oracle-high 10"
"oracle_mild|--select oracle --oracle-low 0.5 --oracle-high 3"
)
G=(0 1 2 3 0 1 2 3); i=0
for c in "${CFG[@]}"; do
  bash run_select.sh ${G[$((i % 8))]} r2_${c%%|*} data/rc_t32_f8.jsonl 50 -- ${c#*|} &
  i=$((i + 1)); [ $((i % 8)) = 0 ] && wait
done
wait; echo ROUND2_DONE
