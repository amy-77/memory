#!/usr/bin/bash
# Robins-style rehearsal (probes = the question's own memory queries, 16 per head; the new chunk's (k, v) and the
# rehearsal pairs (q~ -> y~ read before the write) go into ONE delta update), on top of: every chunk written (C),
# the best label-free selection (K2_attc), and the oracle selection (F). Starts after the fixed-budget runs.
source /data/qyl/lact-insight-env/env.sh >/dev/null 2>&1
W=/data/qyl/memt_locomo_20261122; cd $W; export HF_HUB_OFFLINE=1 TMPDIR=/data/qyl/tmp PYTHONDONTWRITEBYTECODE=1
until grep -q ATTC_DONE logs/attc_run.log; do sleep 60; done
CK=/mnt/airs-business/memory/memt4b_20261120/step_2160
BASE="--model /data/models/Qwen/Qwen3-4B --data data/locomo2_g4.jsonl --attn-impl sdpa --memory $CK --window 2048 --sinks 0 --attn-segment 2048"
SCORES="out/strat/s*.json,out/strat2/s*.json"
run() { n=$1; shift
  for s in 0 1 2 3 4 5 6 7; do
    [ -f out/full/$n.s$s.json ] && continue
    CUDA_VISIBLE_DEVICES=$s python eval_locomo_memt.py $BASE "$@" --shard $s --nshards 8 --out out/full/$n.s$s.json > logs/rp_${n}_s$s.log 2>&1 &
  done; }
run C_write_R16 --mode write --replay-k 16 --replay-alpha 1.0
run C_write_R16a05 --mode write --replay-k 16 --replay-alpha 0.5
run K2_attc_R16 --mode scored --scores "$SCORES" --strategy "att_c_max[L9-17]" --k 2 --replay-k 16 --replay-alpha 1.0
run F_evidence_R16 --mode evidence --replay-k 16 --replay-alpha 1.0
wait
grep -hE "Traceback|Error" logs/rp_*_s*.log | head -3
for n in C_write_R16 C_write_R16a05 K2_attc_R16 F_evidence_R16; do echo "$n $(grep -h '^replay' logs/rp_${n}_s0.log)"; done
python summary_memt.py out/full > /dev/null
CUDA_VISIBLE_DEVICES=6 PYTHONPATH=$W/pyshim /usr/bin/python3 judge.py out/full --tp 1 --seed-from out/full > logs/judge_rp2.log 2>&1
python3 subset_acc.py out/full
echo RP2_DONE
