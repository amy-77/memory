#!/usr/bin/bash
# replaces K2_attmax (attention with all question tokens) by K2_attc (content words only, the better scorer)
source /data/qyl/lact-insight-env/env.sh >/dev/null 2>&1
W=/data/qyl/memt_locomo_20261122; cd $W; export HF_HUB_OFFLINE=1 TMPDIR=/data/qyl/tmp PYTHONDONTWRITEBYTECODE=1
for p in $(pgrep -f "eval_locomo_memt.py.*K2_attmax"); do kill $p; done
sleep 5; rm -f out/full/K2_attmax.s*.json
CK=/mnt/airs-business/memory/memt4b_20261120/step_2160
BASE="--model /data/models/Qwen/Qwen3-4B --data data/locomo2_g4.jsonl --attn-impl sdpa --memory $CK --window 2048 --sinks 0 --attn-segment 2048 --mode scored"
SCORES="out/strat/s*.json,out/strat2/s*.json"
for s in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$s python eval_locomo_memt.py $BASE --scores "$SCORES" --strategy "att_c_max[L9-17]" --k 2 \
    --shard $s --nshards 8 --out out/full/K2_attc.s$s.json > logs/fixk_K2_attc_s$s.log 2>&1 &
done
wait
grep -hE "Traceback|Error" logs/fixk_K2_attc_s*.log | head -3
echo "K2_attc $(cat logs/fixk_K2_attc_s*.log | grep -h 'chunks written' | awk '{w+=$3; c+=$5; h+=$11; e+=$13} END {if (e) printf "written %d/%d answer chunks %d/%d (%.3f)", w, c, h, e, h/e}')"
until grep -q FIXK_DONE logs/fixk_run2.log; do sleep 60; done
python summary_memt.py out/full > /dev/null
CUDA_VISIBLE_DEVICES=6 PYTHONPATH=$W/pyshim /usr/bin/python3 judge.py out/full --tp 1 --seed-from out/full > logs/judge_attc.log 2>&1
python3 subset_acc.py out/full
echo ATTC_DONE
