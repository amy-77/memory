#!/usr/bin/bash
# K=1 with the content-word attention scorer. K2_attc lost to K2_bm25 (529 vs 566) although it hit more answer chunks;
# on questions where both hit, 434 vs 465, so the difference comes from the second written chunk. With K=1 a hit
# means only the answer chunk is written, so K1_attc vs K1_bm25 tests that explanation. Starts after the replay rows.
source /data/qyl/lact-insight-env/env.sh >/dev/null 2>&1
W=/data/qyl/memt_locomo_20261122; cd $W; export HF_HUB_OFFLINE=1 TMPDIR=/data/qyl/tmp PYTHONDONTWRITEBYTECODE=1
until grep -q RP2_DONE logs/rp2_run.log; do sleep 60; done
CK=/mnt/airs-business/memory/memt4b_20261120/step_2160
BASE="--model /data/models/Qwen/Qwen3-4B --data data/locomo2_g4.jsonl --attn-impl sdpa --memory $CK --window 2048 --sinks 0 --attn-segment 2048 --mode scored"
SCORES="out/strat/s*.json,out/strat2/s*.json"
for s in 0 1 2 3 4 5 6 7; do
  [ -f out/full/K1_attc.s$s.json ] && continue
  CUDA_VISIBLE_DEVICES=$s python eval_locomo_memt.py $BASE --scores "$SCORES" --strategy "att_c_max[L9-17]" --k 1 \
    --shard $s --nshards 8 --out out/full/K1_attc.s$s.json > logs/fixk_K1_attc_s$s.log 2>&1 &
done
wait
grep -hE "Traceback|Error" logs/fixk_K1_attc_s*.log | head -3
echo "K1_attc $(cat logs/fixk_K1_attc_s*.log | grep -h 'chunks written' | awk '{w+=$3; c+=$5; h+=$11; e+=$13} END {if (e) printf "written %d/%d answer chunks %d/%d (%.3f)", w, c, h, e, h/e}')"
python summary_memt.py out/full > /dev/null
CUDA_VISIBLE_DEVICES=6 PYTHONPATH=$W/pyshim /usr/bin/python3 judge.py out/full --tp 1 --seed-from out/full > logs/judge_k1.log 2>&1
python3 subset_acc.py out/full
python3 hit_acc.py C_write G_random S_qk F_evidence K2_random K1_bm25 K1_attc K2_bm25 K3_bm25 K2_rrf K2_attc
echo K1_DONE
