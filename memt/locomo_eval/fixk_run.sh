#!/usr/bin/bash
# fixed-budget, label-free write strategies on MemT-Qwen3-4B (scores from score_strategies.py), two waves of 3 modes
source /data/qyl/lact-insight-env/env.sh >/dev/null 2>&1
W=/data/qyl/memt_locomo_20261122; cd $W; export HF_HUB_OFFLINE=1 TMPDIR=/data/qyl/tmp PYTHONDONTWRITEBYTECODE=1
CK=/mnt/airs-business/memory/memt4b_20261120/step_2160
BASE="--model /data/models/Qwen/Qwen3-4B --data data/locomo2_g4.jsonl --attn-impl sdpa --memory $CK --window 2048 --sinks 0 --attn-segment 2048 --mode scored"
SCORES="out/strat/s*.json"   # passed quoted: the shell must not expand it
NS=8
wave() {
  for spec in "$@"; do
    IFS=: read n strat k <<< "$spec"
    for s in $(seq 0 $((NS-1))); do
      [ -f out/full/$n.s$s.json ] && continue
      CUDA_VISIBLE_DEVICES=$s python eval_locomo_memt.py $BASE --scores "$SCORES" --strategy "$strat" --k $k --shard $s --nshards $NS \
        --out out/full/$n.s$s.json > logs/fixk_${n}_s$s.log 2>&1 &
    done
  done
  wait
}
wave "K2_bm25:bm25:2" "K2_random:random:2" "K2_rrf:rrf:2"
wave "K1_bm25:bm25:1" "K3_bm25:bm25:3" "K2_attmax:att_max[L9-17]:2"
grep -hE "Traceback|Error" logs/fixk_*_s*.log | head -5
for n in K1_bm25 K2_bm25 K3_bm25 K2_rrf K2_attmax K2_random; do echo "$n $(cat logs/fixk_${n}_s*.log | grep -h 'chunks written' | awk '{w+=$3; c+=$5; h+=$11; e+=$13} END {if (e) printf "written %d/%d evidence %d/%d (%.3f)", w, c, h, e, h/e}')"; done
python summary_memt.py out/full > /dev/null
CUDA_VISIBLE_DEVICES=6 PYTHONPATH=$W/pyshim /usr/bin/python3 judge.py out/full --tp 1 --seed-from out/full > logs/judge_fixk.log 2>&1
python3 subset_acc.py out/full
echo FIXK_DONE
