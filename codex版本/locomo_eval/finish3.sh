#!/usr/bin/bash
# A_full is compute-bound (two 22K-token prefills per question), so sharing one GPU does not help.
# 12 shards: 0-5 now on GPUs 6 and 7 (3 each), 6-11 on GPUs 0-5 as soon as S_qk has finished there.
W=/data/qyl/memt_locomo_20261122; cd $W
ev() { (source /data/qyl/lact-insight-env/env.sh >/dev/null 2>&1; cd $W; export HF_HUB_OFFLINE=1 TMPDIR=/data/qyl/tmp
        CUDA_VISIBLE_DEVICES=$1 python eval_locomo_memt.py --model /data/models/Qwen/Qwen3-4B --data data/locomo2_g4.jsonl \
          --attn-impl sdpa --mode full --memory none --window 0 --shard $2 --nshards 12 --out out/full/A_full.s$2.json \
          > logs/full_A_full_s$2.log 2>&1); }
n() { ls out/full/$1.s*.json 2>/dev/null | wc -l; }
for s in 0 1 2; do ev 6 $s & done
for s in 3 4 5; do ev 7 $s & done
until [ $(n S_qk) = 6 ]; do sleep 60; done
G=(0 1 2 3 4 5); for i in 0 1 2 3 4 5; do ev ${G[$i]} $((i + 6)) & done
until [ $(n A_full) = 12 ]; do sleep 60; done
wait
echo FULL_DONE >> logs/full.log
for m in B_last C_write W_wrong F_evidence G_random S_qk A_full; do echo "$m shards $(n $m)"; done
(source /data/qyl/lact-insight-env/env.sh >/dev/null 2>&1; cd $W; python summary_memt.py out/full)
CUDA_VISIBLE_DEVICES=6 PYTHONPATH=$W/pyshim HF_HUB_OFFLINE=1 TMPDIR=/data/qyl/tmp /usr/bin/python3 judge.py out/full --tp 1 > logs/judge_full.log 2>&1
python3 subset_acc.py out/full
echo ALL_DONE
