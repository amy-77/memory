#!/usr/bin/bash
# LiveMem-RL + selective writing on LongMemEval-s: one state-32k server per policy (GPUs 0-5), judge on 6,7.
source /data0/qyl/livemem_repro_20261121/env.sh
cd $W/LiveMem; unset http_proxy https_proxy; export no_proxy=localhost,127.0.0.1 PY=python
until grep -qE "MASKS_DONE" $W/logs/setup_sel.log; do sleep 20; done
POL=(ones oracle_0 oracle_0.3 bm25_10_0 bm25_5_0 random_10_0)
for i in "${!POL[@]}"; do
  p=${POL[$i]}; rm -f $W/masks/$p/_server.log
  MEM_SELECT_DIR=$W/masks/$p CKPT=$W/models/LiveMem-RL-vllm MODE=state WINDOW_SIZE=32768 GPUS=$i PORT=$((8900 + i)) \
    nohup bash scripts/serve_livemem.sh > $W/logs/sel_serve_$p.log 2>&1 &
done
MODEL=$W/models/Qwen3.6-35B-A3B GPUS=6,7 TP=2 PORT=8790 nohup bash scripts/serve_judge.sh > $W/logs/sel_serve_judge.log 2>&1 &
for t in $(seq 1 90); do
  n=$(cat $W/logs/sel_serve_*.log 2>/dev/null | grep -c "Application startup complete"); [ "$n" -ge 7 ] && break; sleep 20
done
echo "servers up: $n / 7"
for i in "${!POL[@]}"; do
  p=${POL[$i]}
  python -m eval.pipeline --model livemem --model-label sel-$p --servers "{\"state-32k\":\"http://127.0.0.1:$((8900 + i))/v1\"}" \
    --judge-url http://127.0.0.1:8790/v1 --datasets longmemeval_s --profiles state-32k --out-dir results/eval_sel ${LIMIT:+--limit $LIMIT} \
    > $W/logs/sel_eval_$p.log 2>&1 &
done
wait_pids=$(jobs -p | tail -6)
wait $wait_pids
echo SEL_EVAL_DONE
