#!/usr/bin/bash
# Evaluate one model on the subsets with two write policies (all tokens / oracle sessions), on ONE GPU.
# Usage: bash eval_model.sh <label> <model dir> <gpu> <base port>
source /data0/qyl/livemem_repro_20261121/env.sh
cd $W/LiveMem; unset http_proxy https_proxy; export no_proxy=localhost,127.0.0.1 PY=python
L=$1; M=$2; G=$3; P=$4
pids=()
i=0
for pol in ones oracle_0; do
  for win in 32768 8192; do
    MEM_SELECT_DIR=$W/masks/$pol CKPT=$M MODE=state WINDOW_SIZE=$win GPUS=$G PORT=$((P + i)) UTIL=0.22 MAXLEN=131072 \
      nohup bash scripts/serve_livemem.sh > $W/logs/ev_${L}_${pol}_$win.log 2>&1 &
    pids+=($!); i=$((i + 1)); sleep 50   # stagger: each instance sizes its memory from what is free at start
  done
done
for t in $(seq 1 80); do n=$(cat $W/logs/ev_${L}_*.log | grep -c "Application startup complete"); [ "$n" -ge 4 ] && break; sleep 15; done
echo "$L servers up: $n/4"
j=0
for pol in ones oracle_0; do
  python -m eval.pipeline --model livemem --model-label $L-$pol \
    --servers "{\"state-32k\":\"http://127.0.0.1:$((P + j))/v1\",\"state-8k\":\"http://127.0.0.1:$((P + j + 1))/v1\"}" \
    --judge-url http://127.0.0.1:8790/v1 --datasets longmemeval_s_sub,2wiki_single,musique_single,hotpotqa_single \
    --profiles state-32k,state-8k --limit 200 --out-dir results/eval_sub > $W/logs/evp_${L}_$pol.log 2>&1 &
  j=$((j + 2))
done
wait $(jobs -p | tail -2)
for p in $(ps -eo pid,cmd | grep "vllm.entrypoints" | grep -E "port ($P|$((P+1))|$((P+2))|$((P+3)))( |$)" | grep -v grep | awk '{print $1}'); do kill $p; done
kill "${pids[@]}" 2>/dev/null
echo "EVAL_MODEL_DONE $L"
