#!/usr/bin/bash
# Reference runs for one model on the same subsets: full context (truncate-256k) and plain truncation to the
# window (truncate-32k / truncate-8k), no memory multipliers. One GPU. Usage: bash eval_trunc.sh <label> <model dir> <gpu> <port>
source /data0/qyl/livemem_repro_20261121/env.sh
cd $W/LiveMem; unset http_proxy https_proxy MEM_SELECT_DIR; export no_proxy=localhost,127.0.0.1 PY=python
L=$1; M=$2; G=$3; P=$4
CKPT=$M MODE=truncate WINDOW_SIZE=262144 GPUS=$G PORT=$P UTIL=0.85 MAXLEN=262144 nohup bash scripts/serve_livemem.sh > $W/logs/evt_${L}.log 2>&1 &
sp=$!
for t in $(seq 1 80); do grep -qE "Application startup complete|initialization failed" $W/logs/evt_${L}.log && break; sleep 15; done
python -m eval.pipeline --model livemem --model-label $L-ones --servers "{\"truncate\":\"http://127.0.0.1:$P/v1\"}" \
  --judge-url http://127.0.0.1:8790/v1 --datasets ${DATASETS:-longmemeval_s_sub,2wiki_pack,musique_pack,hotpotqa_pack} \
  --profiles truncate-256k,truncate-32k,truncate-8k --limit 120 --out-dir results/eval_sub > $W/logs/evtp_${L}.log 2>&1
for p in $(ps -eo pid,cmd | grep "vllm.entrypoints" | grep -E "port $P( |$)" | grep -v grep | awk '{print $1}'); do kill $p; done
kill $sp 2>/dev/null
for t in $(seq 1 40); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $G); [ "$u" -lt 3000 ] && break; sleep 5; done
echo "EVAL_TRUNC_DONE $L"
