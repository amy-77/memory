#!/usr/bin/bash
# Evaluate one model on the subsets, one write policy at a time (two servers per policy: state-32k and state-8k)
# on ONE GPU, then the no-memory references. Usage: bash eval_model.sh <label> <model dir> <gpu> <base port>
# POLICIES (default "ones oracle_0") names directories under $W/masks. SKIP_TRUNC=1 skips the references.
source /data0/qyl/livemem_repro_20261121/env.sh
cd $W/LiveMem; unset http_proxy https_proxy; export no_proxy=localhost,127.0.0.1 PY=python
L=$1; M=$2; G=$3; P=$4
DS=${DATASETS:-longmemeval_s_sub,2wiki_pack,musique_pack,hotpotqa_pack}
fail=0; j=0
for pol in ${POLICIES:-ones oracle_0}; do
  pids=(); i=0
  for win in 32768 8192; do
    rm -f $W/logs/ev_${L}_${pol}_$win.log
    MEM_SELECT_DIR=$W/masks/$pol CKPT=$M MODE=state WINDOW_SIZE=$win GPUS=$G PORT=$((P + j + i)) UTIL=0.42 MAXLEN=131072 \
      nohup bash scripts/serve_livemem.sh > $W/logs/ev_${L}_${pol}_$win.log 2>&1 &
    pids+=($!); i=$((i + 1)); sleep 60   # stagger: each instance sizes its memory from what is free at start
  done
  for t in $(seq 1 80); do
    n=$(cat $W/logs/ev_${L}_${pol}_*.log | grep -c "Application startup complete")
    [ "$n" -ge 2 ] && break; grep -q "initialization failed" $W/logs/ev_${L}_${pol}_*.log && break; sleep 15
  done
  echo "$L $pol servers up: $n/2"
  if [ "$n" -ge 2 ]; then
    python -m eval.pipeline --model livemem --model-label $L-$pol \
      --servers "{\"state-32k\":\"http://127.0.0.1:$((P + j))/v1\",\"state-8k\":\"http://127.0.0.1:$((P + j + 1))/v1\"}" \
      --judge-url http://127.0.0.1:8790/v1 --datasets $DS \
      --profiles state-32k,state-8k --limit 120 --out-dir results/eval_sub > $W/logs/evp_${L}_$pol.log 2>&1
    echo "EVAL_POLICY_DONE $L $pol gen-fail=$(grep -c gen-fail $W/logs/evp_${L}_$pol.log)"
  else
    echo "EVAL_POLICY_FAILED $L $pol: servers did not start"; fail=1
  fi
  for p in $(ps -eo pid,cmd | grep "vllm.entrypoints" | grep -E "port ($((P + j))|$((P + j + 1)))( |$)" | grep -v grep | awk '{print $1}'); do kill $p; done
  kill "${pids[@]}" 2>/dev/null
  for t in $(seq 1 40); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $G); [ "$u" -lt 3000 ] && break; sleep 5; done
  j=$((j + 2))
done
[ -n "$SKIP_TRUNC" ] || bash $W/sel/eval_trunc.sh $L $M $G $((P + 9))
[ $fail = 0 ] && echo "EVAL_MODEL_DONE $L" || echo "EVAL_MODEL_FAILED $L"
