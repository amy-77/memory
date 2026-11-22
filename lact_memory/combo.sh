source /data0/qyl/lact-env/env.sh >/dev/null 2>&1
D=/data0/qyl/select_write_20261121; cd $D; export HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=${1:-7}
mkdir -p out/combo
run() { n=$1; d=$2; shift 2
  python scripts/eval_select.py --model Qwen/Qwen3-0.6B --window 2048 --sinks 4 --memory ckpt/dlg16_new/final --data data/$d.jsonl --limit 40 \
    --out out/combo/${d}_$n.json "$@" > logs/combo_${d}_$n.log 2>&1
  echo "== $d $n"; grep -E "turn selection|retention/(all|beyond)|Traceback|Error" logs/combo_${d}_$n.log | cut -c1-110; }
PQ="--query-source alone --replay pseudo --probe-src query"
for d in rc_t32_f8 rc_t64_f8; do
  for k in 8 16; do
    [ $k = 16 ] && run attn_k$k $d --select turn_attn --turn-k $k
    for sc in 1 5; do
      run attn_k${k}_robins_s$sc $d --select turn_attn --turn-k $k $PQ --replay-lr-scale $sc
    done
    run attn_k${k}_robins_both_s5 $d --select turn_attn --turn-k $k --query-source alone --replay pseudo --probe-src both --replay-lr-scale 5
  done
  run attn_k4_robins_s5 $d --select turn_attn --turn-k 4 $PQ --replay-lr-scale 5
done
echo COMBO_DONE
