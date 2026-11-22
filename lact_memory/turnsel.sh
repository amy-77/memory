source /data0/qyl/lact-env/env.sh >/dev/null 2>&1
D=/data0/qyl/select_write_20261121; cd $D; export HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=${1:-7}
mkdir -p out/turnsel
run() { # name data limit args...
  n=$1; d=$2; l=$3; shift 3
  python scripts/eval_select.py --model Qwen/Qwen3-0.6B --window 2048 --sinks 4 --memory ckpt/dlg16_new/final --data data/$d.jsonl --limit $l \
    --out out/turnsel/${d}_$n.json "$@" > logs/turnsel_${d}_$n.log 2>&1
  echo "== $d $n"; grep -E "turn selection|retention/(all|beyond)|Traceback|Error" logs/turnsel_${d}_$n.log | cut -c1-110
}
for d in rc_t32_f8 rc_t64_f8; do
  run none $d 40 --select none
  for k in 4 8; do
    run oracle_k$k $d 40 --select turn_oracle --turn-k $k
    run random_k$k $d 40 --select turn_random --turn-k $k
    run qk_k$k $d 40 --select turn_qk --turn-k $k
    run attn_k$k $d 40 --select turn_attn --turn-k $k
  done
done
echo TURNSEL_DONE
