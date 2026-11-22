source /data0/qyl/lact-env/env.sh >/dev/null 2>&1
D=/data0/qyl/select_write_20261121; cd $D; export HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=${1:-7}
until grep -q COMBO_DONE logs/combo.log; do sleep 60; done
run() { n=$1; d=$2; shift 2
  python scripts/eval_select.py --model Qwen/Qwen3-0.6B --window 2048 --sinks 4 --memory ckpt/dlg16_new/final --data data/$d.jsonl --limit 40 \
    --out out/combo/${d}_$n.json "$@" > logs/combo_${d}_$n.log 2>&1
  echo "== $d $n"; grep -E "turn selection|retention/(all|beyond)|Traceback|Error" logs/combo_${d}_$n.log | cut -c1-110; }
R="--query-source alone --replay pseudo --replay-lr-scale 5"
for d in rc_t64_f8 rc_t32_f8; do
  k=8; A="--select turn_attn --turn-k $k"
  run a${k}_bank_lr      $d $A $R --probe-src bank --replay-keep 32 --replay-samples 256
  run a${k}_bank_rel32   $d $A $R --probe-src bank --bank-score relevance --replay-keep 32 --replay-samples 256
  run a${k}_bank_rel8    $d $A $R --probe-src bank --bank-score relevance --replay-keep 8 --replay-samples 64
  run a${k}_ctxq         $d $A $R --probe-src query --probe-ctx
  run a${k}_ctxq_rel8    $d $A $R --probe-src both --probe-ctx --bank-score relevance --replay-keep 8 --replay-samples 64
done
echo COMBO2_DONE
