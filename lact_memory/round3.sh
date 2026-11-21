#!/usr/bin/bash
# Round 3 (step-200 snapshot, 32 turns, 50 dialogues): input-level (lexical) selection, alone and with Robins rehearsal.
source /data0/qyl/lact-env/env.sh >/dev/null 2>&1
D=/data0/qyl/select_write_20261121; cd $D; export HF_HUB_OFFLINE=1
CUDA_VISIBLE_DEVICES= python -m pytest -q tests/test_select.py -x 2>&1 | tail -2
export CKPT=$D/ckpt/early_snapshot OUTDIR=$D/out/early3
LX="--select lexical --pool-mode forward"; PQ="--query-source alone --replay pseudo --probe-src query"
CFG=(
"lex_b10_f8|$LX --boost 10 --pool 8"
"lex_b10_f16|$LX --boost 10 --pool 16"
"lex_b10_f4|$LX --boost 10 --pool 4"
"lex_b5_f8|$LX --boost 5 --pool 8"
"lex_b30_f8|$LX --boost 30 --pool 8"
"lex_b10_c15|--select lexical --pool-mode center --boost 10 --pool 15"
"lex_b10_f8_corpus|$LX --boost 10 --pool 8 --idf data/v8f_train.jsonl"
"lex_b10_f8_r03|$LX --boost 10 --pool 8 --min-rarity 0.3"
"lex_b10_f8_robins1|$LX --boost 10 --pool 8 $PQ --replay-lr-scale 1"
"lex_b10_f8_robins5|$LX --boost 10 --pool 8 $PQ --replay-lr-scale 5"
"lex_b10_f8_robins20|$LX --boost 10 --pool 8 $PQ --replay-lr-scale 20"
"lex_b10_f8_robinsfull5|$LX --boost 10 --pool 8 $PQ --pseudo-target full --replay-lr-scale 5"
"lex_b10_f8_kv|$LX --boost 10 --pool 8 --replay scored_kv --replay-keep 16 --replay-samples 256"
"lex_b10_f8_corpus_robins5|$LX --boost 10 --pool 8 --idf data/v8f_train.jsonl $PQ --replay-lr-scale 5"
"oracle_boost_robins5|--select oracle --oracle-low 1.0 --oracle-high 10 $PQ --replay-lr-scale 5"
"oracle_boost_robins1|--select oracle --oracle-low 1.0 --oracle-high 10 $PQ --replay-lr-scale 1"
)
G=(0 1 2 3 0 1 2 3); i=0
for c in "${CFG[@]}"; do
  bash run_select.sh ${G[$((i % 8))]} r3_${c%%|*} data/rc_t32_f8.jsonl 50 -- ${c#*|} &
  i=$((i + 1)); [ $((i % 8)) = 0 ] && wait
done
wait; echo ROUND3_DONE
