#!/usr/bin/bash
# Selective-write / rehearsal variants on one checkpoint and one test set, spread over GPUs.
# Usage: CKPT=... OUTDIR=... bash batch_select.sh <data.jsonl> <prefix> <limit> "<gpu list>" [only-regex]
R=$(dirname "$(readlink -f "$0")"); cd $R
DATA=$1; P=$2; N=$3; read -r -a G <<< "$4"; ONLY=${5:-.}
QA="--query-source alone"; PQ="$QA --replay pseudo --probe-src query"
LX="--select lexical --pool-mode forward --boost 10 --pool 8"
KV="--replay scored_kv --replay-keep 16 --replay-samples 256"
CFG=(
"none|--select none"
"oracle|--select oracle"
"oracle_boost|--select oracle --oracle-low 1.0 --oracle-high 10"
"lex|$LX"
"lex_corpus|$LX --idf data/v8f_train.jsonl"
"robins_s1|--select none $PQ --replay-lr-scale 1"
"robins_s5|--select none $PQ --replay-lr-scale 5"
"lex_robins_s1|$LX $PQ --replay-lr-scale 1"
"lex_robins_s5|$LX $PQ --replay-lr-scale 5"
"lex_corpus_robins_s5|$LX --idf data/v8f_train.jsonl $PQ --replay-lr-scale 5"
"oracle_boost_robins_s5|--select oracle --oracle-low 1.0 --oracle-high 10 $PQ --replay-lr-scale 5"
"kv_only|--select none $KV"
"lex_kv|$LX $KV"
"qk_soft|--select query $QA --tau 0.5"
"qk_hard20|--select query $QA --keep 0.2 --pool 7"
"qk_soft_robins_s5|--select query $QA --tau 0.5 --replay pseudo --probe-src query --replay-lr-scale 5"
)
i=0
for c in "${CFG[@]}"; do
  tag=${c%%|*}; a=${c#*|}
  [[ $tag =~ $ONLY ]] || continue
  g=${G[$((i % ${#G[@]}))]}
  bash run_select.sh $g ${P}_$tag $DATA $N -- $a &
  i=$((i + 1)); [ $((i % ${#G[@]})) = 0 ] && wait
done
wait
