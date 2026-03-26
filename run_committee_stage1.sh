#!/bin/bash
# Committee evaluation after Stage 1.
# Automatically collects inst model paths from states/p2_formal/s1/inst*/
# and runs global committee uncertainty scoring.
#
# Usage:
#   bash run_committee_stage1.sh [-k score_key] [-g gpu]

EXP="p2_formal"
SPLIT="experiments/partition2/fets_split.csv"
DATA="data/fets128/trainval"
CHAN="[t1,t1ce,t2,flair]"
LGRP="[[1,2,4]]"
SCORE_KEY="texture_bald"
DIVERSITY_LAMBDA=0.5
GPU=1
ROUNDS=1

while getopts "k:L:g:" opt; do
  case $opt in
    k) SCORE_KEY="$OPTARG"        ;;
    L) DIVERSITY_LAMBDA="$OPTARG" ;;
    g) GPU="$OPTARG"              ;;
  esac
done

OUTPUT="states/${EXP}/s1/global/global_priority.json"

# ── collect inst model paths automatically ───────────────────────────────────
INST_MODELS=""
for INST_DIR in states/${EXP}/s1/inst*/; do
  [ -d "$INST_DIR" ] || continue
  INST_ID=$(basename "$INST_DIR" | sed 's/inst//')
  # find last.pth for round 0
  PTH=$(find "$INST_DIR" -name "*_last.pth" | sort | tail -1)
  if [ -n "$PTH" ]; then
    [ -n "$INST_MODELS" ] && INST_MODELS="${INST_MODELS},"
    INST_MODELS="${INST_MODELS}${INST_ID}:${PTH}"
  fi
done

if [ -z "$INST_MODELS" ]; then
  echo "ERROR: No Stage 1 models found under states/${EXP}/s1/inst*/"
  echo "Run bash run_stage1.sh first."
  exit 1
fi

echo "============================================================"
echo " Committee global evaluation"
echo " Experiment : $EXP"
echo " Score key  : $SCORE_KEY"
echo " Output     : $OUTPUT"
echo " Models found:"
echo "$INST_MODELS" | tr ',' '\n' | sed 's/^/   /'
echo "============================================================"

bash run_committee_global.sh \
  -m "$INST_MODELS"      \
  -c "$SPLIT"            \
  -D "$DATA"             \
  -C "$CHAN"             \
  -G "$LGRP"             \
  -o "$OUTPUT"           \
  -k "$SCORE_KEY"        \
  -L "$DIVERSITY_LAMBDA" \
  -g "$GPU"

echo ""
echo "Global priority saved → $OUTPUT"
