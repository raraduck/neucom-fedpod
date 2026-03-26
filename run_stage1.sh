#!/bin/bash
# Stage 1 local training for all major institutions.
#
# Runs each institution sequentially, then aggregates.
# Output:
#   states/p2_formal/s1/inst{N}/R01r00/models/R01r00_last.pth
#   states/p2_formal/s1/global/R01r01/models/R01r01_agg.pth
#
# Usage:
#   bash run_stage1.sh [-E epochs] [-g gpu] [-W nproc]

# ── defaults ────────────────────────────────────────────────────────────────
EXP="p2_formal"
SPLIT="experiments/partition2/fets_split.csv"
DATA="data/fets128/trainval"
CHAN="[t1,t1ce,t2,flair]"
LGRP="[[1,2,4]]"
LNAM="[wt]"
ROUNDS=1; EPOCHS=60
GPU=1; FREQ=10; NPROC=1; SAVE=1; ALGO="fedavg"

# All 33 institutions in partition2
INSTS=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33)

while getopts "E:g:f:W:a:" opt; do
  case $opt in
    E) EPOCHS="$OPTARG"  ;;
    g) GPU="$OPTARG"     ;;
    f) FREQ="$OPTARG"    ;;
    W) NPROC="$OPTARG"   ;;
    a) ALGO="$OPTARG"    ;;
  esac
done

INIT="states/${EXP}/s1/global/init.pth"
AGG="${EXP}/s1/global"

echo "============================================================"
echo " Stage 1 — Formal FL experiment (partition2)"
echo " Institutions : ${INSTS[*]}"
echo " Epochs       : $EPOCHS"
echo " nproc (DDP)  : $NPROC"
echo " Algorithm    : $ALGO"
echo " Init model   : $INIT"
echo " Data         : $DATA"
echo "============================================================"

if [ ! -f "$INIT" ]; then
  echo "ERROR: init model not found: $INIT"
  echo "Run: bash run_init.sh -A ${EXP}/s1/global -C \"$CHAN\" -G \"$LGRP\""
  exit 1
fi

# ── local training per institution ───────────────────────────────────────────
for INST in "${INSTS[@]}"; do
  JOB="${EXP}/s1/inst${INST}"
  echo ""
  echo ">>> inst${INST}  (job: ${JOB})"
  bash run_train.sh \
    -J "$JOB"    -i "$INST"   -R "$ROUNDS"  -r 0       \
    -E "$EPOCHS" -e 0         -M "$INIT"    -D "$DATA"  \
    -c "$SPLIT"  -C "$CHAN"   -G "$LGRP"    -N "$LNAM"  \
    -f "$FREQ"   -L 1         -g "$GPU"     -W "$NPROC" \
    -s "$SAVE"
done

# ── FedAVG aggregation ────────────────────────────────────────────────────────
echo ""
echo ">>> Aggregation"
JOB_LIST=$(printf "${EXP}/s1/inst%s " "${INSTS[@]}")
bash run_aggregation.sh -j "$JOB_LIST" -A "$AGG" -R "$ROUNDS" -r 0 -a "$ALGO"

echo ""
echo "Stage 1 complete."
echo "  Aggregated model : states/${AGG}/R01r01/models/R01r01_agg.pth"
echo "  Next: bash run_committee_global.sh -m <inst_models> -o states/${EXP}/s1/global/global_priority.json"
