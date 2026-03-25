#!/bin/bash
# Init launcher for fedpod-new.
# Creates a randomly initialized shared model for FL round 0.
#
# Usage:
#   bash run_init.sh \
#     -A fedpod_p2_250325/global \
#     -b residual \
#     -l "[32,64,128,256]" \
#     -C "[t1,t1ce,t2,flair]" \
#     -G "[[1,2,4]]"
#
# Output:
#   states/{agg_name}/init.pth

# ── defaults ────────────────────────────────────────────────────────────────
AGG_NAME="global"
STATES="states"
BLOCK="residual"
CHANNELS="[32,64,128,256]"
CHAN="[t1,t1ce,t2,flair]"
LGRP="[[1,2,4]]"
SEED=0

# ── parse flags ─────────────────────────────────────────────────────────────
while getopts "A:s:b:l:C:G:S:" opt; do
  case $opt in
    A) AGG_NAME="$OPTARG" ;;
    s) STATES="$OPTARG"   ;;
    b) BLOCK="$OPTARG"    ;;
    l) CHANNELS="$OPTARG" ;;
    C) CHAN="$OPTARG"     ;;
    G) LGRP="$OPTARG"    ;;
    S) SEED="$OPTARG"     ;;
  esac
done

python scripts/run_init.py \
  --agg_name       "${AGG_NAME}" \
  --states_root    "${STATES}"   \
  --block          "${BLOCK}"    \
  --channels_list  "${CHANNELS}" \
  --input_channels "${CHAN}"     \
  --label_groups   "${LGRP}"    \
  --seed           "${SEED}"
