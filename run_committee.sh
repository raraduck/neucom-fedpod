#!/bin/bash
# Usage: bash run_committee.sh [options]
# Evaluate training cases with Stage 1 model and produce priority.json

MODEL="none"
INST=1
SPLIT="experiments/partition2/fets_split.csv"
DATA="data/fets128/trainval"
CHAN="[t1,t1ce,t2,flair]"
LGRP="[[1,2,4]]"
BLOCK="residual"
CHANNELS="[32,64,128,256]"
OUTPUT="states/priority.json"
LO=0.3
HI=0.9
GPU=1

while getopts "m:i:c:D:C:G:b:l:o:L:H:g:" opt; do
  case $opt in
    m) MODEL="$OPTARG"   ;; i) INST="$OPTARG"    ;; c) SPLIT="$OPTARG"  ;;
    D) DATA="$OPTARG"    ;; C) CHAN="$OPTARG"     ;; G) LGRP="$OPTARG"   ;;
    b) BLOCK="$OPTARG"   ;; l) CHANNELS="$OPTARG";; o) OUTPUT="$OPTARG"  ;;
    L) LO="$OPTARG"      ;; H) HI="$OPTARG"      ;; g) GPU="$OPTARG"    ;;
  esac
done

python scripts/run_committee.py \
  --model_path     "$MODEL"    \
  --inst_ids       $INST       \
  --cases_split    "$SPLIT"    \
  --data_root      "$DATA"     \
  --input_channels "$CHAN"     \
  --label_groups   "$LGRP"     \
  --block          "$BLOCK"    \
  --channels_list  "$CHANNELS" \
  --output_path    "$OUTPUT"   \
  --priority_low   $LO         \
  --priority_high  $HI         \
  --use_gpu        $GPU
