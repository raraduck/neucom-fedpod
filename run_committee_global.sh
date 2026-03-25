#!/bin/bash
# Usage: bash run_committee_global.sh [options]
# Global committee evaluation: pool all institutions' scores → global_priority.json
#
# -m  inst_models   "1:path1.pth,2:path2.pth"   (required)
# -c  cases_split   CSV path
# -D  data_root
# -C  input_channels
# -G  label_groups
# -b  block         residual|plain
# -l  channels_list
# -o  output_path   (required)
# -g  use_gpu       0|1

INST_MODELS=""
SPLIT="experiments/partition2/fets_split.csv"
DATA="data/fets128/trainval"
CHAN="[t1,t1ce,t2,flair]"
LGRP="[[1,2,4]]"
BLOCK="residual"
CHANNELS="[32,64,128,256]"
OUTPUT=""
GPU=1

while getopts "m:c:D:C:G:b:l:o:g:" opt; do
  case $opt in
    m) INST_MODELS="$OPTARG" ;; c) SPLIT="$OPTARG"    ;; D) DATA="$OPTARG"    ;;
    C) CHAN="$OPTARG"        ;; G) LGRP="$OPTARG"     ;; b) BLOCK="$OPTARG"   ;;
    l) CHANNELS="$OPTARG"   ;; o) OUTPUT="$OPTARG"   ;; g) GPU="$OPTARG"     ;;
  esac
done

if [ -z "$INST_MODELS" ] || [ -z "$OUTPUT" ]; then
  echo "Error: -m (inst_models) and -o (output_path) are required."
  echo "Example:"
  echo "  bash run_committee_global.sh \\"
  echo "    -m \"1:states/s1/inst1/R01r00/models/R01r00_last.pth,2:states/s1/inst2/R01r00/models/R01r00_last.pth\" \\"
  echo "    -o states/stage1_local/global_priority.json"
  exit 1
fi

python scripts/run_committee_global.py \
  --inst_models    "$INST_MODELS"  \
  --cases_split    "$SPLIT"        \
  --data_root      "$DATA"         \
  --input_channels "$CHAN"         \
  --label_groups   "$LGRP"         \
  --block          "$BLOCK"        \
  --channels_list  "$CHANNELS"     \
  --output_path    "$OUTPUT"       \
  --use_gpu        "$GPU"
