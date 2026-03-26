#!/bin/bash
# Usage: bash run_committee_global.sh [options]
# Global committee uncertainty evaluation → global_priority.json
#
# All K institution models evaluate ALL cases (cross-institution).
# Uncertainty is computed from K probability maps per case:
#   BALD MI  : epistemic (inter-model) uncertainty  [default, recommended]
#   variance : prediction variance across models
#   mean_dice: committee mean Dice vs GT (requires GT labels)
#
# Options:
#   -m  inst_models   "1:path1.pth,2:path2.pth"   (required)
#   -c  cases_split   CSV path
#   -D  data_root
#   -C  input_channels
#   -G  label_groups
#   -b  block         residual|plain
#   -l  channels_list
#   -o  output_path   (required)
#   -k  score_key       bald_mi|roi_bald|variance|mean_dice|texture_bald  (default: texture_bald)
#   -L  diversity_lambda  texture diversity weight for texture_bald (default: 0.5)
#   -g  use_gpu         0|1

INST_MODELS=""
SPLIT="experiments/partition2/fets_split.csv"
DATA="data/fets128/trainval"
CHAN="[t1,t1ce,t2,flair]"
LGRP="[[1,2,4]]"
BLOCK="residual"
CHANNELS="[32,64,128,256]"
OUTPUT=""
SCORE_KEY="texture_bald"
DIVERSITY_LAMBDA=0.5
GPU=1

while getopts "m:c:D:C:G:b:l:o:k:L:g:" opt; do
  case $opt in
    m) INST_MODELS="$OPTARG"      ;; c) SPLIT="$OPTARG"    ;; D) DATA="$OPTARG"    ;;
    C) CHAN="$OPTARG"             ;; G) LGRP="$OPTARG"     ;; b) BLOCK="$OPTARG"   ;;
    l) CHANNELS="$OPTARG"        ;; o) OUTPUT="$OPTARG"   ;; k) SCORE_KEY="$OPTARG" ;;
    L) DIVERSITY_LAMBDA="$OPTARG" ;; g) GPU="$OPTARG"      ;;
  esac
done

if [ -z "$INST_MODELS" ] || [ -z "$OUTPUT" ]; then
  echo "Error: -m (inst_models) and -o (output_path) are required."
  echo "Example:"
  echo "  bash run_committee_global.sh \\"
  echo "    -m \"1:states/s1/inst1/R01r00/models/R01r00_last.pth,2:states/s1/inst2/R01r00/models/R01r00_last.pth\" \\"
  echo "    -o states/stage1_local/global_priority.json \\"
  echo "    -k bald_mi"
  exit 1
fi

python scripts/run_committee_global.py \
  --inst_models      "$INST_MODELS"      \
  --cases_split      "$SPLIT"            \
  --data_root        "$DATA"             \
  --input_channels   "$CHAN"             \
  --label_groups     "$LGRP"             \
  --block            "$BLOCK"            \
  --channels_list    "$CHANNELS"         \
  --output_path      "$OUTPUT"           \
  --score_key        "$SCORE_KEY"        \
  --diversity_lambda "$DIVERSITY_LAMBDA" \
  --use_gpu          "$GPU"
