#!/bin/bash
# Usage: bash run_train.sh [options]
# See CLAUDE.md for full parameter descriptions.
#
# DDP (multi-GPU): set NPROC env var or -G flag (default: auto-detect)
#   NPROC=4 bash run_train.sh ...   or   bash run_train.sh -W 4 ...

# ── defaults ────────────────────────────────────────────────────────────────
SEED=42; SAVE=0; FREQ=5; MILE="[20]"; GPU=1
NPROC="${NPROC:-1}"   # number of GPUs for DDP; 1 = single-process
ZOOM=0; FLIP=1
JOB="test_job"
ROUNDS=1; ROUND=0; EPOCHS=30; EPOCH=0
INST=1
SPLIT="experiments/partition2/fets_split.csv"
MODEL="none"
PCT=1.0
DATA="data/fets128/trainval"; DSET="fets"
CHAN="[t1,t1ce,t2,flair]"
LGRP="[[1,2,4]]"; LNAM="[wt]"; LIDX="[1]"
ALGO="fedavg"; MU=0.001
BLOCK="residual"; CHANNELS="[32,64,128,256]"
LR=1e-3; WD=1e-5; LR_GAMMA=0.1
BATCH=1; DEEP_SUP=0; DS_LAYER=1; DROPOUT="None"
NORM="instance"; KSIZE=3

# ── parse flags ─────────────────────────────────────────────────────────────
while getopts "S:s:f:m:g:W:Z:L:J:R:r:E:e:i:c:M:p:D:d:C:G:N:I:a:u:b:l:Q:w:q:B:P:x:n:k:" opt; do
  case $opt in
    S) SEED="$OPTARG"    ;; s) SAVE="$OPTARG"    ;; f) FREQ="$OPTARG"    ;;
    m) MILE="$OPTARG"    ;; g) GPU="$OPTARG"     ;; W) NPROC="$OPTARG"   ;;
    Z) ZOOM="$OPTARG"    ;; L) FLIP="$OPTARG"    ;; J) JOB="$OPTARG"     ;;
    R) ROUNDS="$OPTARG"  ;; r) ROUND="$OPTARG"   ;; E) EPOCHS="$OPTARG"  ;;
    e) EPOCH="$OPTARG"   ;; i) INST="$OPTARG"    ;; c) SPLIT="$OPTARG"   ;;
    M) MODEL="$OPTARG"   ;; p) PCT="$OPTARG"     ;; D) DATA="$OPTARG"    ;;
    d) DSET="$OPTARG"    ;; C) CHAN="$OPTARG"    ;; G) LGRP="$OPTARG"    ;;
    N) LNAM="$OPTARG"    ;; I) LIDX="$OPTARG"    ;; a) ALGO="$OPTARG"    ;;
    u) MU="$OPTARG"      ;; b) BLOCK="$OPTARG"   ;; l) CHANNELS="$OPTARG";;
    Q) LR="$OPTARG"      ;; w) WD="$OPTARG"      ;; q) LR_GAMMA="$OPTARG";;
    B) BATCH="$OPTARG"   ;; P) DEEP_SUP="$OPTARG";; x) DS_LAYER="$OPTARG";;
    n) NORM="$OPTARG"    ;; k) KSIZE="$OPTARG"   ;;
  esac
done

# ── choose launcher ─────────────────────────────────────────────────────────
if [ "$NPROC" -gt 1 ] 2>/dev/null; then
  LAUNCHER="torchrun --nproc_per_node=${NPROC} --standalone"
else
  LAUNCHER="python"
fi

${LAUNCHER} scripts/run_train.py \
  --seed          "$SEED"      \
  --save_infer    "$SAVE"      \
  --eval_freq     "$FREQ"      \
  --milestones    "$MILE"      \
  --use_gpu       "$GPU"       \
  --zoom          "$ZOOM"      \
  --flip_lr       "$FLIP"      \
  --job_name      "$JOB"       \
  --rounds        "$ROUNDS"    \
  --round         "$ROUND"     \
  --epochs        "$EPOCHS"    \
  --epoch         "$EPOCH"     \
  --inst_ids      "$INST"      \
  --cases_split   "$SPLIT"     \
  --weight_path   "$MODEL"     \
  --data_pct      "$PCT"       \
  --data_root     "$DATA"      \
  --dataset       "$DSET"      \
  --input_channels  "$CHAN"    \
  --label_groups    "$LGRP"    \
  --label_names     "$LNAM"    \
  --label_index     "$LIDX"    \
  --algorithm     "$ALGO"      \
  --mu            "$MU"        \
  --block         "$BLOCK"     \
  --channels_list "$CHANNELS"  \
  --lr            "$LR"        \
  --weight_decay  "$WD"        \
  --lr_gamma      "$LR_GAMMA"  \
  --batch_size    "$BATCH"     \
  --deep_supervision "$DEEP_SUP" \
  --ds_layer      "$DS_LAYER"  \
  --norm          "$NORM"      \
  --kernel_size   "$KSIZE"
