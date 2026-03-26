#!/bin/bash
# End-to-end experiment runner: Stage 1 → Committee → Stage 2 comparison.
#
# Steps:
#   1. Init Stage 1 model  (4ch / 1class WT)
#   2. Stage 1 FL training (9 institutions, FedAVG)
#   3. Committee global scoring → global_priority.json
#   4. Init Stage 2 model  (5ch / 3class WT+TC+ET)
#   5. Stage 2 AL vs RND comparison (10/20/30%)
#
# Usage:
#   bash run_e2e.sh [-E epochs] [-W nproc] [-f eval_freq]
#
# Examples:
#   bash run_e2e.sh -E 5 -W 4 -f 5      # quick smoke-test, 4 GPUs
#   bash run_e2e.sh -E 50 -W 4           # full run, 4 GPUs

# ── defaults ─────────────────────────────────────────────────────────────────
EXP="p2_formal"
S1_EPOCHS=60        # Stage 1: 1 round × S1_EPOCHS  = total
S2_ROUNDS=20        # Stage 2: S2_ROUNDS × S2_EPOCHS = total (= S1_EPOCHS)
NPROC=1
FREQ=10
ALGO="fedavg"

while getopts "E:R:W:f:a:" opt; do
  case $opt in
    E) S1_EPOCHS="$OPTARG" ;;
    R) S2_ROUNDS="$OPTARG" ;;
    W) NPROC="$OPTARG"     ;;
    f) FREQ="$OPTARG"      ;;
    a) ALGO="$OPTARG"      ;;
  esac
done

# Stage 2 epochs per round = Stage 1 total epochs / Stage 2 rounds
S2_EPOCHS=$(python3 -c "import math; print(math.ceil($S1_EPOCHS / $S2_ROUNDS))")

S1_INIT="states/${EXP}/s1/global/init.pth"
S2_INIT="states/${EXP}/s2/global/init.pth"
PRIORITY="states/${EXP}/s1/global/global_priority.json"

echo "============================================================"
echo " E2E experiment : $EXP"
echo " Stage 1        : 1 round × $S1_EPOCHS epochs"
echo " Stage 2        : $S2_ROUNDS rounds × $S2_EPOCHS epochs  (total=$((S2_ROUNDS * S2_EPOCHS)))"
echo " nproc (DDP)    : $NPROC"
echo " Eval freq      : every $FREQ epochs"
echo " Algorithm      : $ALGO"
echo "============================================================"

# ── Step 1: Init Stage 1 model ───────────────────────────────────────────────
echo ""
echo "━━━ [1/5] Stage 1 init ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [ -f "$S1_INIT" ]; then
  echo "  Already exists, skipping: $S1_INIT"
else
  bash run_init.sh \
    -A "${EXP}/s1/global" \
    -C "[t1,t1ce,t2,flair]" \
    -G "[[1,2,4]]"
fi

# ── Step 2: Stage 1 FL training ──────────────────────────────────────────────
echo ""
echo "━━━ [2/5] Stage 1 FL training ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
bash run_stage1.sh -E "$S1_EPOCHS" -W "$NPROC" -f "$FREQ" -a "$ALGO"

# ── Step 3: Committee global scoring ─────────────────────────────────────────
echo ""
echo "━━━ [3/5] Committee global scoring ━━━━━━━━━━━━━━━━━━━━━━━━━"
bash run_committee_stage1.sh

# ── Step 4: Init Stage 2 model ───────────────────────────────────────────────
echo ""
echo "━━━ [4/5] Stage 2 init ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [ -f "$S2_INIT" ]; then
  echo "  Already exists, skipping: $S2_INIT"
else
  bash run_init.sh \
    -A "${EXP}/s2/global" \
    -C "[t1,t1ce,t2,flair,seg]" \
    -G "[[1,2,4],[1,4],[4]]"
fi

# ── Step 5: Stage 2 AL vs RND comparison ─────────────────────────────────────
echo ""
echo "━━━ [5/5] Stage 2 AL vs RND comparison ━━━━━━━━━━━━━━━━━━━━━"
bash run_stage2_compare.sh -E "$S2_EPOCHS" -R "$S2_ROUNDS" -W "$NPROC" -f "$FREQ" -a "$ALGO"

# ── Done ─────────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " E2E complete."
echo " Results → results/${EXP}_stage2_compare.txt"
echo "============================================================"
