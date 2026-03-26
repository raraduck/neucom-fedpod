#!/bin/bash
# Stage 2 AL vs Random comparison experiment.
#
# Conditions: {AL, RND} × {10%, 20%, 30%} = 6 conditions
# Institutions: inst1, inst2, inst3 (3 largest, ~163 cases each)
#
# Prerequisites:
#   1. states/p2_formal/s1/global/global_priority.json (from run_committee_global.sh)
#   2. states/p2_formal/s2/global/init.pth (5ch/3class)
#
# Usage:
#   bash run_stage2_compare.sh [-E epochs] [-g gpu] [-W nproc]
#
# Output:
#   states/p2_formal/s2_{al,rnd}{10,20,30}/inst{N}/  (local models)
#   states/p2_formal/s2_{al,rnd}{10,20,30}/global/   (aggregated)
#   results/p2_formal_stage2_compare.txt              (summary)

# ── defaults ────────────────────────────────────────────────────────────────
EXP="p2_formal"
SPLIT="experiments/partition2/fets_split.csv"
DATA="data/fets128/trainval"
CHAN="[t1,t1ce,t2,flair,seg]"
MASK="[seg]"
LGRP="[[1,2,4],[1,4],[4]]"
LNAM="[wt,tc,et]"
LIDX="[1,2,4]"
ROUNDS=20; EPOCHS=3
GPU=1; FREQ=1; NPROC=1; SAVE=1; ALGO="fedavg"

# Institutions for Stage 2 comparison (3 largest, equal size)
INSTS=(1 2 3)
PCTS=(10 20 30)   # select percentages

while getopts "E:R:g:f:W:a:" opt; do
  case $opt in
    E) EPOCHS="$OPTARG"  ;;
    R) ROUNDS="$OPTARG"  ;;
    g) GPU="$OPTARG"     ;;
    f) FREQ="$OPTARG"    ;;
    W) NPROC="$OPTARG"   ;;
    a) ALGO="$OPTARG"    ;;
  esac
done

INIT="states/${EXP}/s2/global/init.pth"
PRIORITY="states/${EXP}/s1/global/global_priority.json"

echo "============================================================"
echo " Stage 2 AL vs RND comparison (partition2)"
echo " Institutions : ${INSTS[*]}"
echo " Percentages  : ${PCTS[*]}%"
echo " Rounds       : $ROUNDS"
echo " Epochs/round : $EPOCHS"
echo " Total epochs : $((ROUNDS * EPOCHS))  (= T_max for cosine)"
echo " nproc (DDP)  : $NPROC"
echo " Algorithm    : $ALGO"
echo " Init model   : $INIT"
echo " Priority     : $PRIORITY"
echo "============================================================"

# ── validation ───────────────────────────────────────────────────────────────
if [ ! -f "$INIT" ]; then
  echo "ERROR: Stage 2 init not found: $INIT"
  echo "Run: bash run_init.sh -A ${EXP}/s2/global -C \"$CHAN\" -G \"$LGRP\""
  exit 1
fi

if [ ! -f "$PRIORITY" ]; then
  echo "ERROR: global_priority.json not found: $PRIORITY"
  echo "Run run_committee_global.sh first."
  exit 1
fi

# ── helper: run one condition (multi-round FL loop) ───────────────────────────
run_condition() {
  local TAG=$1 MODE=$2 PCT_FLOAT=$3 PPATH=$4
  echo ""
  echo "========================================"
  echo " Condition: $TAG  (mode=$MODE  pct=$PCT_FLOAT)"
  echo "========================================"

  local ROUND_MODEL="$INIT"

  for (( R=0; R<ROUNDS; R++ )); do
    # epoch_start / epoch_end keep the cosine curve continuous across rounds:
    #   round 0: train epoch 1 → EPOCHS
    #   round 1: train epoch EPOCHS+1 → 2*EPOCHS
    #   ...
    local EPOCH_START=$(( R * EPOCHS ))
    local EPOCH_END=$(( (R + 1) * EPOCHS ))
    echo ""
    echo "  -- Round $R / $(( ROUNDS - 1 )) --"

    for INST in "${INSTS[@]}"; do
      JOB="${EXP}/${TAG}/inst${INST}"
      echo "--- inst${INST} ---"
      bash run_train.sh \
        -J "$JOB"    -i "$INST"   -R "$ROUNDS"     -r "$R"          \
        -E "$EPOCH_END" -e "$EPOCH_START" -M "$ROUND_MODEL" -D "$DATA" \
        -c "$SPLIT"  -C "$CHAN"   -G "$LGRP"        -N "$LNAM"       \
        -X "$MASK"   -Y "$MODE"   -y "$PCT_FLOAT"   -I "$LIDX"        \
        ${PPATH:+-T "$PPATH"}                                         \
        -f "$FREQ"   -L 1         -g "$GPU"         -W "$NPROC"      \
        -s "$SAVE"
    done

    JOB_LIST=$(printf "${EXP}/${TAG}/inst%s " "${INSTS[@]}")
    echo "--- Aggregation: $TAG round $R ---"
    bash run_aggregation.sh \
      -j "$JOB_LIST" -A "${EXP}/${TAG}/global" -R "$ROUNDS" -r "$R" -a "$ALGO"

    # next round uses the freshly aggregated model
    ROUND_MODEL=$(printf "states/${EXP}/${TAG}/global/R%02dr%02d/models/R%02dr%02d_agg.pth" \
                  "$ROUNDS" "$(( R + 1 ))" "$ROUNDS" "$(( R + 1 ))")
  done
}

# ── run all 6 conditions ──────────────────────────────────────────────────────
for PCT in "${PCTS[@]}"; do
  PCT_FLOAT=$(python3 -c "print($PCT / 100)")
  run_condition "s2_al${PCT}"  committee "$PCT_FLOAT" "$PRIORITY"
  run_condition "s2_rnd${PCT}" random    "$PCT_FLOAT" ""
done

# ── summary ───────────────────────────────────────────────────────────────────
mkdir -p results
python - "$EXP" "${INSTS[@]}" << 'PYEOF'
import sys, torch, glob

exp    = sys.argv[1]
insts  = [int(x) for x in sys.argv[2:]]
pcts   = [10, 20, 30]
labels = ['wt', 'tc', 'et']

def load_post(job, inst):
    pat = f'states/{job}/inst{inst}/R*/models/*_last.pth'
    ps  = sorted(glob.glob(pat))
    if not ps:
        return None
    ck = torch.load(ps[-1], map_location='cpu')
    pm = ck.get('post_metrics', {})
    return {
        'dice':    pm.get('dice', float('nan')),
        'classes': pm.get('dice_per_class', [float('nan')] * 3),
        'P':       ck.get('P', 0),
    }

out_lines = []
header = f"{'Condition':14s}  {'avg_dice':>9}" + ''.join(f'{l:>8}' for l in labels) + \
         ''.join(f'  inst{i}' for i in insts)
out_lines.append('=' * (len(header) + 4))
out_lines.append(f" Stage 2 AL vs RND — {exp}")
out_lines.append('=' * (len(header) + 4))
out_lines.append(header)
out_lines.append('-' * (len(header) + 4))

for pct in pcts:
    for mode, tag in [('AL',  f's2_al{pct}'),
                      ('RND', f's2_rnd{pct}')]:
        cond  = f'{mode}-{pct}%'
        dices = []
        cls_sum = [0.0] * 3
        inst_dices = []
        for inst in insts:
            m = load_post(f'{exp}/{tag}', inst)
            if m:
                dices.append(m['dice'])
                inst_dices.append(f'{m["dice"]:.4f}')
                for i, v in enumerate(m['classes']):
                    cls_sum[i] += v / len(insts)
            else:
                inst_dices.append('  N/A')
        avg = sum(dices) / len(dices) if dices else float('nan')
        cl_str = ''.join(f'{v:8.4f}' for v in cls_sum)
        id_str = '  '.join(inst_dices)
        out_lines.append(f"{cond:14s}  {avg:9.4f}{cl_str}  {id_str}")
    out_lines.append('')

result = '\n'.join(out_lines)
print(result)
with open(f'results/{exp}_stage2_compare.txt', 'w') as f:
    f.write(result + '\n')
print(f"\nSaved → results/{exp}_stage2_compare.txt")
PYEOF
