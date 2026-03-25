#!/bin/bash
# Aggregation launcher for fedpod-new.
# Reads _last.pth from each local job, aggregates, and saves _agg.pth.
#
# Usage:
#   bash run_aggregation.sh \
#     -j "inst01_job inst02_job inst03_job" \
#     -A global_job \
#     -R 3 -r 0 \
#     -a fedpod
#
# Output:
#   states/{agg_name}/R{R:02}r{r+1:02}/models/R{R:02}r{r+1:02}_agg.pth

# ── defaults ────────────────────────────────────────────────────────────────
JOB_NAMES=""         # space-separated list of local job names  (-j)
AGG_NAME="global"    # aggregated output job name               (-A)
ROUNDS=3             # total FL rounds                          (-R)
ROUND=0              # current round being aggregated           (-r)
ALGO="fedavg"        # algorithm                                (-a)
STATES="states"      # states root directory                    (-s)

# ── parse flags ─────────────────────────────────────────────────────────────
while getopts "j:A:R:r:a:s:" opt; do
  case $opt in
    j) JOB_NAMES="$OPTARG" ;;
    A) AGG_NAME="$OPTARG"  ;;
    R) ROUNDS="$OPTARG"    ;;
    r) ROUND="$OPTARG"     ;;
    a) ALGO="$OPTARG"      ;;
    s) STATES="$OPTARG"    ;;
  esac
done

python scripts/run_aggregation.py \
  --job_names   ${JOB_NAMES}  \
  --agg_name    "${AGG_NAME}" \
  --rounds      "${ROUNDS}"   \
  --round       "${ROUND}"    \
  --algorithm   "${ALGO}"     \
  --states_root "${STATES}"
