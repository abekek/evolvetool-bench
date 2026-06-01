#!/bin/bash
# Re-run Code-Evol/Sonnet 3 more seeds with tool preservation enabled.
# Combined with the existing arise_sonnet_v2 single-seed run, this gives n=4
# preserved-tool Code-Evol seeds for apples-to-apples comparison against
# CREATOR-style under the expanded hidden test suite.

set -e
cd "$(dirname "$0")"
export AWS_PROFILE="${AWS_PROFILE:-abekek}"
export PYTHONPATH=src

LOG=run_codeevol_preserved.log

for seed in v2_seed1 v2_seed2 v2_seed3; do
  OUT="results_full/arise_sonnet_${seed}"
  if [ -f "$OUT/aggregate.json" ]; then
    echo "[$(date +%H:%M:%S)] SKIP $seed (already done)"
    continue
  fi
  echo "[$(date +%H:%M:%S)] >>> Code-Evol/Sonnet ${seed}" | tee -a "$LOG"
  python3 run_full_matrix.py 2 "$seed" 2>&1 | tee -a "$LOG"
done

echo "[$(date +%H:%M:%S)] all preserved Code-Evol seeds done" | tee -a "$LOG"

# Re-evaluate everything against expanded hidden tests, then recompute LH and aggregate.
python3 reeval_with_expanded_tests.py 2>&1 | tee -a "$LOG"
python3 recompute_lh_v3.py        2>&1 | tee -a "$LOG"
python3 aggregate_multiseed.py    2>&1 | tee -a "$LOG"
