#!/bin/bash
# Multi-seed CREATOR-style runs (3 more seeds for n=4 total).
set -e
cd "$(dirname "$0")"
export AWS_PROFILE="${AWS_PROFILE:-abekek}"
export PYTHONPATH=src
LOG=run_creator_multiseed.log

for seed in seed1 seed2 seed3; do
  OUT="results_full/creator-style_sonnet_${seed}"
  if [ -f "$OUT/aggregate.json" ]; then
    echo "[$(date +%H:%M:%S)] SKIP $seed (already done)"
    continue
  fi
  echo "[$(date +%H:%M:%S)] >>> CREATOR-style/Sonnet $seed" | tee -a "$LOG"
  python3 run_full_matrix.py 14 "$seed" 2>&1 | tee -a "$LOG"
done

echo "[$(date +%H:%M:%S)] all CREATOR seeds done" | tee -a "$LOG"

# Re-aggregate
python3 aggregate_multiseed.py 2>&1 | tee -a "$LOG"
python3 build_skill_bundles.py 2>&1 | tee -a "$LOG"
