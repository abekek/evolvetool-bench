#!/bin/bash
# Phase 2: after multi-seed completes, run:
#   - 1 Code-Evol seed with tool preservation (so we can re-evaluate Code-Evol's tools with v2 Q)
#   - 1 ToolMaker-style baseline seed on Sonnet (the new full-strength baseline)
#
# Cost estimate: ~$40-60 total (Code-Evol ~$25, ToolMaker-style ~$15-25 since it iterates).
# Wall time: ~1.5 hours sequential.

set -e
cd "$(dirname "$0")"
export AWS_PROFILE="${AWS_PROFILE:-abekek}"
export PYTHONPATH=src

LOG=run_phase2.log

# Wait for multi-seed to finish if still running
while ps aux | grep -v grep | grep -q "run_multiseed.sh\|run_full_matrix.py.*seed[123]"; do
  echo "[$(date +%H:%M:%S)] waiting for multi-seed to finish..."
  sleep 60
done

echo "[$(date +%H:%M:%S)] multi-seed done, starting phase 2" | tee -a "$LOG"

# 1. Code-Evol with tool preservation (only adds 1 new dir; existing arise_sonnet_v2 doesn't exist yet)
if [ ! -f results_full/arise_sonnet_v2/aggregate.json ]; then
  echo "[$(date +%H:%M:%S)] >>> Code-Evol Sonnet (v2 with preserved tools)" | tee -a "$LOG"
  python run_full_matrix.py 2 v2 2>&1 | tee -a "$LOG"
fi

# 2. ToolMaker-style on Sonnet
if [ ! -f results_full/toolmaker-style_sonnet/aggregate.json ]; then
  echo "[$(date +%H:%M:%S)] >>> ToolMaker-style Sonnet" | tee -a "$LOG"
  python run_full_matrix.py 12 2>&1 | tee -a "$LOG"
fi

echo "[$(date +%H:%M:%S)] phase 2 complete" | tee -a "$LOG"

# Summary
python3 <<'EOF' | tee -a "$LOG"
import json, os
print()
print(f"{'config':28s} {'TC':>5s} {'TQS':>5s} {'LH':>5s} {'ETS':>5s}")
print("-"*55)
for d in sorted(os.listdir("results_full")):
    agg = f"results_full/{d}/aggregate.json"
    if not os.path.exists(agg): continue
    with open(agg) as f: a = json.load(f)
    print(f"{d:28s} {a['avg_task_completion']:.3f} {a['avg_tool_quality']:.3f} {a['avg_library_health']:.3f} {a['avg_evolvetool_score']:.3f}")
EOF
