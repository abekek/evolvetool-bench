#!/bin/bash
# Multi-seed runs for No-Evolution, One-Shot, and Strategy-Only on Sonnet.
# Addresses Reviewer 5Wxk W3 / R3: 13pp LH spread needs n>=3 baselines to assess significance.
#
# Each run is ~225 LLM calls, ~$7-12 cost on Sonnet 4 via Bedrock, ~15min wall time.
# Total: 9 runs, est $60-120, est ~2.5 hours sequential.
#
# Usage: ./run_multiseed.sh [seed1|seed2|seed3|all]
#   no arg or "all" = run all 3 seeds
#   seed1/seed2/seed3 = run just that seed across all 3 configs

set -e
cd "$(dirname "$0")"
export AWS_PROFILE="${AWS_PROFILE:-abekek}"
export PYTHONPATH=src

WHICH="${1:-all}"
LOG=run_multiseed.log

# run_one <run_id> <seed> <label>
run_one() {
  local rid=$1 seed=$2 label=$3
  local outdir="results_full/${label}_${seed}"
  if [ -f "$outdir/aggregate.json" ]; then
    echo "[$(date +%H:%M:%S)] SKIP: $outdir already done"
    return
  fi
  echo "[$(date +%H:%M:%S)] >>> ${label} ${seed}  (run_full_matrix.py ${rid} ${seed})"
  python run_full_matrix.py "$rid" "$seed" 2>&1 | tee -a "$LOG"
}

# Map: rid 1=no-evolution/sonnet, 3=evoskill(StrategyOnly)/sonnet, 4=oneshot/sonnet
run_seed() {
  local seed=$1
  echo "===== SEED ${seed} ====="  | tee -a "$LOG"
  run_one 1 "$seed" "no-evolution_sonnet"
  run_one 4 "$seed" "oneshot_sonnet"
  run_one 3 "$seed" "evoskill_sonnet"
}

case "$WHICH" in
  seed1) run_seed seed1 ;;
  seed2) run_seed seed2 ;;
  seed3) run_seed seed3 ;;
  all)
    run_seed seed1
    run_seed seed2
    run_seed seed3
    ;;
  *)
    echo "Usage: $0 [seed1|seed2|seed3|all]"
    exit 1
    ;;
esac

echo "[$(date +%H:%M:%S)] All multi-seed runs complete." | tee -a "$LOG"

# Print aggregate summary
python3 <<'EOF' | tee -a "$LOG"
import json, glob, os, statistics as st

configs = {
  "No-Evol":   "no-evolution_sonnet",
  "One-Shot":  "oneshot_sonnet",
  "Str-Only":  "evoskill_sonnet",
}
print()
print(f"{'config':12s} {'seed':10s} {'TC':>5s} {'TQS':>5s} {'LH':>5s} {'ETS':>5s}")
print("-"*55)
for name, base in configs.items():
    for tag in ["", "_seed1", "_seed2", "_seed3"]:
        p = f"results_full/{base}{tag}/aggregate.json"
        if not os.path.exists(p): continue
        with open(p) as f: a = json.load(f)
        seed_label = tag.lstrip("_") or "orig"
        print(f"{name:12s} {seed_label:10s} {a['avg_task_completion']:.3f} {a['avg_tool_quality']:.3f} {a['avg_library_health']:.3f} {a['avg_evolvetool_score']:.3f}")
    # also print mean and std
    vals = {}
    for tag in ["", "_seed1", "_seed2", "_seed3"]:
        p = f"results_full/{base}{tag}/aggregate.json"
        if not os.path.exists(p): continue
        with open(p) as f: a = json.load(f)
        for k in ["avg_task_completion","avg_library_health","avg_evolvetool_score"]:
            vals.setdefault(k, []).append(a[k])
    if "avg_library_health" in vals and len(vals["avg_library_health"]) > 1:
        m = st.mean(vals["avg_library_health"]); s = st.pstdev(vals["avg_library_health"])
        print(f"{name:12s} {'(mean±sd)':10s}  -    -    {m:.3f}±{s:.3f}  -")
    print()
EOF
