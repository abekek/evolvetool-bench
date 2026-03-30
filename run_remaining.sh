#!/bin/bash
# Run remaining benchmark configurations sequentially
# (must be sequential due to mock server port conflict)

set -e
cd "$(dirname "$0")"
export AWS_PROFILE=abekek
export PYTHONPATH=src

echo "=== Starting remaining benchmark runs ==="

# Run 1: No-Evolution / Sonnet (if not done)
if [ ! -f results_full/no-evolution_sonnet/aggregate.json ]; then
  echo ">>> Run 1: No-Evolution / Sonnet"
  python run_full_matrix.py 1
fi

# Run 3: EvoSkill / Sonnet (if not done)
if [ ! -f results_full/evoskill_sonnet/aggregate.json ]; then
  echo ">>> Run 3: EvoSkill / Sonnet"
  python run_full_matrix.py 3
fi

# Run 4: OneShot / Sonnet (if not done)
if [ ! -f results_full/oneshot_sonnet/aggregate.json ]; then
  echo ">>> Run 4: OneShot / Sonnet"
  python run_full_matrix.py 4
fi

# Run 5: No-Evolution / Haiku
if [ ! -f results_full/no-evolution_haiku/aggregate.json ]; then
  echo ">>> Run 5: No-Evolution / Haiku"
  python run_full_matrix.py 5
fi

# Run 6: ARISE / Haiku
if [ ! -f results_full/arise_haiku/aggregate.json ]; then
  echo ">>> Run 6: ARISE / Haiku"
  python run_full_matrix.py 6
fi

echo "=== All runs complete ==="

# Generate summary
python3 -c "
import json, os, glob
print('\n' + '='*80)
print('FULL BENCHMARK RESULTS')
print('='*80)
print(f'{\"System\":25s} {\"Model\":8s} {\"ETS\":>6s} {\"TC\":>6s} {\"Tools\":>6s} {\"Reuse\":>6s} {\"LH\":>6s}')
print('-'*80)
for f in sorted(glob.glob('results_full/*/aggregate.json')):
    with open(f) as fh:
        d = json.load(fh)
    print(f'{d[\"system\"]:25s} {d[\"model\"]:8s} {d[\"avg_evolvetool_score\"]:6.3f} {d[\"avg_task_completion\"]:6.3f} {d[\"total_tools\"]:6d} {d[\"avg_reuse_rate\"]:6.3f} {d[\"avg_library_health\"]:6.3f}')
"
