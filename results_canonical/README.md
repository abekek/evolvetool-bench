# results_canonical/

Canonical result manifests for EvolveTool-Bench, built from the preserved
per-seed runs in `results_full/` by `scripts/build_canonical.py`. ETS is the
safety-free composite (`0.30*TC + 0.20*TQS + 0.10*max(0,1-RC) + 0.40*LH`).

## Files

- **`run_manifest.jsonl`** — one record per *session run* (system, model, seed,
  session_id, task_completion, task_completion_by_type, mean_tool_quality,
  reuse/redundancy/composition/regression rates, library_health, total_llm_calls,
  recomputed `evolvetool_score`).
- **`tool_manifest.jsonl`** — one record per *preserved tool* (name,
  created_at_task, 16-char source SHA-256, self-validation scores, and
  capability-aligned hidden-test scores from the re-evaluation).
- **`aggregate.json`** — per-(system, model) mean±std over the main (non-variant)
  seed runs; these are the numbers in the paper's headline table.

## Granularity note

The original runs persisted **session-level** summaries and per-tool artifacts,
not per-task traces. `run_manifest` rows are therefore session-level (with
per-type completion). Full per-task trace logging (per-task hashes, tools-used
lists, per-test outcomes) is a planned harness enhancement; the schema here
documents what is reproducible from disk today.

## Regenerate everything

```bash
python scripts/build_canonical.py --source results_full --output results_canonical
python scripts/audit_results.py  --results-dir results_canonical
python scripts/make_tables.py    --results-dir results_canonical --output paper/kdd_eval2026
python scripts/make_figures.py   --results-dir results_canonical --output paper/kdd_eval2026
```

`make_tables.py` reproduces the headline ETS/TC/TQS/Reuse/LH numbers in
`paper/kdd_eval2026/main.tex` Table~2 (e.g. No-Evol 0.439±0.007,
CREATOR-style 0.576±0.030, Code-Evol 0.543±0.014).

## Audit gates

```bash
python scripts/audit_tasks.py                       # verifier coverage (51/99 today)
python scripts/audit_tasks.py --check-capabilities  # every non-seed task has capability_id
python scripts/audit_tasks.py --check-hidden-tests  # every gap task has hidden tests
```
