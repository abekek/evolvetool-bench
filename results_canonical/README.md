# results_canonical/

Canonical result manifests for EvolveTool-Bench.

Every table and figure in the paper is reproducible from:
- run_manifest.jsonl   — one record per benchmark run (system, model, seed, sessions)
- tool_manifest.jsonl  — one record per generated tool (implementation hash, quality scores)
- aggregate.json       — per-system aggregate statistics

Generate tables: python scripts/make_tables.py
Generate figures: python scripts/make_figures.py
Audit tasks:     python scripts/audit_tasks.py
Audit results:   python scripts/audit_results.py
