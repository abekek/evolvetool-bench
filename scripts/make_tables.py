#!/usr/bin/env python3
"""Generate LaTeX tables from canonical results manifest.

Usage:
    python scripts/make_tables.py --results-dir results_canonical/ --output paper/kdd_eval2026/
"""
from __future__ import annotations
import json, pathlib, sys, argparse
from collections import defaultdict

REPO_ROOT = pathlib.Path(__file__).parent.parent


def load_results(results_dir: pathlib.Path) -> list[dict]:
    records = []
    for f in sorted(results_dir.glob("**/*.json")):
        try:
            data = json.loads(f.read_text())
            if isinstance(data, list):
                records.extend(data)
            elif isinstance(data, dict) and "session_id" in data:
                records.append(data)
        except Exception:
            pass
    return records


def fmt(v: float) -> str:
    return f"{v*100:.1f}"


def make_headline_table(records: list[dict]) -> str:
    by_system: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_system[r.get("system", "unknown")].append(r)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Headline results across systems (mean over sessions). TC = task completion rate, CRR = correct reuse rate, IRR = incorrect reuse rate, LH = library health.}",
        r"\label{tab:headline}",
        r"\small",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"\textbf{System} & \textbf{TC (\%)} & \textbf{CRR (\%)} & \textbf{IRR (\%)} & \textbf{LH (\%)} \\",
        r"\midrule",
    ]

    for system, recs in sorted(by_system.items()):
        n = len(recs)
        tc = sum(r.get("task_completion", 0) for r in recs) / n
        crr = sum(r.get("correct_reuse_rate", 0) for r in recs) / n
        irr = sum(r.get("incorrect_reuse_rate", 0) for r in recs) / n
        lh = sum(r.get("library_health", 0) for r in recs) / n
        lines.append(f"{system} & {fmt(tc)} & {fmt(crr)} & {fmt(irr)} & {fmt(lh)} \\\\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="results_canonical")
    p.add_argument("--output", default="paper/kdd_eval2026")
    args = p.parse_args(argv)

    results_dir = pathlib.Path(args.results_dir)
    output_dir = pathlib.Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_results(results_dir)
    if not records:
        print(f"No result records found in {results_dir}. Tables will be empty.")
        records = []

    table = make_headline_table(records)
    out_path = output_dir / "auto_results_table.tex"
    out_path.write_text(table)
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
