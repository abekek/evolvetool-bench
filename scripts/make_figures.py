#!/usr/bin/env python3
"""Generate figures from canonical results (stub — implement per figure).

Usage:
    python scripts/make_figures.py --results-dir results_canonical/ --output paper/kdd_eval2026/
"""
from __future__ import annotations
import argparse, pathlib

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="results_canonical")
    p.add_argument("--output", default="paper/kdd_eval2026")
    args = p.parse_args(argv)
    print(f"make_figures: reading {args.results_dir}, writing to {args.output}")
    print("  (stub — implement generate_figures.py logic here)")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
