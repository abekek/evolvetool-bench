"""Final post-runs step: regenerate tables and figures from latest aggregates,
then swap them into both papers (v2 + kdd2026) and recompile.

Run this AFTER all multi-seed + phase-2 runs are complete.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys


REPO = os.path.dirname(os.path.abspath(__file__))


def run(cmd, cwd=REPO):
    print(f"$ {cmd}")
    r = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  stderr: {r.stderr[:500]}")
        sys.exit(1)
    return r.stdout


def swap_results_table(main_tex_path: str, auto_table_path: str) -> None:
    """Replace the hard-coded Table 3 in main.tex with content of auto_results_table.tex."""
    if not os.path.exists(auto_table_path):
        print(f"  (no auto table at {auto_table_path}; skipping swap)")
        return

    auto = open(auto_table_path).read().strip()
    main = open(main_tex_path).read()

    # Find the existing \begin{table*}...\label{tab:results}...\end{table*} block
    pattern = re.compile(
        r"\\begin\{table\*\}\[t\]\s*\n\\centering\s*\n\\small\s*\n\\begin\{tabular\}.*?"
        r"\\label\{tab:results\}\s*\n\\end\{table\*\}",
        flags=re.DOTALL,
    )
    m = pattern.search(main)
    if not m:
        print(f"  (no tab:results block found in {main_tex_path}; skipping swap)")
        return
    new = main[:m.start()] + auto + main[m.end():]
    with open(main_tex_path, "w") as f:
        f.write(new)
    print(f"  swapped table in {main_tex_path}")


def main():
    run("python3 aggregate_multiseed.py")

    for paper_dir in ["paper/v2", "paper/kdd2026"]:
        swap_results_table(
            os.path.join(REPO, paper_dir, "main.tex"),
            os.path.join(REPO, paper_dir, "auto_results_table.tex"),
        )
        run("pdflatex -interaction=nonstopmode main.tex >/dev/null 2>&1 && "
            "pdflatex -interaction=nonstopmode main.tex >/dev/null 2>&1",
            cwd=os.path.join(REPO, paper_dir))
        info = run("pdfinfo main.pdf | grep Pages", cwd=os.path.join(REPO, paper_dir))
        print(f"  {paper_dir}/main.pdf: {info.strip()}")


if __name__ == "__main__":
    main()
