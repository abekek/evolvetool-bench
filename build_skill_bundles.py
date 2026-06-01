"""Post-run step: walk preserved tools and construct + evaluate skill bundles.

For every preserved tool under results_full/<config>/tools/<session>/<name>.py,
this builds the corresponding skill bundle (SKILL.md + metadata.json + tests),
evaluates the bundle, and writes:

  results_full/<config>/skills/<session>/<name>/
    SKILL.md
    function.py
    tests.py
    metadata.json
    scores.json   <- new: bundle_quality_score + sub-scores

Also produces a per-config aggregate of bundle-level metrics:
  results_full/<config>/skills_aggregate.json
"""

from __future__ import annotations

import json
import os
import glob
from typing import Iterable

import sys
sys.path.insert(0, "src")

from evolvetool_bench.types import ToolRecord
from evolvetool_bench.evaluation.skill_bundle import (
    wrap_as_skill_bundle, evaluate_skill_bundle, write_skill_bundle_to_disk
)


def process_config(config_dir: str) -> dict | None:
    tools_root = os.path.join(config_dir, "tools")
    if not os.path.isdir(tools_root):
        return None

    bundles_summary = []
    for meta_path in sorted(glob.glob(os.path.join(tools_root, "*", "*.meta.json"))):
        session = os.path.basename(os.path.dirname(meta_path))
        with open(meta_path) as f:
            meta = json.load(f)
        source_path = meta_path.replace(".meta.json", ".py")
        tests_path = meta_path.replace(".meta.json", "_tests.py")
        if not os.path.exists(source_path):
            continue
        source = open(source_path).read()
        test_suite = open(tests_path).read() if os.path.exists(tests_path) else ""

        tool = ToolRecord(
            name=meta["name"],
            implementation=source,
            test_suite=test_suite,
            created_at_task=meta.get("created_at_task", ""),
            version=meta.get("version", 1),
            correctness=meta["scores"]["correctness"],
            robustness=meta["scores"]["robustness"],
            generality=meta["scores"]["generality"],
            code_quality=meta["scores"]["code_quality"],
        )
        bundle = wrap_as_skill_bundle(tool, task_id=meta.get("created_at_task", ""))
        bundle = evaluate_skill_bundle(bundle)

        out_dir = os.path.join(config_dir, "skills", session, bundle.name)
        write_skill_bundle_to_disk(bundle, out_dir)
        scores = {
            "bundle_quality": bundle.bundle_quality_score,
            "structure": bundle.structure_score,
            "doc": bundle.doc_score,
            "metadata": bundle.metadata_score,
            "underlying_tool_tqs": tool.quality_score,
        }
        with open(os.path.join(out_dir, "scores.json"), "w") as f:
            json.dump(scores, f, indent=2)
        bundles_summary.append({
            "name": bundle.name,
            "session": session,
            **scores,
        })

    if not bundles_summary:
        return None

    n = len(bundles_summary)
    agg = {
        "n_bundles": n,
        "avg_bundle_quality": sum(b["bundle_quality"] for b in bundles_summary) / n,
        "avg_structure": sum(b["structure"] for b in bundles_summary) / n,
        "avg_doc": sum(b["doc"] for b in bundles_summary) / n,
        "avg_metadata": sum(b["metadata"] for b in bundles_summary) / n,
        "avg_tool_tqs": sum(b["underlying_tool_tqs"] for b in bundles_summary) / n,
        "fraction_passing_skill_gate":  # bundle_quality >= 0.5
            sum(1 for b in bundles_summary if b["bundle_quality"] >= 0.5) / n,
        "bundles": bundles_summary,
    }
    with open(os.path.join(config_dir, "skills_aggregate.json"), "w") as f:
        json.dump(agg, f, indent=2)
    return agg


def main():
    print(f"{'config':32s} {'n':>3s} {'bundle Q':>9s} {'struct':>7s} {'doc':>5s} {'meta':>5s} {'%pass':>6s}")
    print("-" * 75)
    for d in sorted(os.listdir("results_full")):
        config_dir = os.path.join("results_full", d)
        agg = process_config(config_dir)
        if agg:
            print(f"{d:32s} {agg['n_bundles']:>3d} {agg['avg_bundle_quality']:>9.3f} "
                  f"{agg['avg_structure']:>7.3f} {agg['avg_doc']:>5.3f} {agg['avg_metadata']:>5.3f} "
                  f"{agg['fraction_passing_skill_gate']*100:>5.1f}%")


if __name__ == "__main__":
    main()
