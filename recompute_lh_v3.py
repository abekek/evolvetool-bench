"""Recompute Library Health for configs with preserved tools, using the v3 TQS
values (computed under expanded hidden tests).

LH = mean(reuse, 1-redundancy, quality_gate, utilization, composability, 1-regression)

Only the quality_gate (π) sub-metric depends on TQS — the others depend on per-task
outcomes which don't change with the test suite. So we keep the v1 values for
everything except π, and recompute π = fraction of tools with TQS_v3 ≥ 0.5.
"""
from __future__ import annotations

import json
import os
import glob


def recompute_for_config(config_dir: str) -> dict | None:
    tools_root = os.path.join(config_dir, "tools")
    if not os.path.isdir(tools_root):
        return None

    # Load v3 per-tool meta to get new TQS values
    tool_tqs_v3 = []
    tool_tqs_v1 = []
    for meta_path in sorted(glob.glob(os.path.join(tools_root, "*", "*.meta_v3.json"))):
        with open(meta_path) as f:
            m = json.load(f)
        tool_tqs_v3.append(m["scores_v3"]["tqs"])
        tool_tqs_v1.append(m["scores_v1"]["tqs"])

    if not tool_tqs_v3:
        return None

    pi_v1 = sum(1 for t in tool_tqs_v1 if t >= 0.5) / len(tool_tqs_v1)
    pi_v3 = sum(1 for t in tool_tqs_v3 if t >= 0.5) / len(tool_tqs_v3)

    # Pull per-session LH components and recompute average LH using new π
    deltas = []
    for sf in sorted(glob.glob(os.path.join(config_dir, "s*.json"))):
        with open(sf) as f:
            s = json.load(f)
        # The per-session π_v1 is s["library_precision"]; we don't have per-session
        # π_v3 (would need per-session tool partition). Use config-mean π_v3.
        rho_r = s.get("reuse_rate", 0)
        rho_d = s.get("redundancy_rate", 0)
        eta = s.get("creation_efficiency", 0)
        gamma = s.get("composition_success", 0)
        delta = s.get("regression_rate", 0)
        # Recompute LH replacing library_precision with config-mean π_v3
        lh_v1 = s.get("library_health", 0)
        lh_v3 = (rho_r + (1 - rho_d) + pi_v3 + eta + gamma + (1 - delta)) / 6
        deltas.append((lh_v1, lh_v3))

    if not deltas:
        return None

    n = len(deltas)
    return {
        "config": os.path.basename(config_dir),
        "n_tools": len(tool_tqs_v3),
        "n_sessions": n,
        "pi_v1": pi_v1,
        "pi_v3": pi_v3,
        "lh_v1_mean": sum(d[0] for d in deltas) / n,
        "lh_v3_mean": sum(d[1] for d in deltas) / n,
        "lh_delta": (sum(d[1] for d in deltas) - sum(d[0] for d in deltas)) / n,
    }


def main() -> None:
    print(f"{'config':30s} {'n':>3s}  v1 π  v3 π   ΔvLH (v3-v1)   v1 LH   v3 LH")
    print("-" * 85)
    for d in sorted(os.listdir("results_full")):
        config_dir = os.path.join("results_full", d)
        r = recompute_for_config(config_dir)
        if r:
            print(f"{r['config']:30s} {r['n_tools']:>3d}  "
                  f"{r['pi_v1']:.2f}  {r['pi_v3']:.2f}    "
                  f"{r['lh_delta']:+.3f}        "
                  f"{r['lh_v1_mean']:.3f}   {r['lh_v3_mean']:.3f}")


if __name__ == "__main__":
    main()
