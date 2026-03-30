"""Run EvolveTool-Bench across all 5 sessions for a given system."""

import argparse
import json
import os
import shutil

from evolvetool_bench.domains.data_transform.session_1 import create_session as s1
from evolvetool_bench.domains.data_transform.session_2 import create_session as s2
from evolvetool_bench.domains.data_transform.session_3 import create_session as s3
from evolvetool_bench.domains.data_transform.session_4 import create_session as s4
from evolvetool_bench.domains.data_transform.session_5 import create_session as s5
from evolvetool_bench.harness.runner import run_session


def main():
    parser = argparse.ArgumentParser(description="Run EvolveTool-Bench (all sessions)")
    parser.add_argument("--system", choices=["arise", "no-evolution"], default="arise")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    sessions = [s1(), s2(), s3(), s4(), s5()]
    all_results = []

    for i, session in enumerate(sessions, 1):
        print(f"\n{'='*60}")
        print(f"SESSION {i}/5: {session.name}")
        print(f"{'='*60}")

        # Clean state between sessions
        for path in ["./bench_skills", "./bench_trajectories"]:
            shutil.rmtree(path, ignore_errors=True)

        if args.system == "arise":
            from evolvetool_bench.baselines.arise_system import ARISESystem
            system = ARISESystem(model=args.model, synthesis_model=args.model)
        else:
            from evolvetool_bench.baselines.no_evolution import NoEvolutionSystem
            system = NoEvolutionSystem(model=args.model)

        result = run_session(system, session, verbose=True)
        summary = result.summary()
        all_results.append(summary)

        # Save per-session result
        with open(os.path.join(args.output_dir, f"s{i}_{args.system}.json"), "w") as f:
            json.dump(summary, f, indent=2)

    # Aggregate across sessions
    aggregate = {
        "system": args.system,
        "model": args.model,
        "sessions": len(all_results),
        "avg_task_completion": sum(r["task_completion"] for r in all_results) / len(all_results),
        "avg_tool_quality": sum(r["mean_tool_quality"] for r in all_results) / len(all_results),
        "avg_reuse_rate": sum(r["reuse_rate"] for r in all_results) / len(all_results),
        "avg_redundancy_rate": sum(r["redundancy_rate"] for r in all_results) / len(all_results),
        "avg_library_health": sum(r["library_health"] for r in all_results) / len(all_results),
        "avg_evolvetool_score": sum(r["evolvetool_score"] for r in all_results) / len(all_results),
        "per_session": all_results,
    }

    with open(os.path.join(args.output_dir, f"aggregate_{args.system}.json"), "w") as f:
        json.dump(aggregate, f, indent=2)

    print(f"\n{'='*60}")
    print(f"AGGREGATE RESULTS ({args.system})")
    print(f"{'='*60}")
    for k, v in aggregate.items():
        if k == "per_session":
            continue
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")


if __name__ == "__main__":
    main()
