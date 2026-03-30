"""Run EvolveTool-Bench on a baseline system."""

import argparse
import json
import shutil

from evolvetool_bench.domains.data_transform.session_1 import create_session
from evolvetool_bench.harness.runner import run_session


def main():
    parser = argparse.ArgumentParser(description="Run EvolveTool-Bench")
    parser.add_argument("--system", choices=["arise", "no-evolution"], default="arise")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--output", default="results.json")
    args = parser.parse_args()

    session = create_session()

    if args.system == "arise":
        from evolvetool_bench.baselines.arise_system import ARISESystem
        system = ARISESystem(model=args.model, synthesis_model=args.model)
    else:
        from evolvetool_bench.baselines.no_evolution import NoEvolutionSystem
        system = NoEvolutionSystem(model=args.model)

    # Clean previous run
    for path in ["./bench_skills", "./bench_trajectories"]:
        shutil.rmtree(path, ignore_errors=True)

    result = run_session(system, session, verbose=True)

    # Save results
    with open(args.output, "w") as f:
        json.dump(result.summary(), f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
