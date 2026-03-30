"""Run EvolveTool-Bench across sessions for a given system."""

import argparse
import json
import os
import shutil

from evolvetool_bench.harness.runner import run_session


def main():
    parser = argparse.ArgumentParser(description="Run EvolveTool-Bench (all sessions)")
    parser.add_argument("--system", choices=["arise", "no-evolution"], default="arise")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--failure-threshold", type=int, default=3)
    parser.add_argument("--domain", choices=["data_transform", "api_orchestration", "all"], default="all")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    sessions = []
    mock_server = None

    if args.domain in ("data_transform", "all"):
        from evolvetool_bench.domains.data_transform.session_1 import create_session as dt1
        from evolvetool_bench.domains.data_transform.session_2 import create_session as dt2
        from evolvetool_bench.domains.data_transform.session_3 import create_session as dt3
        from evolvetool_bench.domains.data_transform.session_4 import create_session as dt4
        from evolvetool_bench.domains.data_transform.session_5 import create_session as dt5
        sessions.extend([dt1(), dt2(), dt3(), dt4(), dt5()])

    if args.domain in ("api_orchestration", "all"):
        from evolvetool_bench.domains.api_orchestration.session_1 import create_session as api1
        from evolvetool_bench.domains.api_orchestration.mock_server import start_mock_server, stop_mock_server
        sessions.append(api1())
        mock_server_obj, _ = start_mock_server(port=18080)
        mock_server = mock_server_obj
        print("Mock API server started on port 18080")
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
            system = ARISESystem(model=args.model, synthesis_model=args.model, failure_threshold=args.failure_threshold)
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

    # Cleanup
    if mock_server:
        from evolvetool_bench.domains.api_orchestration.mock_server import stop_mock_server
        stop_mock_server(mock_server)
        print("Mock server stopped")


if __name__ == "__main__":
    main()
