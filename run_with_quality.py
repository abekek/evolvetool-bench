"""Run benchmark with tool quality evaluation."""

import json
import os
import shutil

from evolvetool_bench.domains.data_transform.session_1 import create_session as dt1
from evolvetool_bench.domains.data_transform.session_2 import create_session as dt2
from evolvetool_bench.domains.data_transform.session_3 import create_session as dt3
from evolvetool_bench.domains.data_transform.session_4 import create_session as dt4
from evolvetool_bench.domains.data_transform.session_5 import create_session as dt5
from evolvetool_bench.domains.api_orchestration.session_1 import create_session as api1
from evolvetool_bench.domains.api_orchestration.mock_server import start_mock_server, stop_mock_server
from evolvetool_bench.harness.runner import run_session
from evolvetool_bench.baselines.arise_system import ARISESystem
from evolvetool_bench.evaluation.run_quality import evaluate_session_tools


def main():
    model = "bedrock/us.anthropic.claude-sonnet-4-20250514-v1:0"
    output_dir = "results_quality"
    os.makedirs(output_dir, exist_ok=True)

    sessions = [dt1(), dt2(), dt3(), dt4(), dt5()]

    # Start mock server for API sessions
    server, _ = start_mock_server(18080)
    sessions.append(api1())
    print("Mock server started")

    all_results = []
    all_tool_details = []

    for i, session in enumerate(sessions, 1):
        print(f"\n{'='*60}")
        print(f"SESSION {i}/{len(sessions)}: {session.name}")
        print(f"{'='*60}")

        for path in ["./bench_skills", "./bench_trajectories"]:
            shutil.rmtree(path, ignore_errors=True)

        system = ARISESystem(model=model, synthesis_model=model, failure_threshold=1)
        result = run_session(system, session, verbose=True)

        # Run tool quality evaluation
        if result.tools_created:
            print(f"\n  Evaluating {len(result.tools_created)} tool(s)...")
            evaluate_session_tools(result, session.tasks)
            for tool in result.tools_created:
                print(f"    {tool.name}: TQS={tool.quality_score:.2f} "
                      f"(correct={tool.correctness:.2f}, robust={tool.robustness:.2f}, "
                      f"general={tool.generality:.2f}, code={tool.code_quality:.2f})")
                all_tool_details.append({
                    "session": session.id,
                    "name": tool.name,
                    "tqs": tool.quality_score,
                    "correctness": tool.correctness,
                    "robustness": tool.robustness,
                    "generality": tool.generality,
                    "code_quality": tool.code_quality,
                    "implementation_lines": len(tool.implementation.split("\n")),
                    "created_at_task": tool.created_at_task,
                })

        summary = result.summary()
        all_results.append(summary)

        with open(os.path.join(output_dir, f"s{i}.json"), "w") as f:
            json.dump(summary, f, indent=2)

    stop_mock_server(server)

    # Aggregate
    n = len(all_results)
    aggregate = {
        "system": "arise+judge",
        "model": model,
        "sessions": n,
        "total_tasks": sum(len(r.get("task_completion_by_type", {})) for r in all_results) or n * 11,
        "avg_task_completion": sum(r["task_completion"] for r in all_results) / n,
        "avg_tool_quality": sum(r["mean_tool_quality"] for r in all_results) / n,
        "avg_reuse_rate": sum(r["reuse_rate"] for r in all_results) / n,
        "avg_redundancy_rate": sum(r["redundancy_rate"] for r in all_results) / n,
        "avg_library_health": sum(r["library_health"] for r in all_results) / n,
        "avg_evolvetool_score": sum(r["evolvetool_score"] for r in all_results) / n,
        "total_tools_created": sum(r["tools_created"] for r in all_results),
        "tool_details": all_tool_details,
    }

    with open(os.path.join(output_dir, "aggregate.json"), "w") as f:
        json.dump(aggregate, f, indent=2)

    print(f"\n{'='*60}")
    print(f"AGGREGATE WITH TOOL QUALITY")
    print(f"{'='*60}")
    for k, v in aggregate.items():
        if k in ("tool_details",):
            print(f"  {k}: {len(v)} tools evaluated")
            continue
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")

    if all_tool_details:
        avg_tqs = sum(t["tqs"] for t in all_tool_details) / len(all_tool_details)
        print(f"\n  Overall avg TQS: {avg_tqs:.3f}")
        for t in all_tool_details:
            print(f"    {t['session']}/{t['name']}: TQS={t['tqs']:.2f}")


if __name__ == "__main__":
    main()
