"""Run the full benchmark matrix: 6 systems × all domains."""

import json
import os
import shutil
import sys

from evolvetool_bench.domains.data_transform.session_1 import create_session as dt1
from evolvetool_bench.domains.data_transform.session_2 import create_session as dt2
from evolvetool_bench.domains.data_transform.session_3 import create_session as dt3
from evolvetool_bench.domains.data_transform.session_4 import create_session as dt4
from evolvetool_bench.domains.data_transform.session_5 import create_session as dt5
from evolvetool_bench.domains.api_orchestration.session_1 import create_session as api1
from evolvetool_bench.domains.api_orchestration.mock_server import start_mock_server, stop_mock_server
from evolvetool_bench.domains.numerical.session_1 import create_session as num1
from evolvetool_bench.domains.numerical.session_2 import create_session as num2
from evolvetool_bench.domains.numerical.session_3 import create_session as num3
from evolvetool_bench.harness.runner import run_session
from evolvetool_bench.evaluation.run_quality import evaluate_session_tools


SONNET = "bedrock/us.anthropic.claude-sonnet-4-20250514-v1:0"
HAIKU = "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0"


def make_system(name, model):
    if name == "no-evolution":
        from evolvetool_bench.baselines.no_evolution import NoEvolutionSystem
        return NoEvolutionSystem(model=model)
    elif name == "arise":
        from evolvetool_bench.baselines.arise_system import ARISESystem
        return ARISESystem(model=model, synthesis_model=model, failure_threshold=1)
    elif name == "evoskill":
        from evolvetool_bench.baselines.evoskill_system import EvoSkillSystem
        return EvoSkillSystem(model=model, synthesis_model=model)
    elif name == "oneshot":
        from evolvetool_bench.baselines.oneshot_system import OneShotSystem
        return OneShotSystem(model=model, synthesis_model=model)
    else:
        raise ValueError(f"Unknown system: {name}")


def run_one(system_name, model_name, model_id, sessions, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    all_results = []
    all_tools = []

    for i, session in enumerate(sessions, 1):
        print(f"\n{'='*60}")
        print(f"[{system_name}/{model_name}] SESSION {i}/{len(sessions)}: {session.name}")
        print(f"{'='*60}")

        for path in ["./bench_skills", "./bench_trajectories"]:
            shutil.rmtree(path, ignore_errors=True)

        system = make_system(system_name, model_id)
        result = run_session(system, session, verbose=True)

        # Evaluate tool quality
        if result.tools_created:
            evaluate_session_tools(result, session.tasks)
            for t in result.tools_created:
                all_tools.append({
                    "session": session.id, "name": t.name,
                    "tqs": t.quality_score, "correctness": t.correctness,
                    "robustness": t.robustness, "generality": t.generality,
                    "code_quality": t.code_quality,
                })

        summary = result.summary()
        all_results.append(summary)
        with open(os.path.join(output_dir, f"s{i}.json"), "w") as f:
            json.dump(summary, f, indent=2)

    n = len(all_results)
    aggregate = {
        "system": system_name, "model": model_name, "sessions": n,
        "avg_task_completion": sum(r["task_completion"] for r in all_results) / n,
        "avg_tool_quality": sum(r["mean_tool_quality"] for r in all_results) / n,
        "avg_reuse_rate": sum(r["reuse_rate"] for r in all_results) / n,
        "avg_library_health": sum(r["library_health"] for r in all_results) / n,
        "avg_evolvetool_score": sum(r["evolvetool_score"] for r in all_results) / n,
        "total_tools": sum(r["tools_created"] for r in all_results),
        "tool_details": all_tools,
    }
    with open(os.path.join(output_dir, "aggregate.json"), "w") as f:
        json.dump(aggregate, f, indent=2)

    print(f"\n  {system_name}/{model_name}: ETS={aggregate['avg_evolvetool_score']:.3f} TC={aggregate['avg_task_completion']:.3f} Tools={aggregate['total_tools']}")
    return aggregate


def main():
    # Parse which run to do
    if len(sys.argv) < 2:
        print("Usage: python run_full_matrix.py <run_id>")
        print("  run_id: 1=noevol/sonnet, 2=arise/sonnet, 3=evoskill/sonnet, 4=oneshot/sonnet, 5=noevol/haiku, 6=arise/haiku")
        return

    run_id = int(sys.argv[1])

    runs = [
        ("no-evolution", "sonnet", SONNET),
        ("arise", "sonnet", SONNET),
        ("evoskill", "sonnet", SONNET),
        ("oneshot", "sonnet", SONNET),
        ("no-evolution", "haiku", HAIKU),
        ("arise", "haiku", HAIKU),
    ]

    system_name, model_name, model_id = runs[run_id - 1]

    # Build sessions
    sessions = [dt1(), dt2(), dt3(), dt4(), dt5()]

    # Start mock server for API sessions
    server, _ = start_mock_server(18080)
    sessions.append(api1())

    # Add numerical sessions
    sessions.extend([num1(), num2(), num3()])

    print(f"Running {system_name}/{model_name} on {len(sessions)} sessions")

    output_dir = f"results_full/{system_name}_{model_name}"
    result = run_one(system_name, model_name, model_id, sessions, output_dir)

    stop_mock_server(server)
    print("\nDone!")


if __name__ == "__main__":
    main()
