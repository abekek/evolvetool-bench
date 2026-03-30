"""Run tool quality evaluation on a SessionResult's created tools."""

from ..types import SessionResult, Task
from .tool_quality import evaluate_tool, detect_redundancy


def evaluate_session_tools(result: SessionResult, tasks: list[Task]) -> SessionResult:
    """Evaluate all tools created during a session.

    Runs hidden tests and adversarial tests from the corresponding GAP tasks
    on each synthesized tool. Updates tool quality scores in-place.

    Returns the updated SessionResult.
    """
    # Build a map of task_id → task for finding hidden/adversarial tests
    task_map = {t.id: t for t in tasks}

    for tool in result.tools_created:
        # Find the task that triggered this tool's creation
        task = task_map.get(tool.created_at_task)
        hidden = task.hidden_tests if task else []
        adversarial = task.adversarial_tests if task else []

        evaluate_tool(tool, hidden, adversarial)

    # Compute redundancy across all created tools
    if len(result.tools_created) > 1:
        # Use hidden tests from all gap tasks as shared test inputs
        all_test_inputs = []
        for task in tasks:
            if task.hidden_tests:
                all_test_inputs.extend(task.hidden_tests)
        if all_test_inputs:
            result.redundancy_rate = detect_redundancy(result.tools_created, all_test_inputs)

    return result
