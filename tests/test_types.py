"""Test the types and metric computation."""
from evolvetool_bench.types import (
    Session, SessionResult, Task, TaskResult, TaskType, ToolRecord,
)


def test_task_completion_rate():
    result = SessionResult(session_id="test")
    result.task_results = [
        TaskResult(task_id="1", task_type=TaskType.SEED, passed=True),
        TaskResult(task_id="2", task_type=TaskType.GAP, passed=False),
        TaskResult(task_id="3", task_type=TaskType.GAP, passed=True),
    ]
    assert result.task_completion_rate == 2 / 3


def test_task_completion_by_type():
    result = SessionResult(session_id="test")
    result.task_results = [
        TaskResult(task_id="1", task_type=TaskType.SEED, passed=True),
        TaskResult(task_id="2", task_type=TaskType.SEED, passed=True),
        TaskResult(task_id="3", task_type=TaskType.GAP, passed=False),
    ]
    by_type = result.task_completion_by_type()
    assert by_type["seed"] == 1.0
    assert by_type["gap"] == 0.0


def test_reuse_rate():
    result = SessionResult(session_id="test")
    result.task_results = [
        TaskResult(task_id="v1", task_type=TaskType.VARIANT, passed=True, tool_reused=True),
        TaskResult(task_id="v2", task_type=TaskType.VARIANT, passed=True, tool_reused=False),
        TaskResult(task_id="r1", task_type=TaskType.REGRESS, passed=True, tool_reused=True),
        TaskResult(task_id="s1", task_type=TaskType.SEED, passed=True),  # not counted
    ]
    assert result.reuse_rate == 2 / 3


def test_composition_success():
    result = SessionResult(session_id="test")
    result.task_results = [
        TaskResult(task_id="c1", task_type=TaskType.COMPOSE, passed=True),
        TaskResult(task_id="c2", task_type=TaskType.COMPOSE, passed=False),
    ]
    assert result.composition_success == 0.5


def test_regression_rate():
    result = SessionResult(session_id="test")
    result.task_results = [
        TaskResult(task_id="r1", task_type=TaskType.REGRESS, passed=True),
        TaskResult(task_id="r2", task_type=TaskType.REGRESS, passed=False),
    ]
    assert result.regression_rate == 0.5


def test_tool_quality_score():
    tool = ToolRecord(
        name="test", implementation="", test_suite="", created_at_task="t1",
        correctness=0.8, robustness=0.6, generality=0.9, code_quality=0.7,
    )
    assert tool.quality_score == (0.8 + 0.6 + 0.9 + 0.7) / 4


def test_creation_efficiency():
    result = SessionResult(session_id="test")
    result.tools_created = [
        ToolRecord(name="tool_a", implementation="", test_suite="", created_at_task="t1"),
        ToolRecord(name="tool_b", implementation="", test_suite="", created_at_task="t2"),
        ToolRecord(name="tool_c", implementation="", test_suite="", created_at_task="t3"),
    ]
    result.task_results = [
        TaskResult(task_id="t1", task_type=TaskType.GAP, passed=True, tools_used=["tool_a"]),
        TaskResult(task_id="t2", task_type=TaskType.GAP, passed=True, tools_used=["tool_b"]),
        TaskResult(task_id="t3", task_type=TaskType.VARIANT, passed=True, tools_used=["tool_a"]),
    ]
    # tool_a and tool_b used, tool_c never used
    assert result.creation_efficiency == 2 / 3


def test_evolvetool_score():
    result = SessionResult(session_id="test")
    result.task_results = [
        TaskResult(task_id="1", task_type=TaskType.SEED, passed=True, llm_calls=2),
    ]
    score = result.evolvetool_score
    assert 0.0 <= score <= 1.0


def test_session_creation():
    from evolvetool_bench.domains.data_transform.session_1 import create_session
    session = create_session()
    assert len(session.tasks) == 11
    assert len(session.seed_tools) == 3
    assert session.domain == "data_transform"

    # Check task type distribution
    types = [t.task_type for t in session.tasks]
    assert types.count(TaskType.SEED) == 3
    assert types.count(TaskType.GAP) == 2
    assert types.count(TaskType.VARIANT) == 2
    assert types.count(TaskType.COMPOSE) == 1
    assert types.count(TaskType.REGRESS) == 1
    assert types.count(TaskType.ADVERSARIAL) == 2
