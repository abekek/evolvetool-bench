"""Core types for EvolveTool-Bench."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable


class TaskType(Enum):
    """Task types in a diagnostic session."""
    SEED = "seed"              # Solvable with starter tools
    GAP = "gap"                # Requires creating a new tool
    VARIANT = "variant"        # Same capability as a gap task, different params — tests reuse
    COMPOSE = "compose"        # Requires chaining 2+ self-created tools
    REGRESS = "regress"        # Re-tests earlier scenario — tests stability
    ADVERSARIAL = "adversarial"  # Designed to break naive implementations — tests refinement


@dataclass
class Task:
    """A single task in a session."""
    id: str
    description: str
    task_type: TaskType
    # Which prior task's tool should be reused (for VARIANT/REGRESS)
    reuses_task: str | None = None
    # Which prior tasks' tools must be composed (for COMPOSE)
    composes_tasks: list[str] = field(default_factory=list)
    # Which prior task's tool this is designed to break (for ADVERSARIAL)
    breaks_task: str | None = None
    # Test function: (tool_output) -> bool
    verify: Callable[[Any], bool] | None = None
    # Expected output for exact match (alternative to verify)
    expected: Any = None
    # Hidden test inputs for tool quality evaluation
    hidden_tests: list[dict] = field(default_factory=list)
    # Adversarial test inputs for robustness evaluation
    adversarial_tests: list[dict] = field(default_factory=list)


@dataclass
class Session:
    """A structured session of tasks with known dependencies."""
    id: str
    name: str
    domain: str
    tasks: list[Task]
    seed_tools: list[dict] = field(default_factory=list)  # pre-provided tools
    description: str = ""


@dataclass
class ToolRecord:
    """A tool created during a session."""
    name: str
    implementation: str
    test_suite: str
    created_at_task: str  # which task triggered creation
    version: int = 1
    # Quality scores (computed by evaluator)
    correctness: float = 0.0  # passes hidden unit tests
    robustness: float = 0.0   # passes adversarial tests
    generality: float = 0.0   # works on held-out inputs
    code_quality: float = 0.0  # docstring, types, error handling, etc.

    @property
    def quality_score(self) -> float:
        """Tool Quality Score (TQS) = mean of 4 dimensions."""
        return (self.correctness + self.robustness + self.generality + self.code_quality) / 4


@dataclass
class TaskResult:
    """Result of running a single task."""
    task_id: str
    task_type: TaskType
    passed: bool
    partial: bool = False  # correct structure but wrong values
    tool_created: str | None = None  # name of tool created (if any)
    tools_used: list[str] = field(default_factory=list)  # tools invoked
    tool_reused: bool = False  # was an existing tool reused?
    llm_calls: int = 0
    duration_ms: float = 0.0
    error: str | None = None
    agent_output: str = ""


@dataclass
class SessionResult:
    """Complete results from running a session."""
    session_id: str
    task_results: list[TaskResult] = field(default_factory=list)
    tools_created: list[ToolRecord] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

    # ── Axis 1: Task Completion ──────────────────────────────────

    @property
    def task_completion_rate(self) -> float:
        if not self.task_results:
            return 0.0
        passed = sum(1 for r in self.task_results if r.passed)
        partial = sum(0.5 for r in self.task_results if r.partial and not r.passed)
        return (passed + partial) / len(self.task_results)

    def task_completion_by_type(self) -> dict[str, float]:
        by_type: dict[str, list[bool]] = {}
        for r in self.task_results:
            by_type.setdefault(r.task_type.value, []).append(r.passed)
        return {t: sum(v) / len(v) if v else 0.0 for t, v in by_type.items()}

    # ── Axis 2: Tool Quality ─────────────────────────────────────

    @property
    def mean_tool_quality(self) -> float:
        if not self.tools_created:
            return 0.0
        return sum(t.quality_score for t in self.tools_created) / len(self.tools_created)

    # ── Axis 3: Refinement Efficiency ────────────────────────────

    @property
    def total_llm_calls(self) -> int:
        return sum(r.llm_calls for r in self.task_results)

    # ── Axis 4: Library Diagnostics ──────────────────────────────

    @property
    def reuse_rate(self) -> float:
        """How often agent reused existing tool when it could have."""
        variant_regress = [r for r in self.task_results
                          if r.task_type in (TaskType.VARIANT, TaskType.REGRESS)]
        if not variant_regress:
            return 0.0
        reused = sum(1 for r in variant_regress if r.tool_reused)
        return reused / len(variant_regress)

    @property
    def redundancy_rate(self) -> float:
        """Proportion of functionally duplicate tools."""
        if len(self.tools_created) <= 1:
            return 0.0
        # Computed externally by running tools on shared test suites
        # Placeholder — set by evaluator
        return self._redundancy_rate

    @redundancy_rate.setter
    def redundancy_rate(self, value: float) -> None:
        self._redundancy_rate = value

    _redundancy_rate: float = field(default=0.0, init=False, repr=False)

    @property
    def library_precision(self) -> float:
        """Proportion of tools above quality threshold."""
        if not self.tools_created:
            return 0.0
        good = sum(1 for t in self.tools_created if t.quality_score >= 0.5)
        return good / len(self.tools_created)

    @property
    def creation_efficiency(self) -> float:
        """Proportion of created tools that were actually used."""
        if not self.tools_created:
            return 0.0
        used_names = set()
        for r in self.task_results:
            used_names.update(r.tools_used)
        created_names = {t.name for t in self.tools_created}
        used_created = used_names & created_names
        return len(used_created) / len(created_names)

    @property
    def composition_success(self) -> float:
        """Proportion of compose tasks solved."""
        compose = [r for r in self.task_results if r.task_type == TaskType.COMPOSE]
        if not compose:
            return 0.0
        return sum(1 for r in compose if r.passed) / len(compose)

    @property
    def regression_rate(self) -> float:
        """Proportion of regress tasks that failed."""
        regress = [r for r in self.task_results if r.task_type == TaskType.REGRESS]
        if not regress:
            return 0.0
        return sum(1 for r in regress if not r.passed) / len(regress)

    # ── Axis 5: Safety ───────────────────────────────────────────

    @property
    def safety_score(self) -> float:
        """Placeholder — computed by static analysis."""
        return self._safety_score

    @safety_score.setter
    def safety_score(self, value: float) -> None:
        self._safety_score = value

    _safety_score: float = field(default=1.0, init=False, repr=False)

    # ── Composite Score ──────────────────────────────────────────

    @property
    def library_health(self) -> float:
        """Mean of library diagnostic metrics."""
        metrics = [
            self.reuse_rate,
            1 - self.redundancy_rate,
            self.library_precision,
            self.creation_efficiency,
            self.composition_success,
            1 - self.regression_rate,
        ]
        return sum(metrics) / len(metrics)

    @property
    def evolvetool_score(self) -> float:
        """ETS — composite score (higher is better)."""
        refinement_cost = self.total_llm_calls / max(len(self.task_results) * 10, 1)
        return (
            0.25 * self.task_completion_rate
            + 0.20 * self.mean_tool_quality
            + 0.10 * max(0, 1 - refinement_cost)
            + 0.30 * self.library_health
            + 0.15 * self.safety_score
        )

    def summary(self) -> dict:
        """Full metrics summary."""
        return {
            "session_id": self.session_id,
            "task_completion": self.task_completion_rate,
            "task_completion_by_type": self.task_completion_by_type(),
            "tools_created": len(self.tools_created),
            "mean_tool_quality": self.mean_tool_quality,
            "reuse_rate": self.reuse_rate,
            "redundancy_rate": self.redundancy_rate,
            "library_precision": self.library_precision,
            "creation_efficiency": self.creation_efficiency,
            "composition_success": self.composition_success,
            "regression_rate": self.regression_rate,
            "library_health": self.library_health,
            "safety_score": self.safety_score,
            "total_llm_calls": self.total_llm_calls,
            "evolvetool_score": self.evolvetool_score,
        }
