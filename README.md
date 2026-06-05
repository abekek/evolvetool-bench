# EvolveTool-Bench

**A diagnostic framework for evaluating evolving LLM-generated tool libraries as auditable software artifacts.**

[![Paper](https://img.shields.io/badge/paper-KDD%20Eval%202026-blue)](paper/kdd_eval2026/main.tex)
[![arXiv](https://img.shields.io/badge/arXiv-2604.00392-b31b1b.svg)](https://arxiv.org/abs/2604.00392)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

EvolveTool-Bench evaluates LLM agents that create and accumulate tools at runtime — not just whether they solve the immediate task, but whether the generated tool library is **correct, reusable, composable, auditable, and regression-free**.

## What the Benchmark Diagnoses

| Question | Metric |
|----------|--------|
| Did the agent solve the task? | Verified task completion (TC) |
| Does the generated tool pass independent tests? | Tool conformance / correctness |
| Did tool reuse help or hurt? | Correct reuse vs. incorrect reuse |
| Is the library accumulating duplicates? | Redundancy rate |
| Are created tools actually used later? | Utilization |
| Can tools be chained across tasks? | Composition success |
| Did library growth break prior behavior? | Regression rate |
| Can the run be inspected and reproduced? | Per-task audit trace |

The contribution is a **diagnostic methodology and set of reusable evaluation practices**, not a leaderboard. Any tool-generating or skill-generating agent can be evaluated against the same tasks and metrics.

## Key Finding

Task completion alone is insufficient for tool-evolving agents. Systems with similar TC (63–72%) differ substantially in library quality dimensions. The *reuse paradox* illustrates why: an agent with high raw reuse can still propagate defects, making **correct vs. incorrect reuse decomposition** essential for honest reporting.

## Benchmark Structure

**3 domains · 9 sessions · 99 tasks.** 51/99 tasks currently carry deterministic verifiers (`expected` mappings or `verify` predicates); the harness fails closed on unverified tasks. Extending deterministic verification to all 99 tasks is in progress (see `scripts/audit_tasks.py`):

| Domain | Sessions | What it probes |
|--------|----------|----------------|
| A: Data Transform | 5 | Proprietary binary formats (ABR, RLE, VDL, QLOG, TPACK) |
| B: API Orchestration | 1 | HMAC auth, encrypted cursors, mock server |
| C: Numerical | 3 | Curve fitting, signal processing, optimization |

Each session contains **11 tasks with known dependency relationships**:

| Task type | What it diagnoses |
|-----------|------------------|
| Seed (×3) | Can the agent use provided tools? |
| Gap (×2) | Can it create a missing capability? |
| Variant (×2) | Does it reuse or duplicate? |
| Compose (×1) | Can it chain self-created tools? |
| Regress (×1) | Did library growth break prior behavior? |
| Adversarial (×2) | Does the tool handle edge cases? |

All proprietary formats are designed so the agent must create and execute tools — prior training data is insufficient.

## Metrics

### Per-tool: Tool Quality Score (TQS)

| Dimension | Test source |
|-----------|-------------|
| Correctness | Capability-aligned hidden unit tests |
| Generality | Held-out same-distribution inputs |
| Robustness | Adversarial edge cases |
| Code quality | Static analysis (radon CC, MI, control flow) |

### Per-library: Library Health (LH)

Correct reuse rate · incorrect reuse rate · redundancy · utilization · composition · regression stability.

### Composite: EvolveTool Score (ETS)

```
ETS = 0.30·TC + 0.20·TQS + 0.10·(1−cost) + 0.40·LH
```

Safety is excluded from the composite pending a proper implementation (see Limitations).

## Quick Start

```bash
pip install -e ".[dev]"

# Audit verifier coverage (expected/verify predicates per task)
python scripts/audit_tasks.py

# Run a session against the no-evolution baseline
python -c "
from evolvetool_bench.domains.data_transform.session_1 import SESSION
from evolvetool_bench.baselines.no_evolution import NoEvolution
from evolvetool_bench.harness.runner import run_session
result = run_session(NoEvolution(), SESSION)
print(result.summary())
"

# Regenerate paper tables from canonical results
python scripts/make_tables.py --results-dir results_canonical/ --output paper/kdd_eval2026/
```

## Adding Your Own System

Implement the `AgentSystem` interface:

```python
from evolvetool_bench.harness.runner import AgentSystem

class MySystem(AgentSystem):
    def setup(self, seed_tools: list[dict]) -> None: ...
    def run_task(self, task_description: str) -> dict: ...
    def get_library(self) -> list[dict]: ...
```

Return format for `run_task`:
```python
{
    "output": str,           # agent's answer
    "tools_created": [...],  # [{name, implementation, test_suite}]
    "tools_used": [...],     # [tool_name, ...]
    "llm_calls": int,
}
```

## Reusable Evaluation Practices

EvolveTool-Bench is intended less as a fixed leaderboard than as a set of reusable evaluation practices for tool-evolving agents. We recommend that future benchmarks:

1. Separate task success from generated-artifact conformance
2. Report correct and incorrect reuse rather than raw reuse alone
3. Include explicit regression tasks after library growth
4. Preserve generated artifacts and source hashes
5. Publish per-task traces sufficient to reproduce decisions
6. Evaluate library-management policies (promotion, deduplication, retirement)
7. Use capability-aligned hidden tests, not a single test pool per session

## Repository Structure

```
evolvetool-bench/
├── src/evolvetool_bench/
│   ├── types.py                    # Task, Session, ToolRecord, SessionResult
│   ├── harness/runner.py           # Session runner + AgentSystem interface
│   ├── evaluation/tool_quality.py  # TQS (capability-aligned hidden tests)
│   ├── baselines/                  # no_evolution, arise, creator, oneshot, toolmaker, toolcoder
│   └── domains/                    # data_transform, api_orchestration, numerical
├── paper/kdd_eval2026/             # Canonical KDD 2026 workshop submission
├── results_canonical/              # Canonical result manifests
│   ├── run_manifest.jsonl
│   ├── tool_manifest.jsonl
│   └── aggregate.json
└── scripts/
    ├── audit_tasks.py              # Audit verifier coverage (expected/verify per task)
    ├── audit_results.py            # Check result file completeness
    ├── make_tables.py              # Regenerate LaTeX tables from results
    └── make_figures.py             # Regenerate figures from results
```

## Citation

```bibtex
@inproceedings{kaliyev2026evolvetoolbench,
  title     = {EvolveTool-Bench: A Diagnostic Framework for Trustworthy Tool-Library Evolution},
  author    = {Kaliyev, Alibek T. and Maryanskyy, Artem},
  booktitle = {Workshop on Evaluation and Trustworthiness of Agentic AI at KDD 2026},
  year      = {2026}
}
```

## License

MIT
