# EvolveTool-Bench

**Diagnosing tool library quality in self-evolving LLM agents.**

[![Paper](https://img.shields.io/badge/paper-LLM4SE%202026-blue)](paper/submission/main.tex)
[![arXiv](https://img.shields.io/badge/arXiv-2604.00392-b31b1b.svg)](https://arxiv.org/abs/2604.00392)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

The first benchmark that evaluates the quality of LLM-generated tool libraries as **software artifacts** — not just whether the agent solves tasks, but whether the tools are correct, reusable, composable, and regression-free.

## The Problem

Every existing benchmark (Tool-Genesis, EvoSkill, VOYAGER) treats the tool library as a black box — if the agent solves tasks, the library is deemed good. This is like evaluating a software engineer solely by whether their code runs, ignoring redundancy, regression, and technical debt.

**Our finding:** Systems with similar task completion (63–68%) differ by up to 18% in library health. Task completion alone hides critical software quality differences.

## Key Results

| System | Model | ETS↑ | TC (%) | Tools | Reuse (%) | LH (%) |
|--------|-------|------|--------|-------|-----------|--------|
| No-Evolution | Sonnet | .518 | 66.7 | 0 | 48.1 | 41.4 |
| EvoSkill | Sonnet | .520 | **68.2** | 0 | 48.1 | 41.4 |
| One-Shot | Sonnet | .519 | 67.7 | 3 | 48.1 | 38.3 |
| **ARISE** | **Sonnet** | .603 | 63.6 | **22** | **70.4** | **51.7** |
| No-Evolution | Haiku | .516 | 66.7 | 0 | 51.9 | 40.1 |
| **ARISE** | **Haiku** | **.612** | 65.2 | 13 | 66.7 | 51.2 |

**6 findings:**
1. Task completion hides software quality differences
2. Code generation outperforms prompt engineering (ARISE vs EvoSkill)
3. Untested code generation is worse than no generation (One-Shot LH < No-Evolution)
4. Reward design is the critical bottleneck, not the synthesis pipeline
5. Tool quality (TQS=0.34) is the frontier — correctness is the weakest dimension
6. Cheaper models produce comparable library quality (Haiku slightly beats Sonnet)

## What We Measure

### Per-Tool: Tool Quality Score (TQS)
| Dimension | What It Tests |
|-----------|---------------|
| Correctness | Hidden unit tests the agent never sees |
| Robustness | Adversarial inputs (empty, null, malformed) |
| Generality | Held-out inputs from the same distribution |
| Code Quality | Docstrings, type hints, error handling, length |

### Per-Library: Library Health (LH)
| Metric | SE Analog |
|--------|-----------|
| Reuse Rate | Code reuse vs duplication |
| Redundancy | Dead/duplicate code detection |
| Precision | Quality gate (TQS ≥ 0.5) |
| Efficiency | Dead code (created but never used) |
| Composition | Function composability |
| Regression | Regression testing |

### Composite: EvolveTool Score (ETS)
```
ETS = 0.25·TC + 0.20·TQS + 0.10·(1-RC) + 0.30·LH + 0.15·SS
```

## Benchmark Structure

**3 domains, 9 sessions, 99 tasks:**

| Domain | Sessions | Format | Tasks |
|--------|----------|--------|-------|
| A: Data Transform | 5 | ABR, RLE, VDL, QLOG, TPACK | 55 |
| B: API Orchestration | 1 | HMAC-timestamp auth, encrypted cursors | 11 |
| C: Numerical | 3 | ARCFIT, ARCSIG, ARCOPT | 33 |

Each session has **11 tasks** with known dependency relationships:

```
Seed (3)  →  Can the agent use provided tools?
Gap (2)   →  Can it create new tools? (proprietary formats)
Variant (2) → Does it reuse or duplicate?
Compose (1) → Can it chain its own tools?
Regress (1) → Do old tools still work?
Adversarial (2) → Can it handle edge cases?
```

All proprietary formats are designed so LLMs **cannot solve them from training data** — the agent must create and execute tools.

## Quick Start

```bash
# Install
pip install -e .

# Run ARISE on all domains (requires Bedrock access)
AWS_PROFILE=your_profile python run_full_matrix.py 2

# Run no-evolution baseline
AWS_PROFILE=your_profile python run_full_matrix.py 1

# Run all 6 configurations sequentially
bash run_remaining.sh

# Regenerate paper figures from results
python generate_figures.py
```

### Run IDs
| ID | System | Model |
|----|--------|-------|
| 1 | No-Evolution | Claude Sonnet |
| 2 | ARISE | Claude Sonnet |
| 3 | EvoSkill | Claude Sonnet |
| 4 | One-Shot | Claude Sonnet |
| 5 | No-Evolution | Claude Haiku |
| 6 | ARISE | Claude Haiku |
| 7 | Human Oracle | Claude Sonnet |

## Baselines

| System | Type | Description |
|--------|------|-------------|
| **No-Evolution** | Lower bound | Seed tools only, no code generation |
| **ARISE** | Code-level | Iterative synthesis + sandbox + adversarial testing |
| **EvoSkill** | Strategy-level | Text prompt evolution, no executable code |
| **One-Shot** | Ablation | Single synthesis attempt, no validation |
| **Human Oracle** | Upper bound | Hand-written reference tools for all gap tasks |

## Project Structure

```
evolvetool-bench/
├── src/evolvetool_bench/
│   ├── types.py                    # Core types: Task, Session, ToolRecord, SessionResult
│   ├── harness/runner.py           # Session runner + AgentSystem interface
│   ├── evaluation/
│   │   ├── tool_quality.py         # TQS evaluator (correctness, robustness, generality, code)
│   │   └── run_quality.py          # Run quality eval on session results
│   ├── baselines/
│   │   ├── no_evolution.py         # Seed tools only
│   │   ├── arise_system.py         # ARISE with LLM-as-judge reward
│   │   ├── evoskill_system.py      # Strategy-level evolution
│   │   ├── oneshot_system.py       # One-shot creation, no validation
│   │   └── human_oracle.py         # Hand-written reference tools
│   └── domains/
│       ├── data_transform/         # Domain A: 5 sessions, proprietary binary formats
│       ├── api_orchestration/      # Domain B: mock server + HMAC auth + pagination
│       └── numerical/              # Domain C: curve fitting, signals, optimization
├── paper/
│   ├── submission/                 # LLM4SE 2026 submission (CEUR format)
│   └── *.pdf                       # Figures
├── results_full/                   # All experimental results (JSON)
├── run_full_matrix.py              # Run benchmark for any system/model
├── run_remaining.sh                # Run all remaining configurations
└── generate_figures.py             # Generate paper figures from results
```

## Adding Your Own System

Implement the `AgentSystem` interface:

```python
from evolvetool_bench.harness.runner import AgentSystem

class MySystem(AgentSystem):
    def setup(self, seed_tools: list[dict]) -> None:
        """Initialize with seed tools."""
        ...

    def run_task(self, task_description: str) -> dict:
        """Run a task. Return {output, tools_created, tools_used, llm_calls}."""
        ...

    def get_library(self) -> list[dict]:
        """Return current tool library."""
        ...
```

Then run:
```python
from evolvetool_bench.domains.data_transform.session_1 import create_session
from evolvetool_bench.harness.runner import run_session

session = create_session()
result = run_session(MySystem(), session)
print(result.summary())
```

## Citation

```bibtex
@inproceedings{kaliyev2026evolvetoolbench,
  title={EvolveTool-Bench: Evaluating the Quality of LLM-Generated Tool Libraries as Software Artifacts},
  author={Kaliyev, Alibek T. and Maryanskyy, Artem},
  booktitle={LLM4SE 2026: Workshop on Large Language Models for Software Engineering},
  year={2026}
}
```

## Links

- [ARISE Framework](https://github.com/abekek/arise) — the code-level evolution system evaluated
- [ARISE Documentation](https://arise-ai.dev)
- [strands-arise](https://github.com/abekek/strands-arise) — ARISE as a Strands Agents plugin

## License

MIT
