# EvolveTool-Bench

Diagnosing how well agents build and maintain tool libraries.

**The first benchmark that evaluates the quality of self-evolved tool libraries — not just whether the agent solves tasks, but whether the tool library is healthy.**

## The Gap

Every existing benchmark (Tool-Genesis, EvoSkill, VOYAGER) treats the tool library as a black box — if the agent solves tasks, the library is deemed good. This is like evaluating a software engineer solely by whether their code runs, ignoring maintainability, duplication, and technical debt.

## What We Measure

| Axis | Metric | What It Shows |
|------|--------|---------------|
| **Task Completion** | Pass rate by task type | Can the agent solve tasks? (standard) |
| **Tool Quality** | Correctness, robustness, generality, code quality | Are individual tools well-built? |
| **Reuse Rate** | % of variant/regress tasks using existing tools | Does the agent recognize reuse opportunities? |
| **Redundancy** | % of functionally duplicate tools | Is the library clean or bloated? |
| **Composition** | % of compose tasks solved | Can the agent chain its own tools? |
| **Regression** | % of regress tasks that fail | Did new tools break old ones? |

## Controlled Sessions

Tasks are structured in sessions with known dependency relationships:

```
Seed → Gap → Variant (reuse?) → Compose (chain?) → Regress (stable?) → Adversarial (refine?)
```

## Quick Start

```bash
pip install -e .
python run_benchmark.py --system arise --model gpt-4o-mini
```

## Baselines

- **No evolution** — seed tools only
- **ARISE** — full iterative evolution
- **One-shot creation** (TODO)
- **EvoSkill** (TODO)
- **VOYAGER-style** (TODO)
