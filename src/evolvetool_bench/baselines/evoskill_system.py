"""EvoSkill baseline — evolves strategy-level text skills (not executable code).

EvoSkill maintains a library of text strategies: named instructions that describe
*how* to approach a class of task. When a task fails, it asks the LLM to analyse
the failure and synthesise a new strategy.  On subsequent tasks it retrieves
relevant strategies by keyword matching and injects them into the system prompt.

Key distinction from ARISE: strategies are prompt text, not Python functions.
The tool library is still composed of executable seed tools; what evolves is the
*guidance* injected around them.
"""

from __future__ import annotations

import json
import re
from typing import Any

from ..harness.runner import AgentSystem


class EvoSkillSystem(AgentSystem):
    """Evolves strategy-level prompt instructions, not executable code."""

    def __init__(self, model: str = "gpt-4o-mini", synthesis_model: str = "gpt-4o-mini",
                 max_strategies: int = 50):
        self.model = model
        self.synthesis_model = synthesis_model
        self.max_strategies = max_strategies

        # Executable tool library (seed tools only — EvoSkill does not create new code)
        self._tools: dict[str, Any] = {}
        self._tool_defs: list[dict] = []
        self._tool_impls: list[dict] = []   # raw dicts for get_library()

        # Strategy library: list of dicts with keys:
        #   name, description, trigger_pattern, instruction, use_count
        self._strategies: list[dict] = []

        self._tools_used: list[str] = []
        self._llm_calls: int = 0

    # ------------------------------------------------------------------
    # AgentSystem interface
    # ------------------------------------------------------------------

    def setup(self, seed_tools: list[dict]) -> None:
        """Load seed tools and reset strategy library."""
        self._tools = {}
        self._tool_defs = []
        self._tool_impls = []
        self._strategies = []

        import inspect
        for tool_def in seed_tools:
            name = tool_def["name"]
            impl = tool_def["implementation"].strip()
            ns: dict = {}
            exec(impl, ns)  # noqa: S102
            fn = ns[name]
            self._tools[name] = fn
            sig = inspect.signature(fn)
            params = {
                "type": "object",
                "properties": {p: {"type": "string"} for p in sig.parameters},
                "required": list(sig.parameters.keys()),
            }
            self._tool_defs.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": tool_def.get("description", ""),
                    "parameters": params,
                },
            })
            self._tool_impls.append({
                "name": name,
                "implementation": tool_def["implementation"],
                "test_suite": tool_def.get("test_suite", ""),
            })

    def run_task(self, task_description: str) -> dict:
        import litellm

        self._tools_used = []
        self._llm_calls = 0

        # 1. Retrieve relevant strategies and build an augmented system prompt
        relevant = self._find_relevant_strategies(task_description)
        system_prompt = self._build_system_prompt(relevant)

        # Mark retrieved strategies as used
        for s in relevant:
            s["use_count"] = s.get("use_count", 0) + 1

        # 2. Run the agent loop (tool-use)
        messages: list[dict] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task_description},
        ]
        final_output = ""
        success = False

        for _ in range(5):
            resp = litellm.completion(
                model=self.model,
                messages=messages,
                tools=self._tool_defs if self._tool_defs else None,
                max_tokens=4096,
            )
            self._llm_calls += 1
            msg = resp.choices[0].message

            if not msg.tool_calls:
                final_output = msg.content or ""
                success = len(final_output) > 20
                break

            messages.append(msg)
            for tc in msg.tool_calls:
                fn = self._tools.get(tc.function.name)
                self._tools_used.append(tc.function.name)
                if fn is None:
                    tool_result = f"Error: tool '{tc.function.name}' not found"
                else:
                    try:
                        args = json.loads(tc.function.arguments)
                        tool_result = str(fn(**args))
                    except Exception as e:
                        tool_result = f"Error: {e}"
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": tool_result,
                })
        else:
            # Exhausted turns — grab whatever the last assistant message said
            for m in reversed(messages):
                role = m if isinstance(m, dict) else {}
                if getattr(m, "role", None) == "assistant" or (isinstance(m, dict) and m.get("role") == "assistant"):
                    content = getattr(m, "content", None) or (m.get("content") if isinstance(m, dict) else None)
                    if content:
                        final_output = content
                        break

        # 3. If the task failed, synthesise a new strategy
        if not success:
            self._evolve_strategy(task_description, messages, final_output)

        return {
            "output": final_output,
            "tools_created": [],          # EvoSkill never creates executable tools
            "tools_used": self._tools_used,
            "llm_calls": self._llm_calls,
        }

    def get_library(self) -> list[dict]:
        """Return the seed-tool library (strategies are not executable tools)."""
        return list(self._tool_impls)

    def reset(self) -> None:
        self._tools = {}
        self._tool_defs = []
        self._tool_impls = []
        self._strategies = []

    # ------------------------------------------------------------------
    # Strategy retrieval
    # ------------------------------------------------------------------

    def _find_relevant_strategies(self, task_description: str) -> list[dict]:
        """Return strategies whose trigger_pattern matches the task description."""
        task_lower = task_description.lower()
        relevant: list[dict] = []
        for strategy in self._strategies:
            pattern = strategy.get("trigger_pattern", "")
            if not pattern:
                continue
            # Support simple keyword list (comma-separated) or regex
            keywords = [k.strip().lower() for k in pattern.split(",") if k.strip()]
            if any(kw in task_lower for kw in keywords):
                relevant.append(strategy)
        # Limit to top 3 most-used (proxy for most-validated)
        relevant.sort(key=lambda s: s.get("use_count", 0), reverse=True)
        return relevant[:3]

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_system_prompt(self, strategies: list[dict]) -> str:
        base = (
            "You are a helpful assistant that solves tasks using the tools provided. "
            "Always call the appropriate tool when available and return a concrete result."
        )
        if not strategies:
            return base

        strategy_text = "\n\n".join(
            f"[Strategy: {s['name']}]\n{s['instruction']}"
            for s in strategies
        )
        return (
            f"{base}\n\n"
            "The following strategies have been learned from previous experience. "
            "Apply them when relevant:\n\n"
            f"{strategy_text}"
        )

    # ------------------------------------------------------------------
    # Strategy evolution
    # ------------------------------------------------------------------

    def _evolve_strategy(self, task: str, messages: list, output: str) -> None:
        """Ask the LLM to analyse the failure and generate a new strategy."""
        if len(self._strategies) >= self.max_strategies:
            return

        import litellm

        # Build a concise conversation summary for the LLM
        tool_calls_summary = ", ".join(self._tools_used) if self._tools_used else "none"
        prompt = (
            "You are analysing why an AI agent failed a task, in order to derive a reusable strategy.\n\n"
            f"TASK: {task}\n\n"
            f"TOOLS CALLED: {tool_calls_summary}\n\n"
            f"FINAL OUTPUT: {output[:800]}\n\n"
            "The agent did not produce a correct, concrete answer.\n\n"
            "Your job: produce a SHORT, reusable strategy (a text instruction) that would help "
            "an agent succeed on similar tasks in the future.\n\n"
            "Respond with a JSON object containing exactly these keys:\n"
            "  name          — short identifier (snake_case, ≤ 5 words)\n"
            "  description   — one sentence explaining when to use this strategy\n"
            "  trigger_pattern — comma-separated keywords that identify tasks needing this strategy\n"
            "  instruction   — 2-5 sentence instruction injected into the system prompt\n\n"
            "Return ONLY the JSON object, no markdown fences."
        )

        try:
            resp = litellm.completion(
                model=self.synthesis_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512,
            )
            self._llm_calls += 1
            raw = resp.choices[0].message.content or ""

            # Strip accidental markdown fences
            raw = re.sub(r"^```[a-z]*\n?", "", raw.strip())
            raw = re.sub(r"\n?```$", "", raw.strip())

            strategy = json.loads(raw)
            required_keys = {"name", "description", "trigger_pattern", "instruction"}
            if not required_keys.issubset(strategy.keys()):
                return

            strategy["use_count"] = 0
            self._strategies.append(strategy)
        except Exception:
            # Synthesis failures are non-fatal — we simply skip
            pass

    # ------------------------------------------------------------------
    # Introspection helpers (useful for analysis scripts)
    # ------------------------------------------------------------------

    def get_strategies(self) -> list[dict]:
        """Return the full strategy library (not part of AgentSystem interface)."""
        return list(self._strategies)
