"""ARISE baseline — full iterative evolution with sandbox validation."""

from __future__ import annotations

import json
import os
from typing import Any

from ..harness.runner import AgentSystem


class ARISESystem(AgentSystem):
    """Wraps ARISE as a benchmark agent system."""

    def __init__(self, model: str = "gpt-4o-mini", synthesis_model: str = "gpt-4o-mini",
                 failure_threshold: int = 3, skills_path: str = "./bench_skills",
                 trajectories_path: str = "./bench_trajectories"):
        self.model = model
        self.synthesis_model = synthesis_model
        self.failure_threshold = failure_threshold
        self.skills_path = skills_path
        self.trajectories_path = trajectories_path
        self._arise = None
        self._tools_created_this_task: list[dict] = []
        self._tools_used_this_task: list[str] = []
        self._llm_calls = 0

    def setup(self, seed_tools: list[dict]) -> None:
        from arise import ARISE, ARISEConfig
        from arise.skills.library import SkillLibrary
        from arise.types import Skill, SkillStatus, SkillOrigin

        config = ARISEConfig(
            model=self.synthesis_model,
            failure_threshold=self.failure_threshold,
            skill_store_path=self.skills_path,
            trajectory_store_path=self.trajectories_path,
            verbose=False,
            allowed_imports=[
                "json", "csv", "re", "io", "hashlib", "math",
                "collections", "itertools", "datetime", "base64",
            ],
        )

        library = SkillLibrary(self.skills_path)

        # Add seed tools
        for tool_def in seed_tools:
            skill = Skill(
                name=tool_def["name"],
                description=tool_def.get("description", ""),
                implementation=tool_def["implementation"].strip(),
                test_suite="",
                status=SkillStatus.ACTIVE,
                origin=SkillOrigin.MANUAL,
            )
            library.add(skill)
            library.promote(skill.id)

        def agent_fn(task: str, tools: list) -> str:
            import litellm
            tool_map = {t.name: t.fn for t in tools}
            tool_defs = [
                {"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.parameters}}
                for t in tools
            ]
            messages = [{"role": "user", "content": task}]
            self._tools_used_this_task = []

            for _ in range(5):
                resp = litellm.completion(
                    model=self.model, messages=messages,
                    tools=tool_defs if tool_defs else None,
                    max_tokens=4096,
                )
                self._llm_calls += 1
                msg = resp.choices[0].message
                if not msg.tool_calls:
                    return msg.content or ""
                messages.append(msg)
                for tc in msg.tool_calls:
                    fn = tool_map.get(tc.function.name)
                    self._tools_used_this_task.append(tc.function.name)
                    if fn is None:
                        result = f"Error: tool '{tc.function.name}' not found"
                    else:
                        try:
                            args = json.loads(tc.function.arguments)
                            result = str(fn(**args))
                        except Exception as e:
                            result = f"Error: {e}"
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
            return msg.content or ""

        def reward_fn(trajectory):
            if any(s.error for s in trajectory.steps):
                return 0.0
            if len(trajectory.outcome) < 20:
                return 0.0
            # Must have used a non-seed tool to get credit
            # (prevents the agent from just reasoning through the answer)
            seed_names = {t["name"] for t in seed_tools}
            real_tool_used = any(
                s.action and s.action not in seed_names
                for s in trajectory.steps
            )
            if not real_tool_used and trajectory.steps:
                return 0.0
            # If no tools were called at all, check for failure signals
            if not trajectory.steps:
                outcome = trajectory.outcome.lower()
                fail_signals = ["cannot", "don't have", "unable", "no tool", "not possible"]
                if any(s in outcome for s in fail_signals):
                    return 0.0
            return 1.0

        self._arise = ARISE(
            agent_fn=agent_fn,
            reward_fn=reward_fn,
            config=config,
            skill_library=library,
        )

    def run_task(self, task_description: str) -> dict:
        self._tools_used_this_task = []
        self._llm_calls = 0

        skills_before = {s.name for s in self._arise.skills}
        result = self._arise.run(task_description)
        skills_after = {s.name for s in self._arise.skills}

        new_skills = skills_after - skills_before
        tools_created = []
        for name in new_skills:
            for s in self._arise.skills:
                if s.name == name:
                    tools_created.append({
                        "name": s.name,
                        "implementation": s.implementation,
                        "test_suite": s.test_suite,
                    })

        return {
            "output": result,
            "tools_created": tools_created,
            "tools_used": self._tools_used_this_task,
            "llm_calls": self._llm_calls,
        }

    def get_library(self) -> list[dict]:
        return [
            {"name": s.name, "implementation": s.implementation, "test_suite": s.test_suite}
            for s in self._arise.skills
        ]

    def reset(self) -> None:
        import shutil
        for path in [self.skills_path, self.trajectories_path]:
            if os.path.exists(path):
                shutil.rmtree(path)
        self._arise = None
