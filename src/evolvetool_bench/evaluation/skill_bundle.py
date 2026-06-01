"""Skill bundle construction + evaluation.

A *skill bundle* extends a single Python function (ToolRecord) into a multi-file
artifact resembling Anthropic Claude Skills / CoEvoSkills:

    skill_name/
      SKILL.md           # human-readable description
      function.py        # the callable
      tests.py           # auto-generated tests
      metadata.json      # {name, version, dependencies, tags, created_at_task, ...}

This module provides two functions:

    wrap_as_skill_bundle(tool, task_id, ...) -> SkillBundle
        constructs the bundle from a ToolRecord. SKILL.md is generated from the
        function's docstring + a usage example; metadata.json is filled from the
        tool's record. Used by Code-Evol (and any system that opts in) to emit
        skills rather than bare tools.

    evaluate_skill_bundle(bundle) -> SkillBundle
        mutates and returns the bundle with three bundle-level scores filled in:
        structure_score, doc_score, metadata_score. These complement the
        underlying ToolRecord's TQS to form the bundle_quality_score.
"""

from __future__ import annotations

import ast
import json
import re
from typing import Iterable

from ..types import ToolRecord, SkillBundle


def wrap_as_skill_bundle(tool: ToolRecord, task_id: str,
                          tags: Iterable[str] | None = None,
                          dependencies: Iterable[str] | None = None) -> SkillBundle:
    """Construct a SkillBundle around an already-synthesized tool."""
    docstring = _extract_docstring(tool.implementation) or ""
    name = tool.name
    skill_md = _build_skill_md(name, docstring, tool.implementation, tool.test_suite)
    metadata = {
        "name": name,
        "version": tool.version,
        "created_at_task": task_id,
        "tags": list(tags) if tags else [],
        "dependencies": list(dependencies) if dependencies else _infer_stdlib_imports(tool.implementation),
    }
    return SkillBundle(name=name, tool=tool, skill_md=skill_md, metadata=metadata)


def evaluate_skill_bundle(bundle: SkillBundle) -> SkillBundle:
    """Score the bundle's structure, documentation, and metadata."""
    bundle.structure_score = _eval_structure(bundle)
    bundle.doc_score = _eval_doc(bundle)
    bundle.metadata_score = _eval_metadata(bundle)
    return bundle


# ── Internal: constructors ───────────────────────────────────────────


def _build_skill_md(name: str, docstring: str, implementation: str, tests: str) -> str:
    """Generate a SKILL.md description from the tool's source.

    Format mirrors Anthropic Claude Skills:
        # <name>
        ## Description
        ## Usage
        ## Inputs / Outputs
        ## Example
    """
    sig = _extract_signature(implementation)
    inputs, returns = _parse_doc_io(docstring)

    sections = []
    sections.append(f"# {name}\n")
    sections.append("## Description\n")
    sections.append((docstring.split("\n\n")[0].strip() or
                     f"Auto-generated skill {name}.") + "\n")
    sections.append("## Usage\n")
    sections.append("```python\n"
                    f"from {name} import {name}\n"
                    f"result = {name}({_call_args(sig)})\n"
                    f"print(result)\n"
                    "```\n")
    if inputs:
        sections.append("## Inputs\n" + inputs.strip() + "\n")
    if returns:
        sections.append("## Returns\n" + returns.strip() + "\n")
    if tests:
        sections.append("## Example Tests\n```python\n" + tests.strip()[:600] + "\n```\n")
    return "\n".join(sections)


def _extract_docstring(code: str) -> str | None:
    """Pull the docstring out of the first function in code."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            doc = ast.get_docstring(node)
            return doc
    return None


def _extract_signature(code: str) -> str:
    """Return the first def-line as `name(args)`."""
    m = re.search(r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\((.*?)\)\s*(?:->.*?)?:", code, flags=re.DOTALL)
    if not m:
        return ""
    return f"{m.group(1)}({m.group(2).strip()})"


def _call_args(sig: str) -> str:
    """Build a sample call-args string from a signature."""
    m = re.search(r"\((.*?)\)", sig, flags=re.DOTALL)
    if not m:
        return ""
    params = m.group(1)
    out = []
    for p in params.split(","):
        name = p.strip().split(":")[0].split("=")[0].strip()
        if name in {"self", "cls", ""}:
            continue
        out.append(f"<{name}>")
    return ", ".join(out)


def _parse_doc_io(doc: str) -> tuple[str, str]:
    """Extract Args/Returns blocks from a docstring."""
    if not doc:
        return "", ""
    args_block = ""
    ret_block = ""
    m = re.search(r"(?:Args|Inputs|Parameters):\s*\n(.*?)(?=\n\s*(?:Returns|Yields|Raises|$))",
                  doc, flags=re.DOTALL | re.IGNORECASE)
    if m:
        args_block = m.group(1).strip()
    m = re.search(r"Returns?:\s*\n?(.*?)(?=\n\s*(?:Raises|Examples?|$))",
                  doc, flags=re.DOTALL | re.IGNORECASE)
    if m:
        ret_block = m.group(1).strip()
    return args_block, ret_block


def _infer_stdlib_imports(code: str) -> list[str]:
    """Walk the AST and return all imported modules (best-effort)."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    out: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.extend(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            out.append(node.module.split(".")[0])
    return sorted(set(out))


# ── Internal: bundle quality scoring ─────────────────────────────────


def _eval_structure(bundle: SkillBundle) -> float:
    """All four pieces present and well-formed? 0--1."""
    score = 0.0
    checks = 0
    # Has skill_md
    checks += 1
    if bundle.skill_md and len(bundle.skill_md) > 50:
        score += 1
    # Has function implementation (parses)
    checks += 1
    try:
        ast.parse(bundle.tool.implementation)
        score += 1
    except SyntaxError:
        pass
    # Has test suite (if present, must parse)
    checks += 1
    if bundle.tool.test_suite:
        try:
            ast.parse(bundle.tool.test_suite)
            score += 1
        except SyntaxError:
            pass
    else:
        score += 0.5  # half credit: no tests is permissible but not preferred
    # Has metadata dict
    checks += 1
    if isinstance(bundle.metadata, dict) and bundle.metadata.get("name"):
        score += 1
    return score / checks if checks else 0.0


def _eval_doc(bundle: SkillBundle) -> float:
    """Quality of the SKILL.md: required sections present, non-trivial content."""
    md = bundle.skill_md or ""
    score = 0.0
    checks = 0
    required = ["## Description", "## Usage"]
    optional = ["## Inputs", "## Returns", "## Example"]
    for header in required:
        checks += 1
        if header in md:
            score += 1
    for header in optional:
        checks += 1
        if header in md:
            score += 1
    # Non-trivial Description body
    checks += 1
    desc_match = re.search(r"## Description\s*\n(.*?)(?=\n##|\Z)", md, flags=re.DOTALL)
    if desc_match and len(desc_match.group(1).strip()) > 30:
        score += 1
    return score / checks if checks else 0.0


def _eval_metadata(bundle: SkillBundle) -> float:
    """Metadata.json correctness: required fields present and sensible."""
    meta = bundle.metadata or {}
    score = 0.0
    checks = 0
    required_fields = ["name", "version", "created_at_task"]
    for f in required_fields:
        checks += 1
        if meta.get(f):
            score += 1
    # version is a positive int
    checks += 1
    v = meta.get("version")
    if isinstance(v, int) and v >= 1:
        score += 1
    # dependencies is a list (possibly empty)
    checks += 1
    if isinstance(meta.get("dependencies"), list):
        score += 1
    return score / checks if checks else 0.0


# ── Serialisation: write a SkillBundle to disk (Claude-Skills-shape) ──


def write_skill_bundle_to_disk(bundle: SkillBundle, bundle_dir: str) -> None:
    """Write the four files of a skill bundle to a directory.

    bundle_dir/
      SKILL.md
      function.py
      tests.py        (only if test_suite is non-empty)
      metadata.json
    """
    import os
    os.makedirs(bundle_dir, exist_ok=True)
    with open(os.path.join(bundle_dir, "SKILL.md"), "w") as f:
        f.write(bundle.skill_md)
    with open(os.path.join(bundle_dir, "function.py"), "w") as f:
        f.write(bundle.tool.implementation)
    if bundle.tool.test_suite:
        with open(os.path.join(bundle_dir, "tests.py"), "w") as f:
            f.write(bundle.tool.test_suite)
    with open(os.path.join(bundle_dir, "metadata.json"), "w") as f:
        json.dump(bundle.metadata, f, indent=2)
