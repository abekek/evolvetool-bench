"""Generate an appendix-style artifact gallery from preserved tool source.

For each preserved-tool system (Code-Evol/Sonnet v2 and CREATOR-style/Sonnet),
emit a long-form gallery LaTeX block that:
  - selects ~8-10 representative tools spanning the quality spectrum
  - groups them by category (correct exemplars, fragile-high-Q-low-C tools,
    duplicates across sessions, low-TQS trivial ones)
  - shows each tool's full signature, full docstring, ~8 body lines, and
    per-dimension scores

The output is meant to live in the paper's appendix, not in the §5 body. Keep
the body summary tight and let readers go to the appendix for the catalogue.

Outputs:
  paper/v2/appendix_gallery.tex            — one combined appendix
  paper/kdd2026/appendix_gallery.tex       — same
"""

from __future__ import annotations

import ast
import json
import os
import glob
import re
import textwrap
from collections import defaultdict
from typing import Iterable


# ── load + select ─────────────────────────────────────────────────────


def load_tools(results_dir: str) -> list[dict]:
    tools = []
    for meta_path in glob.glob(os.path.join(results_dir, "tools", "*", "*.meta.json")):
        with open(meta_path) as f:
            meta = json.load(f)
        source_path = meta_path.replace(".meta.json", ".py")
        if not os.path.exists(source_path):
            continue
        source = open(source_path).read()
        session = os.path.basename(os.path.dirname(meta_path))
        tools.append({
            "name": meta["name"],
            "session": session,
            "source": source,
            "lines": len(source.splitlines()),
            **meta["scores"],
        })
    return tools


def categorise(tools: list[dict]) -> dict[str, list[dict]]:
    """Bucket tools into categories. A tool may appear in multiple categories."""
    name_counts: dict[str, list] = defaultdict(list)
    for t in tools:
        name_counts[t["name"]].append(t)

    cats: dict[str, list[dict]] = defaultdict(list)
    for t in tools:
        if t["correctness"] >= 0.5 and t["code_quality"] >= 0.7:
            cats["correct"].append(t)
        if t["correctness"] < 0.5 and t["code_quality"] >= 0.7:
            cats["fragile"].append(t)
        if t["tqs"] < 0.3:
            cats["trivial"].append(t)

    for name, instances in name_counts.items():
        if len(instances) >= 2:
            cats["duplicate"].append(instances[0])  # representative

    return cats


def pick_for_appendix(tools: list[dict], k_per_category: int = 2) -> list[tuple[str, dict]]:
    """Return up to k tools from each category, ordered to tell a story."""
    cats = categorise(tools)
    ordered: list[tuple[str, dict]] = []

    # Best correct tools (highest C, then Q)
    for t in sorted(cats.get("correct", []),
                     key=lambda t: (-t["correctness"], -t["code_quality"]))[:k_per_category]:
        ordered.append(("correct", t))

    # Worst fragile tools (lowest C among high-Q)
    seen = {id(t) for _, t in ordered}
    for t in sorted(cats.get("fragile", []),
                     key=lambda t: (t["correctness"], -t["code_quality"])):
        if id(t) in seen:
            continue
        ordered.append(("fragile", t))
        seen.add(id(t))
        if sum(1 for c, _ in ordered if c == "fragile") >= k_per_category:
            break

    # One duplicate
    for t in cats.get("duplicate", [])[:1]:
        if id(t) in seen:
            continue
        ordered.append(("duplicate", t))
        seen.add(id(t))

    # Trivial / worst-quality
    for t in sorted(cats.get("trivial", []), key=lambda t: t["tqs"])[:k_per_category]:
        if id(t) in seen:
            continue
        ordered.append(("trivial", t))
        seen.add(id(t))

    return ordered


# ── source extraction ────────────────────────────────────────────────


def _signature(code: str) -> str:
    m = re.search(r"def\s+[a-zA-Z_][a-zA-Z0-9_]*\s*\(.*?\)\s*(?:->\s*.*?)?\s*:",
                  code, re.DOTALL)
    return re.sub(r"\s+", " ", m.group(0)).strip() if m else ""


def _full_docstring(code: str) -> str:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return ""
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            return (ast.get_docstring(node) or "").strip()
    return ""


def _body_excerpt(code: str, max_lines: int = 8) -> str:
    """Return the body (after docstring), trimmed to max_lines significant lines."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code.splitlines()[:max_lines]
    # Find the function body's start line after the docstring node.
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            body = node.body
            # Skip the docstring if present
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                body = body[1:]
            if not body:
                return ""
            start = body[0].lineno - 1
            end = body[-1].end_lineno
            lines = code.splitlines()[start:end]
            if len(lines) <= max_lines:
                return "\n".join(lines)
            # Take first (max_lines-2) + ... + last 2
            head = lines[:max_lines - 2]
            tail = lines[-2:]
            return "\n".join(head + ["    # ..."] + tail)
    return ""


# ── LaTeX rendering ──────────────────────────────────────────────────


def _escape_tex_text(s: str) -> str:
    """Escape for prose / docstring text (not code listings)."""
    return (s.replace("\\", r"\textbackslash{}")
             .replace("&", r"\&").replace("%", r"\%")
             .replace("$", r"\$").replace("#", r"\#")
             .replace("_", r"\_").replace("{", r"\{")
             .replace("}", r"\}").replace("^", r"\textasciicircum{}")
             .replace("~", r"\textasciitilde{}"))


CATEGORY_BLURB = {
    "correct":   ("\\textbf{Correct exemplar}",
                  "passes hidden tests with $C{\\geq}0.5$"),
    "fragile":   ("\\textbf{Fragile}",
                  "high $Q$, low $C$ -- looks well-formed, fails on hidden inputs"),
    "duplicate": ("\\textbf{Duplicate}",
                  "synthesised across multiple sessions"),
    "trivial":   ("\\textbf{Trivial}",
                  "low overall TQS"),
}


def _render_one_tool(category: str, t: dict) -> str:
    sig = _signature(t["source"])
    doc = _full_docstring(t["source"])
    body = _body_excerpt(t["source"], max_lines=8)

    cat_label, cat_blurb = CATEGORY_BLURB[category]
    name_safe = _escape_tex_text(t["name"])
    session_safe = _escape_tex_text(t["session"])
    sig_safe = _escape_tex_text(sig)
    doc_safe = _escape_tex_text(doc)

    # Score line: bold each value >= 0.5
    def _b(v: float) -> str:
        return f"\\textbf{{{v:.2f}}}" if v >= 0.5 else f"{v:.2f}"

    scores = (f"$C{{=}}{_b(t['correctness'])}$\\;"
              f"$R{{=}}{_b(t['robustness'])}$\\;"
              f"$G{{=}}{_b(t['generality'])}$\\;"
              f"$Q{{=}}{_b(t['code_quality'])}$\\;"
              f"$\\text{{TQS}}{{=}}{_b(t['tqs'])}$")

    # Code body: lstlisting with line-wrapping so long signatures/URLs don't
    # bleed across columns.
    code_listing = (
        "\\begin{lstlisting}[basicstyle=\\scriptsize\\ttfamily,"
        "language=Python,frame=none,xleftmargin=1em,breaklines=true,"
        "postbreak=\\mbox{\\textcolor{gray}{$\\hookrightarrow$}\\space},"
        "showstringspaces=false]\n"
        + sig + "\n"
        + (("    \"\"\"" + doc.split("\n\n")[0].strip() + "\"\"\"\n") if doc else "")
        + body
        + "\n\\end{lstlisting}"
    )

    # Use \linewidth, not \textwidth — in 2-column documents \textwidth is
    # the whole page and the minipage overflows the column gutter.
    return textwrap.dedent(rf"""
    \noindent {cat_label} \textit{{({cat_blurb})}}\\
    \texttt{{\textbf{{{name_safe}}}}} \footnotesize\textit{{({session_safe})}}\quad {scores}
    {code_listing}
    \vspace{{0.8em}}
    """).strip()


def _render_system_section(system_label: str, tools: list[dict]) -> str:
    """Render one system's section in the appendix."""
    picks = pick_for_appendix(tools, k_per_category=2)
    blocks = [_render_one_tool(cat, t) for cat, t in picks]

    return textwrap.dedent(rf"""
    \subsection{{{system_label}}}

    {len(tools)} tools were synthesised and preserved on disk across 9 sessions.
    Below we show {len(picks)} representative tools spanning the quality
    spectrum: correct exemplars, fragile high-$Q$/low-$C$ tools, duplicates,
    and trivial (low-TQS) cases.

    """).strip() + "\n\n" + "\n\n".join(blocks)


def main() -> None:
    sources = [
        ("results_full/arise_sonnet_v2",       "Code-Evol/Sonnet (semantic $Q$)"),
        ("results_full/creator-style_sonnet",  "CREATOR-style/Sonnet"),
    ]
    sections = []
    for results_dir, label in sources:
        if not os.path.isdir(results_dir):
            continue
        tools = load_tools(results_dir)
        if not tools:
            continue
        print(f"{results_dir}: {len(tools)} preserved tools")
        sections.append(_render_system_section(label, tools))

    appendix = textwrap.dedent(r"""
    \onecolumn
    \appendix
    \section{Artifact Gallery}
    \label{app:artifact_gallery}

    The benchmark preserves every synthesised tool to disk under
    \texttt{results\_full/<config>/tools/<session>/<name>.py} (plus a
    \texttt{.meta.json} with the per-tool TQS scores). This appendix presents
    a side-by-side gallery of representative tools from the two
    tool-creating systems with multi-seed preserved source --- Code-Evol
    (semantic-$Q$ run) and CREATOR-style --- to make the per-tool failure
    modes the benchmark surfaces concrete. Per-dimension scores are shown
    inline ($C$, $R$, $G$, $Q$ and the composite TQS); bold indicates
    $\geq 0.5$.

    """).strip() + "\n\n" + "\n\n".join(sections) + "\n"

    for paper in ["paper/v2", "paper/kdd2026"]:
        out = os.path.join(paper, "appendix_gallery.tex")
        with open(out, "w") as f:
            f.write(appendix)
        print(f"  wrote {out}")

    # Also keep the previous compact in-body gallery files as no-ops so
    # \input{gallery_arise_sonnet_v2.tex} in main.tex still resolves.
    # We empty them out so the body doesn't render the old gallery; the body
    # text already points readers to the appendix.
    for paper in ["paper/v2", "paper/kdd2026"]:
        for fname in ["gallery_arise_sonnet_v2.tex", "gallery_creator-style_sonnet.tex"]:
            p = os.path.join(paper, fname)
            with open(p, "w") as f:
                f.write("% deprecated: see appendix_gallery.tex\n")


if __name__ == "__main__":
    main()
