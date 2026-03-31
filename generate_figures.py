"""Generate all paper figures from results_full/ directory."""

import json
import glob
import os
import matplotlib.pyplot as plt
import numpy as np


def load_results():
    """Load all aggregate results."""
    results = {}
    for f in sorted(glob.glob("results_full/*/aggregate.json")):
        with open(f) as fh:
            d = json.load(fh)
        key = f"{d['system']}/{d['model']}"
        results[key] = d
    return results


def fig1_system_comparison(results):
    """Bar chart comparing all systems on key metrics."""
    # Sort: no-evolution first, then alphabetical
    order = ["no-evolution", "arise", "evoskill", "oneshot"]
    models = ["sonnet", "haiku"]

    entries = []
    for sys_name in order:
        for model in models:
            key = f"{sys_name}/{model}"
            if key in results:
                label = f"{sys_name}\n({model})"
                entries.append((label, results[key]))

    if not entries:
        print("No results found for fig1")
        return

    labels = [e[0] for e in entries]
    metrics = {
        "Task Comp.": [e[1]["avg_task_completion"] * 100 for e in entries],
        "Tool Quality": [e[1]["avg_tool_quality"] * 100 for e in entries],
        "Reuse": [e[1]["avg_reuse_rate"] * 100 for e in entries],
        "Library Health": [e[1]["avg_library_health"] * 100 for e in entries],
    }

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(labels))
    width = 0.2
    colors = ["#60a5fa", "#86efac", "#fbbf24", "#c084fc"]

    for i, (name, vals) in enumerate(metrics.items()):
        bars = ax.bar(x + i * width, vals, width, label=name, color=colors[i],
                      edgecolor="white", linewidth=0.5)
        if name == "Tool Quality":
            for j, v in enumerate(vals):
                if v == 0:
                    ax.text(x[j] + i * width, 1.5, "N/A", ha="center",
                            va="bottom", fontsize=5, color="#6b7280", style="italic")

    ax.set_ylabel("Score (%)", fontsize=10)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(labels, fontsize=7)
    ax.legend(fontsize=8, ncol=2, loc="upper left")
    ax.set_ylim(0, 100)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("paper/fig_comparison.pdf", dpi=300, bbox_inches="tight")
    plt.savefig("paper/fig_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Generated fig_comparison")


def fig2_tool_quality(results):
    """Per-tool quality scores across all systems."""
    all_tools = []
    for key, r in results.items():
        for t in r.get("tool_details", []):
            all_tools.append({**t, "system": key})

    if not all_tools:
        print("No tool details for fig2")
        return

    # Sort by TQS
    all_tools.sort(key=lambda t: t["tqs"], reverse=True)

    # Keep underscores, use monospace-like short names
    names = [t["name"][:20] for t in all_tools]
    tqs = [t["tqs"] for t in all_tools]

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#86efac" if v >= 0.5 else "#f87171" for v in tqs]
    ax.bar(range(len(names)), tqs, color=colors, alpha=0.8, edgecolor="white", linewidth=0.5)
    ax.axhline(y=0.5, color="#fbbf24", linestyle="--", alpha=0.7, label="Quality threshold")
    ax.set_ylabel("Tool Quality Score", fontsize=10)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=6, rotation=60, ha="right")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("paper/fig_tool_quality.pdf", dpi=300, bbox_inches="tight")
    plt.savefig("paper/fig_tool_quality.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Generated fig_tool_quality")


def fig3_domain_comparison(results):
    """Grouped bar chart: performance by domain for each system."""
    # Need per-session data to group by domain
    systems_to_plot = []
    for key, r in results.items():
        if "sonnet" in key:
            systems_to_plot.append((key, r))

    if not systems_to_plot:
        print("No sonnet results for fig3")
        return

    domains = {"Data Transform": (0, 5), "API Orch.": (5, 6), "Numerical": (6, 9)}

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(domains))
    width = 0.15
    colors = ["#94a3b8", "#60a5fa", "#f59e0b", "#10b981"]

    for i, (key, r) in enumerate(systems_to_plot):
        sys_name = key.split("/")[0]
        domain_scores = []
        for dname, (start, end) in domains.items():
            sessions = r.get("per_session", []) if "per_session" in r else []
            # Read individual session files
            dir_path = f"results_full/{sys_name}_sonnet"
            scores = []
            for si in range(start + 1, end + 1):
                sf = os.path.join(dir_path, f"s{si}.json")
                if os.path.exists(sf):
                    with open(sf) as fh:
                        sd = json.load(fh)
                    scores.append(sd.get("task_completion", 0))
            domain_scores.append(np.mean(scores) * 100 if scores else 0)

        ax.bar(x + i * width, domain_scores, width, label=sys_name,
               color=colors[i % len(colors)], edgecolor="white", linewidth=0.5)

    ax.set_ylabel("Task Completion (%)", fontsize=10)
    ax.set_xticks(x + width * (len(systems_to_plot) - 1) / 2)
    ax.set_xticklabels(list(domains.keys()), fontsize=9)
    ax.legend(fontsize=8)
    ax.set_ylim(0, 100)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("paper/fig_domains.pdf", dpi=300, bbox_inches="tight")
    plt.savefig("paper/fig_domains.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Generated fig_domains")


def fig4_ets_composite(results):
    """Horizontal bar chart of ETS scores for all systems."""
    entries = sorted(results.items(), key=lambda x: x[1]["avg_evolvetool_score"])

    labels = [k.replace("/", "\n") for k, _ in entries]
    scores = [v["avg_evolvetool_score"] for _, v in entries]
    colors = ["#10b981" if s == max(scores) else "#60a5fa" for s in scores]

    fig, ax = plt.subplots(figsize=(6, 3))
    bars = ax.barh(range(len(labels)), scores, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("EvolveTool Score (ETS)", fontsize=10)
    ax.set_xlim(0, 1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.3)

    for bar, score in zip(bars, scores):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{score:.3f}", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig("paper/fig_ets.pdf", dpi=300, bbox_inches="tight")
    plt.savefig("paper/fig_ets.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Generated fig_ets")


def print_latex_table(results):
    """Print LaTeX table for paper."""
    print("\n% LaTeX table for paper:")
    print("\\begin{table*}[t]")
    print("\\centering\\footnotesize")
    print("\\begin{tabular}{@{}llccccccc@{}}")
    print("\\toprule")
    print("\\textbf{System} & \\textbf{Model} & \\textbf{ETS}$\\uparrow$ & \\textbf{TC} & \\textbf{TQS} & \\textbf{Tools} & \\textbf{Reuse} & \\textbf{LH} \\\\")
    print("\\midrule")

    for key in sorted(results.keys()):
        r = results[key]
        sys_name = r["system"].replace("_", "\\_")
        model = r["model"]
        bold = "\\textbf" if r["avg_evolvetool_score"] == max(v["avg_evolvetool_score"] for v in results.values()) else ""
        ets = f"{r['avg_evolvetool_score']:.3f}"
        if bold:
            ets = f"\\textbf{{{ets}}}"
        print(f"{sys_name} & {model} & {ets} & {r['avg_task_completion']*100:.1f} & {r['avg_tool_quality']*100:.1f} & {r['total_tools']} & {r['avg_reuse_rate']*100:.1f} & {r['avg_library_health']*100:.1f} \\\\")

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\caption{Full benchmark results across all systems, models, and domains (99 tasks per run).}")
    print("\\end{table*}")


if __name__ == "__main__":
    results = load_results()
    if not results:
        print("No results found in results_full/. Run benchmarks first.")
        exit(1)

    print(f"Found {len(results)} result sets")
    fig1_system_comparison(results)
    fig2_tool_quality(results)
    fig3_domain_comparison(results)
    fig4_ets_composite(results)
    print_latex_table(results)
