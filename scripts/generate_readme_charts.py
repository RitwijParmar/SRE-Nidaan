#!/usr/bin/env python3
"""
Generate README chart assets for SRE-Nidaan.

Outputs PNG charts into assets/readme/ using real project data from:
  - data/sre_nidaan_dataset.json
  - results/final_evaluation_report.json
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "data" / "sre_nidaan_dataset.json"
EVAL_REPORT_PATH = ROOT / "results" / "final_evaluation_report.json"
OUT_DIR = ROOT / "assets" / "readme"

COLORS = {
    "ink": "#102a43",
    "muted": "#486581",
    "accent": "#0f766e",
    "accent_soft": "#99f6e4",
    "danger": "#dc2626",
    "warning": "#d97706",
    "ok": "#16a34a",
    "card": "#f8fbff",
    "grid": "#d9e2ec",
}


def _setup_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 160,
            "font.size": 11,
            "axes.titlesize": 16,
            "axes.labelsize": 11,
            "axes.edgecolor": COLORS["grid"],
            "axes.labelcolor": COLORS["muted"],
            "axes.titleweight": "semibold",
            "xtick.color": COLORS["muted"],
            "ytick.color": COLORS["muted"],
            "text.color": COLORS["ink"],
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "grid.color": COLORS["grid"],
        }
    )


def _load_dataset() -> list[dict]:
    return json.loads(DATASET_PATH.read_text(encoding="utf-8"))


def _load_eval_report() -> dict:
    return json.loads(EVAL_REPORT_PATH.read_text(encoding="utf-8"))


def _save(fig: plt.Figure, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / name
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path.relative_to(ROOT)}")


def generate_dataset_domain_distribution(dataset: list[dict]) -> None:
    domain_counts = Counter(str(item.get("domain", "unknown")) for item in dataset)
    ordered = sorted(domain_counts.items(), key=lambda kv: kv[1], reverse=True)
    domains = [name.replace("_", " ").title() for name, _ in ordered]
    counts = [count for _, count in ordered]

    fig, ax = plt.subplots(figsize=(12.5, 6.8))
    bars = ax.barh(domains, counts, color=COLORS["accent"], alpha=0.9)
    ax.invert_yaxis()
    ax.set_xlabel("Number of training incidents")
    ax.set_title("Dataset Domain Distribution (2,500 incidents)")
    ax.grid(axis="x", linestyle="-", linewidth=0.8, alpha=0.65)
    ax.set_axisbelow(True)

    for bar, value in zip(bars, counts):
        ax.text(
            value + 4,
            bar.get_y() + bar.get_height() / 2,
            str(value),
            va="center",
            ha="left",
            fontsize=10,
            color=COLORS["ink"],
        )

    _save(fig, "dataset_domain_distribution.png")


def generate_dataset_pearl_mix(dataset: list[dict]) -> None:
    pearl_counts = Counter(int(item.get("pearl_level", 0)) for item in dataset)
    labels = ["L1 Association", "L2 Intervention", "L3 Counterfactual"]
    values = [pearl_counts.get(1, 0), pearl_counts.get(2, 0), pearl_counts.get(3, 0)]
    colors = ["#38bdf8", "#0ea5e9", "#0369a1"]

    fig, ax = plt.subplots(figsize=(10.8, 5.8))
    wedges, texts, autotexts = ax.pie(
        values,
        labels=labels,
        colors=colors,
        autopct=lambda pct: f"{pct:.1f}%",
        startangle=95,
        wedgeprops={"linewidth": 1.2, "edgecolor": "white"},
        textprops={"color": COLORS["ink"], "fontsize": 11},
    )
    for autotext in autotexts:
        autotext.set_color("white")
        autotext.set_fontweight("bold")
        autotext.set_fontsize(10)
    ax.set_title("Pearl Level Mix in Training Dataset")

    _save(fig, "dataset_pearl_level_mix.png")


def generate_evaluation_category_scores(report: dict) -> None:
    category_scores = report.get("category_scores", {})
    labels = [
        "L1 Association",
        "L2 Intervention",
        "L3 Counterfactual",
    ]
    values = [
        float(category_scores.get("L1_Association", 0.0)),
        float(category_scores.get("L2_Intervention", 0.0)),
        float(category_scores.get("L3_Counterfactual", 0.0)),
    ]

    fig, ax = plt.subplots(figsize=(10.8, 5.8))
    bars = ax.bar(labels, values, color=["#60a5fa", "#3b82f6", "#1d4ed8"], width=0.56)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score (0-1)")
    ax.set_title("Final Evaluation: Category Scores")
    ax.grid(axis="y", linestyle="-", linewidth=0.8, alpha=0.65)
    ax.set_axisbelow(True)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.02,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="semibold",
            color=COLORS["ink"],
        )

    _save(fig, "evaluation_category_scores.png")


def generate_evaluation_domain_scores(report: dict) -> None:
    detailed = report.get("detailed_results", [])
    by_domain: dict[str, list[float]] = defaultdict(list)
    for row in detailed:
        domain = str(row.get("domain", "unknown"))
        by_domain[domain].append(float(row.get("composite_score", 0.0)))

    ordered = sorted(
        ((domain, sum(values) / max(1, len(values))) for domain, values in by_domain.items()),
        key=lambda kv: kv[1],
        reverse=True,
    )
    domains = [name.replace("_", " ").title() for name, _ in ordered]
    scores = [score for _, score in ordered]

    fig, ax = plt.subplots(figsize=(12.5, 6.8))
    bars = ax.barh(domains, scores, color="#0284c7")
    ax.invert_yaxis()
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Average composite score")
    ax.set_title("Final Evaluation: Domain-wise Average Scores")
    ax.grid(axis="x", linestyle="-", linewidth=0.8, alpha=0.65)
    ax.set_axisbelow(True)

    for bar, value in zip(bars, scores):
        ax.text(
            value + 0.012,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va="center",
            ha="left",
            fontsize=10,
            color=COLORS["ink"],
        )

    _save(fig, "evaluation_domain_scores.png")


def generate_quality_signals(report: dict) -> None:
    detailed = report.get("detailed_results", [])
    overall = float(report.get("overall_score", 0.0))
    safety = float(report.get("safety_compliance_rate", 0.0))
    dag_rate = (
        sum(1 for row in detailed if bool(row.get("has_dag_structure")))
        / max(1, len(detailed))
    )

    labels = ["Overall Score", "Safety Compliance", "DAG Presence"]
    values = [overall, safety, dag_rate]
    colors = [COLORS["accent"], COLORS["danger"], COLORS["warning"]]

    fig, ax = plt.subplots(figsize=(10.8, 5.4))
    bars = ax.bar(labels, values, color=colors, width=0.5)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Rate (0-1)")
    ax.set_title("Quality Signals from Final Evaluation Report")
    ax.grid(axis="y", linestyle="-", linewidth=0.8, alpha=0.65)
    ax.set_axisbelow(True)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.02,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="semibold",
            color=COLORS["ink"],
        )

    _save(fig, "evaluation_quality_signals.png")


def generate_architecture_diagram() -> None:
    fig, ax = plt.subplots(figsize=(13.2, 7.3))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    outer = FancyBboxPatch(
        (3, 8),
        94,
        84,
        boxstyle="round,pad=0.8,rounding_size=2.8",
        linewidth=1.2,
        edgecolor=COLORS["grid"],
        facecolor=COLORS["card"],
    )
    ax.add_patch(outer)
    ax.text(
        50,
        87,
        "SRE-Nidaan Split-Compute Architecture",
        ha="center",
        va="center",
        fontsize=17,
        color=COLORS["ink"],
        fontweight="semibold",
    )

    def box(x: float, y: float, w: float, h: float, title: str, lines: list[str], color: str) -> None:
        rect = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.4,rounding_size=2.0",
            linewidth=1.2,
            edgecolor=color,
            facecolor="white",
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h - 8, title, ha="center", va="center", fontsize=13, fontweight="bold")
        line_y = y + h - 16
        for line in lines:
            ax.text(x + 3, line_y, line, ha="left", va="center", fontsize=10.3, color=COLORS["muted"])
            line_y -= 7.2

    box(
        8,
        44,
        24,
        34,
        "The Face",
        [
            "Next.js 14 + React",
            "Causal graph workspace",
            "Safety approval controls",
            "Analyst feedback loop",
            "Port 3000",
        ],
        "#0ea5e9",
    )
    box(
        38,
        44,
        24,
        34,
        "The Body",
        [
            "FastAPI orchestration",
            "MCP-style tool routing",
            "Grounding + candidate verifier",
            "Auth/tenant middleware",
            "Port 8001",
        ],
        "#0f766e",
    )
    box(
        68,
        44,
        24,
        34,
        "The Brain",
        [
            "vLLM OpenAI-compatible API",
            "Meta-Llama-3-8B-Instruct",
            "LoRA production adapter",
            "Guided JSON decoding",
            "Port 8000",
        ],
        "#1d4ed8",
    )

    def arrow(x1: float, y1: float, x2: float, y2: float) -> None:
        ax.add_patch(
            FancyArrowPatch(
                (x1, y1),
                (x2, y2),
                arrowstyle="-|>",
                mutation_scale=14,
                linewidth=1.8,
                color=COLORS["muted"],
            )
        )

    arrow(32, 61, 38, 61)
    arrow(62, 61, 68, 61)
    arrow(38, 55, 32, 55)
    arrow(68, 55, 62, 55)

    ax.text(50, 34, "Operational Flow", ha="center", va="center", fontsize=13, fontweight="semibold")
    flow_steps = [
        "1) Face submits incident summary + telemetry context",
        "2) Body retrieves grounding evidence + scores candidates",
        "3) Brain returns strict causal JSON (DAG + intervention)",
        "4) Body enforces human approval gate and persists feedback",
    ]
    y = 28
    for step in flow_steps:
        ax.text(8, y, step, ha="left", va="center", fontsize=10.8, color=COLORS["ink"])
        y -= 5.6

    _save(fig, "architecture_split_compute.png")


def generate_training_runtime_profile() -> None:
    phases = [
        "Dataset",
        "SFT",
        "Reward Model",
        "RLHF",
        "Evaluation",
    ]
    # Approximate single-run durations in minutes from project runbooks.
    mins = [1, 180, 45, 60, 10]
    colors = ["#94a3b8", "#0ea5e9", "#2563eb", "#1d4ed8", "#334155"]

    fig, ax = plt.subplots(figsize=(11.8, 5.6))
    bars = ax.bar(phases, mins, color=colors, width=0.58)
    ax.set_ylabel("Approx. duration (minutes)")
    ax.set_title("Training Runtime Profile (Typical GPU Run)")
    ax.grid(axis="y", linestyle="-", linewidth=0.8, alpha=0.65)
    ax.set_axisbelow(True)

    for bar, minute in zip(bars, mins):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            minute + 4,
            f"{minute}m",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="semibold",
            color=COLORS["ink"],
        )

    _save(fig, "training_runtime_profile.png")


def main() -> None:
    _setup_matplotlib()
    dataset = _load_dataset()
    report = _load_eval_report()

    generate_architecture_diagram()
    generate_dataset_domain_distribution(dataset)
    generate_dataset_pearl_mix(dataset)
    generate_evaluation_category_scores(report)
    generate_evaluation_domain_scores(report)
    generate_quality_signals(report)
    generate_training_runtime_profile()
    print("README chart asset generation complete.")


if __name__ == "__main__":
    main()
