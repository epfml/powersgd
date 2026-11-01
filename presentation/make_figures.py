#!/usr/bin/env python3
"""
Slide figures for the PowerSGD reproduction, built from results/paper_reproduction/summary.md.

    python presentation/make_figures.py
"""

import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
SUMMARY = os.path.join(HERE, "..", "results", "paper_reproduction", "summary.md")
OUT = os.path.join(HERE, "figures")

OURS, PAPER = "#2E86AB", "#B8B8B8"
COMPUTE, COMM = "#8D99AE", "#E07A5F"
FASTER, SLOWER = "#3A9D5D", "#C8553D"
METHODS = ["SGD", "Rank 1", "Rank 2", "Rank 4"]
TASK_TITLES = {"cifar": "ResNet18 on CIFAR-10", "lm": "LSTM on WikiText-2"}
# Time per batch vs SGD as printed in Table 3 of the paper.
PAPER_SPEED = {"cifar": [-26, -23, -16], "lm": [-56, -53, -55]}

plt.rcParams.update(
    {
        "font.size": 13,
        "axes.titlesize": 15,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    }
)


def parse_value(cell):
    """'93.3 [93.1, 93.4]' -> (93.3, 93.1, 93.4); '94.1' -> (94.1, 94.1, 94.1)."""
    numbers = [float(x) for x in re.findall(r"-?\d+\.?\d*", cell)]
    return (numbers[0], numbers[1], numbers[2]) if len(numbers) == 3 else (numbers[0],) * 3


def load_summary():
    """{(task, method): row} for the 16-GPU groups."""
    rows, group = {}, None
    with open(SUMMARY) as fp:
        for line in fp:
            if line.startswith("## "):
                group = line[3:].strip()
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if group not in ("cifar16", "lm16") or len(cells) != 14 or cells[0] not in TASK_TITLES:
                continue
            mean, lo, hi = parse_value(cells[5])
            rows[(cells[0], cells[1])] = dict(
                mean=mean,
                lo=lo,
                hi=hi,
                paper=float(cells[6]),
                mb=float(cells[8]),
                mb_paper=float(cells[9]),
                ms=float(cells[10]),
                vs_sgd=float(cells[11].rstrip("%")) if cells[11] != "-" else 0.0,
                ms_paper=float(cells[12]),
                reduce_ms=float(cells[13]),
            )
    return rows


def save(fig, name):
    fig.tight_layout()
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


def bar_pairs(ax, x, ours, paper, ours_err=None, width=0.38):
    ax.bar([i - width / 2 for i in x], paper, width, color=PAPER, label="Paper (2019)")
    ax.bar(
        [i + width / 2 for i in x],
        ours,
        width,
        color=OURS,
        label="Ours (Perlmutter)",
        yerr=ours_err,
        capsize=4,
        error_kw=dict(lw=1.2),
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(METHODS)


def label_bars(ax, xs, values, fmt, offset):
    for x, v in zip(xs, values):
        ax.text(x, v + offset, fmt.format(v), ha="center", va="bottom", fontsize=11)


def fig_method():
    fig, ax = plt.subplots(figsize=(12, 3.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 3.6)
    ax.axis("off")
    # Gradient matrix M (n x m), P (n x r), Q^T (r x m)
    ax.add_patch(Rectangle((0.3, 0.4), 3.6, 2.4, color=OURS, alpha=0.85))
    ax.text(2.1, 1.6, "M", color="white", fontsize=30, ha="center", va="center", weight="bold")
    ax.text(2.1, 3.05, "layer gradient  n × m\n(e.g. 512 × 4608)", ha="center", fontsize=12)
    ax.text(4.45, 1.6, "≈", fontsize=34, ha="center", va="center")
    ax.add_patch(Rectangle((5.0, 0.4), 0.45, 2.4, color=COMM))
    ax.text(5.22, 1.6, "P", color="white", fontsize=20, ha="center", va="center", weight="bold")
    ax.text(5.22, 3.05, "n × r", ha="center", fontsize=12)
    ax.text(5.85, 1.6, "×", fontsize=26, ha="center", va="center")
    ax.add_patch(Rectangle((6.25, 2.35), 3.6, 0.45, color=COMM))
    ax.text(8.05, 2.57, "Qᵀ", color="white", fontsize=20, ha="center", va="center", weight="bold")
    ax.text(8.05, 3.05, "r × m", ha="center", fontsize=12)
    ax.text(
        6.25,
        1.55,
        "Rank r = 2 for this layer:\n"
        "send 2 × (512 + 4608) = 10,240 numbers\n"
        "instead of 512 × 4608 = 2,359,296  →  230× less",
        fontsize=13,
        va="center",
    )
    save(fig, "fig1_method.png")


def fig_quality(rows):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    x = range(len(METHODS))
    for ax, task in zip(axes, ("cifar", "lm")):
        r = [rows[(task, m)] for m in METHODS]
        ours = [v["mean"] for v in r]
        err = [[v["mean"] - v["lo"] for v in r], [v["hi"] - v["mean"] for v in r]]
        paper = [v["paper"] for v in r]
        bar_pairs(ax, x, ours, paper, ours_err=err)
        tops = [v["hi"] for v in r]
        if task == "cifar":
            ax.set_ylim(92, 95.2)
            ax.set_ylabel("Test accuracy (%)  ↑ better")
            for xi, top, v in zip([i + 0.19 for i in x], tops, ours):
                ax.text(xi, top + 0.05, f"{v:.1f}", ha="center", va="bottom", fontsize=11)
            label_bars(ax, [i - 0.19 for i in x], paper, "{:.1f}", 0.05)
        else:
            ax.set_ylim(80, 118)
            ax.set_ylabel("Test perplexity  ↓ better")
            for xi, top, v in zip([i + 0.19 for i in x], tops, ours):
                ax.text(xi, top + 0.5, f"{v:.1f}", ha="center", va="bottom", fontsize=11)
            label_bars(ax, [i - 0.19 for i in x], paper, "{:.0f}", 0.5)
        ax.set_title(TASK_TITLES[task])
    axes[0].legend(loc="upper left")
    fig.text(0.5, -0.02, "Error bars: min-max over 3 seeds", ha="center", fontsize=11, color="gray")
    save(fig, "fig2_quality.png")


def fig_gap(rows):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    compressed = METHODS[1:]
    x = range(len(compressed))
    for ax, task in zip(axes, ("cifar", "lm")):
        sgd = rows[(task, "SGD")]
        ours = [rows[(task, m)]["mean"] - sgd["mean"] for m in compressed]
        paper = [rows[(task, m)]["paper"] - sgd["paper"] for m in compressed]
        width = 0.38
        ax.bar([i - width / 2 for i in x], paper, width, color=PAPER, label="Paper (2019)")
        ax.bar([i + width / 2 for i in x], ours, width, color=OURS, label="Ours (Perlmutter)")
        ax.axhline(0, color="black", lw=1)
        ax.set_xticks(list(x))
        ax.set_xticklabels(compressed)
        unit = "pts" if task == "cifar" else ""
        for xs, vals in (([i - width / 2 for i in x], paper), ([i + width / 2 for i in x], ours)):
            for xi, v in zip(xs, vals):
                ax.text(
                    xi,
                    v + (0.03 if task == "cifar" else 0.3) * (1 if v >= 0 else -1),
                    f"{v:+.1f}{unit}",
                    ha="center",
                    va="bottom" if v >= 0 else "top",
                    fontsize=11,
                )
        if task == "cifar":
            ax.set_ylabel("Accuracy minus SGD (pts)  ↑ better")
            ax.set_ylim(-1.1, 0.7)
        else:
            ax.set_ylabel("Perplexity minus SGD  ↓ better")
            ax.set_ylim(-2.5, 14)
        ax.set_title(TASK_TITLES[task])
    axes[1].legend(loc="upper right")
    save(fig, "fig3_gap_to_sgd.png")


def fig_communication(rows):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    x = range(len(METHODS))
    for ax, task in zip(axes, ("cifar", "lm")):
        r = [rows[(task, m)] for m in METHODS]
        ours = [v["mb"] for v in r]
        paper = [v["mb_paper"] for v in r]
        bar_pairs(ax, x, ours, paper)
        ax.set_yscale("log")
        ax.set_ylim(1, max(ours + paper) * 6)
        for xs, vals in (([i - 0.19 for i in x], paper), ([i + 0.19 for i in x], ours)):
            for xi, v in zip(xs, vals):
                ax.text(xi, v * 1.15, f"{v:,.0f}", ha="center", va="bottom", fontsize=10)
        ax.set_ylabel("MB sent per GPU per epoch (log)")
        ax.set_title(TASK_TITLES[task])
    axes[0].legend(loc="upper right")
    save(fig, "fig4_communication.png")


def fig_speed(rows):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    compressed = METHODS[1:]
    x = range(len(compressed))
    width = 0.38
    for ax, task in zip(axes, ("cifar", "lm")):
        ours = [rows[(task, m)]["vs_sgd"] for m in compressed]
        paper = PAPER_SPEED[task]
        ax.bar([i - width / 2 for i in x], paper, width, color=PAPER, label="Paper (2019)")
        ax.bar(
            [i + width / 2 for i in x],
            ours,
            width,
            color=[FASTER if v < 0 else SLOWER for v in ours],
            label="Ours (Perlmutter)",
        )
        ax.axhline(0, color="black", lw=1)
        for xs, vals in (([i - width / 2 for i in x], paper), ([i + width / 2 for i in x], ours)):
            for xi, v in zip(xs, vals):
                ax.text(
                    xi,
                    v + (1.5 if v >= 0 else -1.5),
                    f"{v:+.0f}%",
                    ha="center",
                    va="bottom" if v >= 0 else "top",
                    fontsize=11,
                )
        ax.set_xticks(list(x))
        ax.set_xticklabels(compressed)
        ax.set_ylim(-68, 38)
        ax.set_ylabel("Time per batch vs SGD  ↓ faster")
        ax.set_title(TASK_TITLES[task])
    handles = [
        Rectangle((0, 0), 1, 1, color=PAPER),
        Rectangle((0, 0), 1, 1, color=FASTER),
        Rectangle((0, 0), 1, 1, color=SLOWER),
    ]
    axes[1].legend(
        handles, ["Paper (2019)", "Ours: faster", "Ours: slower"], loc="upper right"
    )
    save(fig, "fig5_speed.png")


def fig_breakdown(rows):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    x = range(len(METHODS))
    for ax, task in zip(axes, ("cifar", "lm")):
        r = [rows[(task, m)] for m in METHODS]
        reduce_ms = [v["reduce_ms"] for v in r]
        other_ms = [v["ms"] - v["reduce_ms"] for v in r]
        ax.bar(x, other_ms, 0.6, color=COMPUTE, label="Forward, backward, optimizer")
        ax.bar(x, reduce_ms, 0.6, bottom=other_ms, color=COMM, label="Compression + communication")
        for xi, o, red, total in zip(x, other_ms, reduce_ms, [v["ms"] for v in r]):
            ax.text(xi, o + red / 2, f"{red:.1f} ms", ha="center", va="center", color="white",
                    fontsize=11, weight="bold")
            ax.text(xi, total + 0.5, f"{total:.1f} ms", ha="center", va="bottom", fontsize=11)
        ax.set_xticks(list(x))
        ax.set_xticklabels(METHODS)
        ax.set_ylim(0, 38)
        ax.set_ylabel("Time per batch (ms)")
        ax.set_title(TASK_TITLES[task])
    axes[0].legend(loc="upper left", fontsize=11)
    save(fig, "fig6_breakdown.png")


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = load_summary()
    missing = [(t, m) for t in TASK_TITLES for m in METHODS if (t, m) not in rows]
    if missing:
        raise SystemExit(f"Missing rows in {SUMMARY}: {missing}")
    fig_method()
    fig_quality(rows)
    fig_gap(rows)
    fig_communication(rows)
    fig_speed(rows)
    fig_breakdown(rows)


if __name__ == "__main__":
    main()
