#!/usr/bin/env python3
"""
Summarize runs produced by run_experiment.py and compare them with the PowerSGD paper.

Usage:
    python analyze_results.py $SCRATCH/powersgd/results            # print tables
    python analyze_results.py $SCRATCH/powersgd/results --plots    # also save PNG curves

Writes summary.md (and curves_*.png with --plots) into the results directory.
"""

import argparse
import json
import os
from collections import defaultdict

import numpy as np

# (test accuracy % or perplexity, MB sent per epoch, ms per batch) from Tables 1-4, 6, 7.
PAPER = {
    ("cifar", "SGD"): (94.3, 1023, 312),
    ("cifar", "Rank 1"): (93.6, 4, 229),
    ("cifar", "Rank 2"): (94.4, 8, 239),
    ("cifar", "Rank 4"): (94.5, 14, 260),
    ("cifar", "Rank 7"): (94.6, 24, 285),
    ("cifar", "Rank 2 (no warm start)"): (94.0, None, None),
    ("cifar", "Unbiased Rank 1"): (71.2, 3, None),
    ("cifar", "Unbiased Rank 2"): (75.9, 4, None),
    ("cifar", "Random Block (rank 7)"): (93.3, 24, 243),
    ("cifar", "Random Block (rank 2)"): (87.8, 8, 240),
    ("cifar", "Random K (rank 7)"): (94.0, 24, 540),
    ("cifar", "Random K (rank 2)"): (92.6, 8, 534),
    ("cifar", "Sign+Norm"): (93.9, 32, 429),
    ("cifar", "Top K (medium)"): (94.4, 32, 444),
    ("cifar", "Top K (high)"): (93.6, 8, 411),
    ("cifar", "Atomo (rank 2)"): (92.6, 113, 948),
    ("cifar", "Signum"): (93.6, 32, 301),
    ("lm", "SGD"): (91, 7730, 300),
    ("lm", "Rank 1"): (102, 25, 131),
    ("lm", "Rank 2"): (93, 38, 141),
    ("lm", "Rank 4"): (91, 64, 134),
    ("lm", "Signum"): (142, 242, 424),
}

TEST_METRIC = {"cifar": "last_accuracy", "lm": "last_perplexity"}


def load_run(run_dir):
    with open(os.path.join(run_dir, "config.json")) as fp:
        config = json.load(fp)
    task = config["task_name"]

    curve = []  # (epoch, test metric, wallclock seconds since first record)
    runavg_final = None
    bits, epochs, t0 = 0, 0, None
    metrics_path = os.path.join(run_dir, "metrics.jsonl")
    if os.path.exists(metrics_path):
        with open(metrics_path) as fp:
            for line in fp:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue  # partially written last line of a killed job
                t0 = record["time"] if t0 is None else t0
                if record.get("split") != "test":
                    continue
                value = record["value"] * (100 if task == "cifar" else 1)
                if record["name"] == TEST_METRIC[task]:
                    curve.append((record["epoch"], value, record["time"] - t0))
                    bits, epochs = record["bits"], record["epoch"]
                elif record["name"] == TEST_METRIC[task].replace("last_", "runavg_"):
                    runavg_final = value

    timers = {}
    timer_path = os.path.join(run_dir, "timer_summary.json")
    if os.path.exists(timer_path):
        with open(timer_path) as fp:
            timers = json.load(fp)

    def avg_ms(label):
        return 1000 * timers[label]["average_duration"] if label in timers else None

    return dict(
        dir=run_dir,
        task=task,
        preset=config["reducer_name"],
        seed=config["seed"],
        workers=config["world_size"],
        backend=config["distributed_backend"],
        target_epochs=config["num_epochs"],
        epochs=epochs,
        finished=os.path.exists(os.path.join(run_dir, "DONE")),
        final=curve[-1][1] if curve else None,
        runavg_final=runavg_final,
        mb_per_epoch=bits / 8 / 1e6 / epochs if epochs else None,
        batch_ms=avg_ms("batch"),
        reduce_ms=avg_ms("batch.reduce"),
        forward_ms=avg_ms("batch.forward"),
        backward_ms=avg_ms("batch.backward"),
        curve=curve,
    )


def find_runs(root):
    runs = []
    for dirpath, _, filenames in os.walk(root):
        if "config.json" in filenames:
            runs.append(load_run(dirpath))
    return runs


def fmt(x, digits=1):
    return "-" if x is None else f"{x:.{digits}f}"


def spread(values, digits=1):
    values = [v for v in values if v is not None]
    if not values:
        return "-"
    if len(values) == 1:
        return fmt(values[0], digits)
    return f"{np.mean(values):.{digits}f} [{min(values):.{digits}f}, {max(values):.{digits}f}]"


def mean_or_none(values):
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def accuracy_table(runs, group):
    """One row per preset: ours (mean [min, max] over seeds) next to the paper."""
    by_preset = defaultdict(list)
    for run in runs:
        by_preset[(run["task"], run["preset"], run["workers"], run["backend"])].append(run)

    task_label = {"cifar": "Test acc. (%)", "lm": "Test ppl."}
    lines = [f"## {group}", ""]
    for task in ("cifar", "lm"):
        keys = sorted(k for k in by_preset if k[0] == task)
        if not keys:
            continue
        sgd_ms = {
            (w, b): mean_or_none([r["batch_ms"] for r in by_preset[(task, "SGD", w, b)]])
            for (_, _, w, b) in keys
            if (task, "SGD", w, b) in by_preset
        }
        lines += [
            f"| Task | Algorithm | GPUs | Seeds (done) | Epochs | {task_label[task]} ours | paper "
            "| Running-avg model | MB/epoch ours | paper | ms/batch ours | vs SGD | paper "
            "| reduce ms |",
            "|" + "---|" * 14,
        ]
        for key in keys:
            _, preset, workers, backend = key
            group_runs = by_preset[key]
            paper = PAPER.get((task, preset), (None, None, None))
            batch_ms = mean_or_none([r["batch_ms"] for r in group_runs])
            baseline = sgd_ms.get((workers, backend))
            vs_sgd = (
                f"{100 * (batch_ms / baseline - 1):+.0f}%" if batch_ms and baseline else "-"
            )
            lines.append(
                "| "
                + " | ".join(
                    [
                        task,
                        preset,
                        f"{workers} ({backend})",
                        f"{len(group_runs)} ({sum(r['finished'] for r in group_runs)})",
                        fmt(min(r["epochs"] for r in group_runs), 0)
                        + f"/{group_runs[0]['target_epochs']}",
                        spread([r["final"] for r in group_runs]),
                        fmt(paper[0]),
                        # LanguageModelingTask.test evaluates the live model in training mode,
                        # so its "running-average" perplexity is not meaningful.
                        "n/a" if task == "lm" else spread([r["runavg_final"] for r in group_runs]),
                        fmt(mean_or_none([r["mb_per_epoch"] for r in group_runs]), 0),
                        fmt(paper[1], 0),
                        fmt(batch_ms),
                        vs_sgd,
                        fmt(paper[2], 0),
                        fmt(mean_or_none([r["reduce_ms"] for r in group_runs])),
                    ]
                )
                + " |"
            )
        lines.append("")
    return lines


def scaling_table(runs, group):
    """Figure 3 / Table 5: epoch-time speedup over 1-worker SGD with batch size growing with W."""
    lines = [
        f"## {group}",
        "",
        "Speedup = W x (1-worker SGD ms/batch) / (W-worker ms/batch), i.e. epoch time vs 1-GPU SGD.",
        "",
        "| Backend | Algorithm | GPUs | ms/batch | forward | backward | reduce | speedup |",
        "|---|---|---|---|---|---|---|---|",
    ]
    cells = defaultdict(list)
    for run in runs:
        cells[(run["backend"], run["preset"], run["workers"])].append(run)
    for backend, preset, workers in sorted(cells):
        cell = cells[(backend, preset, workers)]
        batch_ms = mean_or_none([r["batch_ms"] for r in cell])
        base = mean_or_none([r["batch_ms"] for r in cells.get((backend, "SGD", 1), [])])
        speedup = f"{workers * base / batch_ms:.2f}x" if base and batch_ms else "-"
        lines.append(
            f"| {backend} | {preset} | {workers} | {fmt(batch_ms)} "
            f"| {fmt(mean_or_none([r['forward_ms'] for r in cell]))} "
            f"| {fmt(mean_or_none([r['backward_ms'] for r in cell]))} "
            f"| {fmt(mean_or_none([r['reduce_ms'] for r in cell]))} | {speedup} |"
        )
    lines.append("")
    return lines


def per_run_table(runs, results_dir):
    """One row per run (every seed), with unrounded results and absolute timings."""
    lines = [
        "| Group | Task | Algorithm | Seed | GPUs | Epochs | Final test metric | MB/epoch "
        "| ms/batch | forward | backward | reduce | Run directory |",
        "|" + "---|" * 13,
    ]
    for run in sorted(runs, key=lambda r: (r["dir"], r["seed"])):
        lines.append(
            "| "
            + " | ".join(
                [
                    os.path.relpath(run["dir"], results_dir).split(os.sep)[0],
                    run["task"],
                    run["preset"],
                    str(run["seed"]),
                    f"{run['workers']} ({run['backend']})",
                    f"{fmt(run['epochs'], 0)}/{run['target_epochs']}",
                    fmt(run["final"], 2),
                    fmt(run["mb_per_epoch"], 1),
                    fmt(run["batch_ms"], 2),
                    fmt(run["forward_ms"], 2),
                    fmt(run["backward_ms"], 2),
                    fmt(run["reduce_ms"], 2),
                    os.path.relpath(run["dir"], results_dir),
                ]
            )
            + " |"
        )
    return lines


def plot_curves(runs, group, out_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for task in ("cifar", "lm"):
        task_runs = [r for r in runs if r["task"] == task and r["curve"]]
        if not task_runs:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        by_preset = defaultdict(list)
        for run in task_runs:
            by_preset[run["preset"]].append(run)
        for preset, preset_runs in sorted(by_preset.items()):
            n = min(len(r["curve"]) for r in preset_runs)
            epochs = np.array([c[0] for c in preset_runs[0]["curve"][:n]])
            values = np.array([[c[1] for c in r["curve"][:n]] for r in preset_runs])
            minutes = np.array([[c[2] / 60 for c in r["curve"][:n]] for r in preset_runs]).mean(0)
            for ax, x in zip(axes, (epochs, minutes)):
                (line,) = ax.plot(x, values.mean(0), label=preset)
                ax.fill_between(x, values.min(0), values.max(0), alpha=0.2, color=line.get_color())
        ylabel = "Test accuracy (%)" if task == "cifar" else "Test perplexity"
        # Zoom to the paper's axes (Figures 4 and 5) once all runs end inside them.
        paper_ylim = (85, 96) if task == "cifar" else (50, 300)
        finals = [r["curve"][-1][1] for r in task_runs]
        zoom = all(paper_ylim[0] <= v <= paper_ylim[1] for v in finals)
        for ax, xlabel in zip(axes, ("Epoch", "Training time (minutes)")):
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.3)
            if zoom:
                ax.set_ylim(*paper_ylim)
        axes[0].legend()
        fig.suptitle(f"{group} - {task} (shaded: min-max over seeds)")
        fig.tight_layout()
        path = os.path.join(out_dir, f"curves_{group}_{task}.png")
        fig.savefig(path, dpi=120)
        plt.close(fig)
        print(f"saved {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir")
    parser.add_argument("--plots", action="store_true")
    args = parser.parse_args()

    runs = find_runs(args.results_dir)
    if not runs:
        raise SystemExit(f"No runs found under {args.results_dir}")

    groups = defaultdict(list)
    for run in runs:
        group = os.path.relpath(run["dir"], args.results_dir).split(os.sep)[0]
        groups[group].append(run)

    lines = []
    for group in sorted(groups):
        if group.startswith("scaling"):
            lines += scaling_table(groups[group], group)
        else:
            lines += accuracy_table(groups[group], group)
        if args.plots and not group.startswith("scaling"):
            plot_curves(groups[group], group, args.results_dir)

    text = "\n".join(lines)
    print(text)
    with open(os.path.join(args.results_dir, "summary.md"), "w") as fp:
        fp.write(text + "\n")
    with open(os.path.join(args.results_dir, "per_run.md"), "w") as fp:
        fp.write("\n".join(per_run_table(runs, args.results_dir)) + "\n")
    print(f"wrote {os.path.join(args.results_dir, 'per_run.md')}")


if __name__ == "__main__":
    main()
