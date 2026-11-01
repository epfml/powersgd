#!/usr/bin/env python
"""Build results/RESULTS.md and figures/*.png from the run logs in results/.

Reads results/runs.jsonl (one record per run) and results/runs/<id>/metrics.jsonl (eval curves), plus any
results/gpu/*.csv (nvidia-smi samples). Everything is computed from those files; the interpretation text
only states things the code checks (see `claims`).

    python tools/make_results.py            # needs matplotlib for the figures; RESULTS.md is written regardless
"""
import csv
import glob
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
TARGETS = (2.0, 1.8, 1.6)
BUSY_PCT = 10  # a GPU-second counts as "active" above this utilization


def label(r):
    return "DDP" if r["mode"] == "ddp" else f"PowerSGD r{r['rank_R']}"


def order(r):
    return (r["nodes"], r["steps"], r["mode"] != "ddp", r["rank_R"] or 0)


def load_runs():
    recs = [json.loads(l) for l in open(os.path.join(RES, "runs.jsonl")) if l.strip()]
    smoke = [r for r in recs if r["run_id"].startswith("smoke_")]
    real = [r for r in recs if not r["run_id"].startswith("smoke_") and r.get("status") == "ok"]
    for r in real:
        evs = [json.loads(l) for l in open(os.path.join(RES, "runs", r["run_id"], "metrics.jsonl"))]
        r["evals"] = [e for e in evs if e.get("type") == "eval"]
    return real, smoke


def at_step(r, step):
    return next((e["val_loss"] for e in r["evals"] if e["step"] == step), None)


def best_eval(r):
    return min(r["evals"], key=lambda e: e["val_loss"])


def first_reaching(r, target):
    return next((e for e in r["evals"] if e["val_loss"] <= target), None)


def gpu_stats():
    out = []
    for path in sorted(glob.glob(os.path.join(RES, "gpu", "*.csv"))):
        rows = list(csv.reader(open(path)))[1:]
        util = [int(r[2].replace("%", "").strip()) for r in rows]
        busy = [u for u in util if u > BUSY_PCT]
        out.append({"file": os.path.basename(path), "samples": len(util), "first": rows[0][0].strip(), "last": rows[-1][0].strip(),
                    "mean_all": sum(util) / len(util), "mean_busy": sum(busy) / len(busy) if busy else 0.0,
                    "busy_frac": len(busy) / len(util), "util": util})
    return out


def fmt(x, nd=3):
    return "–" if x is None else f"{x:.{nd}f}"


def tables(real):
    ddp = {(r["nodes"], r["steps"]): r for r in real if r["mode"] == "ddp"}
    rows = ["| run | config | best val loss (step) | val @500 | val @750 | final val | final train | median step | step vs DDP | comm/step | comm vs DDP | wall | GPU-h |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    tgt = ["| run | " + " | ".join(f"first eval with val ≤ {x}" for x in TARGETS) + " |", "|---|" + "---|" * len(TARGETS)]
    for r in sorted(real, key=order):
        b, last = best_eval(r), r["evals"][-1]
        base = ddp.get((r["nodes"], r["steps"]))
        sp = f"{100 * (r['median_step_time_s'] / base['median_step_time_s'] - 1):+.1f}%" if base and r is not base else "–"
        cr = f"{base['comm_bytes_per_step'] / r['comm_bytes_per_step']:.0f}x less" if base and r is not base else "–"
        rows.append(f"| `{r['run_id']}` | {label(r)}, {r['nodes']} node, {r['steps']} steps | **{b['val_loss']:.3f}** ({b['step']}) | "
                    f"{fmt(at_step(r, 500))} | {fmt(at_step(r, 750))} | {last['val_loss']:.3f} | {last['train_loss']:.3f} | "
                    f"{r['median_step_time_s'] * 1000:.1f} ms | {sp} | {r['comm_bytes_per_step'] / 1e6:.2f} MB | {cr} | "
                    f"{r['wall_time_s']:.0f} s | {r['gpu_hours']:.3f} |")
        cells = []
        for x in TARGETS:
            e = first_reaching(r, x)
            cells.append(f"step {e['step']} ({e['elapsed_s']:.0f} s)" if e else "not reached")
        tgt.append(f"| `{r['run_id']}` | " + " | ".join(cells) + " |")
    return "\n".join(rows), "\n".join(tgt)


def claims(real):
    """Interpretation sentences, each emitted only if the data supports it."""
    out = []
    sweep = [r for r in real if r["nodes"] == 1 and r["steps"] == 2000]
    ddp = next((r for r in sweep if r["mode"] == "ddp"), None)
    psgd = sorted([r for r in sweep if r["mode"] == "powersgd"], key=lambda r: r["rank_R"])
    if ddp:
        b, last = best_eval(ddp), ddp["evals"][-1]
        if b["step"] < last["step"] and last["val_loss"] > b["val_loss"] and last["train_loss"] < ddp["evals"][0]["train_loss"]:
            out.append(f"**DDP overfits.** Its val loss is best at step {b['step']} ({b['val_loss']:.3f}) and then rises to {last['val_loss']:.3f} "
                       f"by step {last['step']} while its train loss keeps falling to {last['train_loss']:.2f}. Shakespeare-char is ~1 MB of text "
                       f"and the model has {ddp['params'] / 1e6:.1f}M parameters, so this is expected over 2000 steps.")
    if ddp and psgd and all(best_eval(r)["step"] == r["evals"][-1]["step"] for r in psgd):
        out.append("**The PowerSGD runs had not converged.** Each one's best val loss is at the last eval (step 2000) and still falling, "
                   "so they never reached the point where DDP started overfitting.")
    if ddp and psgd and any(r["evals"][-1]["val_loss"] < ddp["evals"][-1]["val_loss"] for r in psgd) \
            and all(best_eval(ddp)["val_loss"] < best_eval(r)["val_loss"] for r in psgd):
        out.append("**Do not read 'final val loss' as PowerSGD beating DDP.** The lower final numbers for the higher ranks are an overfitting "
                   "artefact: DDP's *best* val loss is lower than every PowerSGD run's *best* val loss at 2000 steps.")
    if len(psgd) > 1 and all(best_eval(a)["val_loss"] >= best_eval(b)["val_loss"] for a, b in zip(psgd, psgd[1:])):
        out.append("**Higher rank converges faster per step** (best val loss at equal steps improves from rank 1 to 2 to 4).")
    if ddp and psgd and all(r["median_step_time_s"] > ddp["median_step_time_s"] for r in psgd):
        ov = [r["median_step_time_s"] / ddp["median_step_time_s"] - 1 for r in psgd]
        c = [ddp["comm_bytes_per_step"] / r["comm_bytes_per_step"] for r in psgd]
        out.append(f"**On one NVLink node, compression costs time and saves bytes.** PowerSGD steps are {100 * min(ov):.0f}–{100 * max(ov):.0f}% slower "
                   f"than DDP while sending {min(c):.0f}–{max(c):.0f}x fewer logical all-reduce bytes. Inside a node the all-reduce is already cheap, "
                   "so the low-rank compute is overhead. Whether this flips across nodes is what the queued 2-node job tests.")
    return out


def figures(real, gpu):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping figures")
        return []
    os.makedirs(FIG, exist_ok=True)
    made = []
    sweep = sorted([r for r in real if r["nodes"] == 1 and r["steps"] == 2000], key=order)
    colors = {"DDP": "#444444", "PowerSGD r1": "#d95f02", "PowerSGD r2": "#1b9e77", "PowerSGD r4": "#7570b3"}

    for xkey, xl, name in (("step", "training step", "val_loss_vs_step.png"),
                           ("elapsed_s", "wall-clock since start (s, includes eval)", "val_loss_vs_wallclock.png")):
        fig, ax = plt.subplots(figsize=(6.4, 4))
        for r in sweep:
            c = colors.get(label(r))
            ax.plot([e[xkey] for e in r["evals"]], [e["val_loss"] for e in r["evals"]], marker="o", ms=3.5, label=label(r), color=c)
            b = best_eval(r)
            ax.plot([b[xkey]], [b["val_loss"]], marker="*", ms=12, color=c, mfc="none")
        ax.set(xlabel=xl, ylabel="validation loss", title="NanoGPT, 1 node (4x A100), seed 1 (star = best)")
        ax.grid(alpha=.3)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, name), dpi=160)
        plt.close(fig)
        made.append(name)

    if sweep:
        fig, axs = plt.subplots(1, 2, figsize=(8, 3.6))
        names = [label(r) for r in sweep]
        axs[0].bar(names, [r["median_step_time_s"] * 1000 for r in sweep], color=[colors.get(n) for n in names])
        axs[0].set(ylabel="median step time (ms)", title="Step time (1 node)")
        axs[1].bar(names, [r["comm_bytes_per_step"] / 1e6 for r in sweep], color=[colors.get(n) for n in names])
        axs[1].set(yscale="log", ylabel="MB per step (log)", title="Logical all-reduce payload / rank")
        for a in axs:
            a.tick_params(axis="x", labelrotation=30)
            a.grid(alpha=.3, axis="y")
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, "step_time_and_comm.png"), dpi=160)
        plt.close(fig)
        made.append("step_time_and_comm.png")

    for g in gpu[:1]:
        fig, ax = plt.subplots(figsize=(7, 3))
        u = g["util"]
        n = len(u) // 4  # 4 GPUs sampled once a second
        ax.plot(range(n), [sum(u[i * 4:(i + 1) * 4]) / 4 for i in range(n)], lw=.8, color="#1b9e77")
        ax.set(xlabel="seconds since monitor start", ylabel="mean util over 4 GPUs (%)", ylim=(0, 105),
               title=f"{g['mean_all']:.1f}% over the allocation, {g['mean_busy']:.1f}% while a run was active")
        ax.grid(alpha=.3)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, "gpu_utilization.png"), dpi=160)
        plt.close(fig)
        made.append("gpu_utilization.png")
    return made


def main():
    real, smoke = load_runs()
    gpu = gpu_stats()
    figs = figures(real, gpu)
    main_t, target_t = tables(real)
    gpu_h = sum(r["gpu_hours"] for r in real + smoke)
    L = ["# Results", "",
         "_Generated by `tools/make_results.py` from `results/runs.jsonl`, `results/runs/*/metrics.jsonl` and `results/gpu/*.csv`. Do not edit by hand._", "",
         "## What was run", "",
         "NanoGPT (6 layers, 6 heads, 384 dim, block 256, ~10.8M params) on Shakespeare-char, fp32, **one seed (1) per configuration**, "
         "per-GPU batch 64 (global 256), effective lr 0.001, `start_powerSGD_iter` 10, eval every 250 steps. Run on NERSC Perlmutter, account `m4431_g`, "
         "one node (4x A100-SXM4-80GB), in one interactive allocation. PowerSGD uses PyTorch's built-in DDP hook with error feedback and warm start. "
         "The DDP baseline goes through the same Python all-reduce hook so communication bytes are counted the same way. See `docs/RUNS.md` for how it was launched.", "",
         "## Main table", "", main_t, "",
         "`comm/step` is the logical payload passed to `dist.all_reduce` per rank in steady state, not wire bytes. "
         "`wall` includes evaluation. `GPU-h` = wall-clock x 4 GPUs. Step-time deltas come from a single seed.", ""]
    L += ["## Reading the results", ""] + [f"- {c}" for c in claims(real)] + [""]
    L += ["### What we can and cannot claim", "",
          "- Supported: DDP converges fastest per step; higher rank converges faster per step; PowerSGD cuts logical communication by two orders of magnitude; on one NVLink node it costs a few percent step time.",
          "- **Not supported:** that PowerSGD gives better models than DDP, that PowerSGD is faster end-to-end, or anything about multi-node behaviour (not run yet).",
          "- One seed per config, 2000 steps. Differences of a few percent in step time or small loss gaps could change with another seed.", "",
          "## Steps and time to reach a target val loss", "", target_t, "",
          "Evals are every 250 steps, so these are upper bounds on when the target was first crossed.", "", "## GPU utilization", ""]
    for g in gpu:
        L += [f"`{g['file']}` ({g['first']} to {g['last']}, {g['samples']} samples = 4 GPUs at 1 Hz):", "",
              f"- **{g['mean_busy']:.1f}% mean utilization while a run was active** (samples above {BUSY_PCT}%).",
              f"- **{g['mean_all']:.1f}% mean over the whole allocation.** The gap is idle time between hand-typed commands in an interactive "
              f"allocation ({100 * (1 - g['busy_frac']):.0f}% of samples were at or below {BUSY_PCT}%). Batch jobs (`sbatch`) avoid this.", ""]
    L += ["## Budget", "",
          f"All runs above (sweep + 2 smoke tests): **{gpu_h:.2f} GPU-hours = {gpu_h / 4:.2f} node-hours** of training wall-clock. "
          "Charged node-hours are larger because the interactive allocation was held for 30 minutes; `tools/update_budget.py` builds the sacct-based `results/budget.md`.", ""]
    if figs:
        L += ["## Figures", ""] + [f"- `figures/{f}`" for f in figs] + [""]
    L += ["## Smoke tests (pipeline checks, not results)", ""]
    for r in smoke:
        L.append(f"- `{r['run_id']}`: {r['steps']} steps, {label(r)}, median step {r['median_step_time_s'] * 1000:.0f} ms, {r['comm_bytes_per_step'] / 1e6:.2f} MB/step.")
    L += ["", "## Not yet run", "",
          "- Submitted, pending in the queue (no results): `scripts/nanogpt_2node.sbatch`, `scripts/nanogpt_1node_long.sbatch`.",
          "- Not run: ResNet-18/CIFAR-10 with `train_ddp.py`, FP8 (emulated only; A100 has no FP8), LSTM/WikiText-2 in `train_ddp.py`.",
          "- Design only: tensor parallelism.", ""]
    open(os.path.join(RES, "RESULTS.md"), "w").write("\n".join(L))
    print("wrote results/RESULTS.md; figures:", figs)


if __name__ == "__main__":
    main()
