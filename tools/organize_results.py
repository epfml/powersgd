#!/usr/bin/env python
"""Organize raw scp copies of Perlmutter results into <out>/organized/ and write <out>/MANIFEST.md.

COPY ONLY: raw files are never modified, moved or deleted. Everything is derived from the files themselves
(config.json, metrics.jsonl, step_times.csv, runs.jsonl, logs*/ .out files, gpu_*.csv). The hand-typed provenance below
(STATED_RUNS, STATED_JOBS) comes from Aaron and is only used to cross-check the files; disagreements are reported.

    python tools/organize_results.py --raw powersgd_results --out results

Raw layout: every subfolder with a runs.jsonl is a snapshot of results/ (ordered by file mtime, newest wins); every
subfolder named logs* holds .out logs and gpu_<job>_<node>.csv. Output in <out>/organized is rebuilt from scratch.
Doubled metrics.jsonl files (two runs wrote one folder) are split per job only when each block is attributed by exact
match (a) to the eval lines in a batch job's .out and/or (b) to a runs.jsonl record's final val loss; otherwise the block
stays unattributed and nothing is guessed.
"""
import argparse
import csv
import glob
import hashlib
import json
import os
import re
import shutil
from collections import defaultdict

BUSY_PCT = 10  # a GPU-second counts as "active" above this utilization
RUN_FILES = ("config.json", "metrics.jsonl", "step_times.csv")
GPUS_PER_NODE = 4

# Stated by Aaron (not derived from files). run_id -> (job, nodes, world_size, seed, global_batch, steps)
S1 = ("60622124", 1, 4, 1, 256)
STATED_RUNS = {
    "smoke_gpt_ddp": S1 + (50,), "smoke_gpt_r2": S1 + (50,),
    "gpt_ddp_r0_1node": S1 + (2000,), "gpt_powersgd_r1_1node": S1 + (2000,),
    "gpt_powersgd_r2_1node": S1 + (2000,), "gpt_powersgd_r4_1node": S1 + (2000,),
    "rerun_ddp_s1": ("60626982", 1, 4, 1, 256, 2000), "rerun_r4_s1": ("60626982", 1, 4, 1, 256, 2000),
    "gpt_ddp_1000_1node": ("60623330", 1, 4, 1, 256, 1000), "gpt_r2_6000_1node": ("60623330", 1, 4, 1, 256, 6000),
    "gpt_r4_6000_1node": ("60623330", 1, 4, 1, 256, 6000),
}
TWO_NODE_JOBS = ("60627319", "60623329")  # both stated to have written gpt_ddp_2node / gpt_r2_2node / gpt_r4_2node
PROVISIONAL = {"60623330": "from `squeue` at COMPLETING, not the final sacct figure; run `sacct -j 60623330 -X --format=JobID,Elapsed,State` to confirm"}
# job -> (nodes, elapsed H:MM:SS or None, state, kind); from sacct as relayed by Aaron; 60623330's elapsed is not stated
STATED_JOBS = {
    "60622124": (1, "0:30:02", "TIMEOUT", "interactive"), "60623329": (2, "0:09:59", "COMPLETED", "sbatch"),
    "60623330": (1, "0:19:58", "COMPLETED", "sbatch"), "60626982": (1, "0:10:42", "COMPLETED", "interactive"),
    "60627319": (2, "0:13:39", "COMPLETED", "interactive"),
}


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def hms(t):
    h, m, s = (int(x) for x in t.split(":"))
    return h + m / 60 + s / 3600


def jl(path):
    return [json.loads(l) for l in open(path) if l.strip()] if os.path.exists(path) else []


def median(v):
    v = sorted(v)
    n = len(v)
    return None if not n else (v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2)


def best(evals):
    return min(evals, key=lambda e: e["val_loss"]) if evals else None


def curve(evals):
    return [(e["val_loss"], e["train_loss"]) for e in evals]


def profile(run_dir):
    p = {"dir": run_dir, "hash": {f: md5(os.path.join(run_dir, f)) for f in RUN_FILES if os.path.exists(os.path.join(run_dir, f))}}
    p["cfg"] = json.load(open(os.path.join(run_dir, "config.json"))) if "config.json" in p["hash"] else {}
    lines = open(os.path.join(run_dir, "metrics.jsonl")).read().splitlines() if "metrics.jsonl" in p["hash"] else []
    p["lines"] = lines
    ml = [json.loads(l) for l in lines if l.strip()]
    blocks, cur = [], None
    for raw, m in zip([l for l in lines if l.strip()], ml):
        if m["type"] == "start":
            cur = {"start": m, "evals": [], "done": None, "lines": []}
            blocks.append(cur)
        if cur is not None:
            cur["lines"].append(raw)
            if m["type"] == "eval":
                cur["evals"].append(m)
            elif m["type"] == "done":
                cur["done"] = m
    p["blocks"] = blocks
    p["all_evals"] = [e for m in ml for e in ([m] if m["type"] == "eval" else [])]
    p["step_rows"] = None
    p["step_median"] = None
    st = os.path.join(run_dir, "step_times.csv")
    if os.path.exists(st):
        rows = list(csv.reader(open(st)))[1:]
        p["step_rows"] = len(rows)
        warm = cfg_warm(p["cfg"])
        p["step_median"] = median([float(x[1]) for x in rows[warm:]] or [float(x[1]) for x in rows])
    return p


def cfg_warm(cfg):  # same rule as train_ddp.py: timing_warmup or max(20, start_iter + 10)
    return cfg["timing_warmup"] if cfg.get("timing_warmup") is not None else max(20, (cfg.get("start_iter") or 0) + 10)


def out_segments(txt):
    segs, cur = [], None
    for line in txt.splitlines():
        if not line.startswith('{"type"'):
            continue
        try:
            m = json.loads(line)
        except ValueError:
            continue
        if m["type"] == "start":
            cur = {"run_id": m["run_id"], "evals": [], "done": None}
            segs.append(cur)
        elif cur is not None and m["type"] == "eval":
            cur["evals"].append(m)
        elif cur is not None and m["type"] == "done":
            cur["done"] = m
    return segs


def footer_elapsed(txt, jid):
    m = re.search(rf"^{jid}\s+(\d+:\d+:\d+)\s+(\d+)", txt, re.M)
    return (m.group(1), m.group(2)) if m else None


def gpu_stats(path):
    rows = list(csv.reader(open(path)))[1:]
    util = [int(r[2].replace("%", "").strip()) for r in rows]
    busy = [u for u in util if u > BUSY_PCT]
    n_gpu = len({r[1].strip() for r in rows})
    return {"samples": len(util), "gpus": n_gpu, "first": rows[0][0].strip(), "last": rows[-1][0].strip(), "seconds": len(util) / max(n_gpu, 1),
            "mean_all": sum(util) / len(util), "mean_busy": sum(busy) / len(busy) if busy else None, "busy_frac": len(busy) / len(util)}


def f3(x):
    return "–" if x is None else f"{x:.3f}"


def label(e):
    return "DDP" if e["mode"] == "ddp" else f"PowerSGD r{e['rank']}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    raw, out = os.path.abspath(a.raw), os.path.abspath(a.out)
    org = os.path.join(out, "organized")
    if os.path.exists(org):
        shutil.rmtree(org)  # only ever the generated output dir, never --raw

    # ---- discover snapshots, oldest -> newest ----------------------------------------------------------------
    copies = sorted((d for d in os.listdir(raw) if os.path.exists(os.path.join(raw, d, "runs.jsonl"))), key=lambda d: os.stat(os.path.join(raw, d, "runs.jsonl")).st_mtime)
    logdirs = sorted((d for d in os.listdir(raw) if d.startswith("logs") and os.path.isdir(os.path.join(raw, d))),
                     key=lambda d: max([os.stat(f).st_mtime for f in glob.glob(os.path.join(raw, d, "*"))] or [0]))
    rank_of = {c: i for i, c in enumerate(copies)}
    records = defaultdict(dict)  # run_id -> {json: record}
    variants = defaultdict(list)
    for c in copies:
        for r in jl(os.path.join(raw, c, "runs.jsonl")):
            records[r["run_id"]][json.dumps(r, sort_keys=True)] = r
        for d in sorted(glob.glob(os.path.join(raw, c, "runs", "*"))):
            variants[os.path.basename(d)].append({"copy": c, "p": profile(d)})

    # logs: latest snapshot of each file name wins; older differing snapshots are listed as superseded
    outs, gpus, superseded_logs = {}, {}, []
    for d in logdirs:  # oldest -> newest, so later overwrite earlier
        for path in sorted(glob.glob(os.path.join(raw, d, "*.out")) + glob.glob(os.path.join(raw, d, "gpu_*.csv"))):
            name = os.path.basename(path)
            prev = (outs if name.endswith(".out") else gpus).get(name)
            if prev and prev["hash"] != md5(path):
                superseded_logs.append((name, prev["dir"], os.path.getsize(prev["path"]), d, os.path.getsize(path)))
            ent = {"path": path, "dir": d, "hash": md5(path)}
            if name.endswith(".out"):
                txt = open(path, errors="replace").read()
                jid = re.search(r"_(\d+)\.out$", name).group(1)
                ent.update(job=jid, segs=out_segments(txt), footer=footer_elapsed(txt, jid))
                outs[name] = ent
            else:
                m = re.match(r"gpu_(\d+)_(.+)\.csv", name)
                ent.update(job=m.group(1), node=m.group(2), **gpu_stats(path))
                gpus[name] = ent
    for p in glob.glob(os.path.join(raw, "gpu_*.csv")):  # stray top-level copies: only noted if they add nothing
        name = os.path.basename(p)
        if name in gpus and gpus[name]["hash"] == md5(p):
            gpus[name].setdefault("dups", []).append("top level")

    # ---- per-run analysis ------------------------------------------------------------------------------------
    entries, flags, superseded = [], [], []
    for rid in sorted(variants):
        vs = variants[rid]
        fin = vs[-1]
        p = fin["p"]
        recs = list(records.get(rid, {}).values())
        for v in vs[:-1]:
            if v["p"]["hash"] != p["hash"]:
                old_lines, new_lines = v["p"]["lines"], p["lines"]
                rel = "metrics.jsonl is a line-prefix of the final one (file grew)" if new_lines[:len(old_lines)] == old_lines else "metrics.jsonl is NOT a prefix of the final one"
                chg = [f for f in RUN_FILES if v["p"]["hash"].get(f) != p["hash"].get(f)]
                superseded.append((rid, v["copy"], fin["copy"], ", ".join(chg), rel))
                if "NOT a prefix" in rel:
                    flags.append(f"`{rid}`: snapshot in `{v['copy']}` is not a prefix of the final `{fin['copy']}` metrics.jsonl")
        cfg = p["cfg"]
        blocks = p["blocks"]
        planned = (blocks[0]["start"].get("total_steps") if blocks else None) or cfg.get("max_steps")
        base_notes = []
        if cfg.get("mode") == "ddp":
            base_notes.append("config.json `rank` is a CLI default for ddp, not a PowerSGD rank")

        def mk(job, rec, block_idx, notes, validity, evals, start, done, step_src, copy_files, split):
            return {"id": rid, "job": job, "rec": rec, "p": p, "mode": cfg.get("mode"), "rank": cfg.get("rank"), "seed": cfg.get("seed"),
                    "world": start.get("world_size"), "gb": start.get("global_batch"), "lr_eff": start.get("lr_eff"), "device": start.get("device"),
                    "planned": planned, "evals": evals, "done": done, "validity": validity, "notes": base_notes + notes, "copy": fin["copy"],
                    "step_src": step_src, "files": copy_files, "split": split, "block": block_idx}

        if len(blocks) > 1:  # two or more runs wrote this folder -> attribute each block by exact match
            attr = []
            for i, b in enumerate(blocks, 1):
                ev, jobs = [], set()
                fv = b["evals"][-1]["val_loss"] if b["evals"] else None
                for r in recs:
                    if fv is not None and abs(fv - r["final_metric"]["val_loss"]) < 1e-9:
                        ev.append(f"final val {fv:.4f} equals the runs.jsonl record of job {r['slurm_job_id']}")
                        jobs.add(r["slurm_job_id"])
                for o in outs.values():
                    for sg in o["segs"]:
                        if sg["run_id"] == rid and b["evals"] and curve(sg["evals"]) == curve(b["evals"]):
                            ev.append(f"all {len(b['evals'])} (val_loss, train_loss) pairs identical to the eval lines in {os.path.basename(o['path'])}")
                            jobs.add(o["job"])
                attr.append({"i": i, "b": b, "ev": ev, "jobs": jobs})
            ok = all(len(x["jobs"]) == 1 for x in attr) and len({next(iter(x["jobs"])) for x in attr}) == len(attr)
            if ok:
                for x in attr:
                    b, job = x["b"], next(iter(x["jobs"]))
                    rec = next((r for r in recs if r["slurm_job_id"] == job), None)
                    # step_times.csv: any snapshot whose median step time equals this block's `done` median (exact to the CSV's 1e-6 rounding)
                    src = None
                    for v in vs:
                        pm = v["p"]["step_median"]
                        if pm is not None and b["done"] and abs(pm - b["done"]["median_step_time_s"]) < 2e-6 and v["p"]["step_rows"] == planned:
                            src = v
                    notes = [f"split from a shared metrics.jsonl (block {x['i']} of {len(attr)}); config.json is identical for both runs",
                             "attributed to job " + job + ": " + "; ".join(x["ev"])]
                    notes.append(f"step_times.csv taken from snapshot `{src['copy']}` (its median step time equals this block's `done` line)" if src else "step_times.csv NOT recoverable (the other job overwrote it)")
                    if rec and b["done"] and abs(rec["median_step_time_s"] - b["done"]["median_step_time_s"]) > 1e-9:
                        notes.append("runs.jsonl median step time differs from the metrics `done` line")
                    entries.append(mk(job, rec, x["i"], notes, "valid (split)", b["evals"], b["start"], b["done"], src,
                                      {"metrics": b["lines"], "step_times": os.path.join(src["p"]["dir"], "step_times.csv") if src else None}, True))
            else:
                notes = []
                for x in attr:
                    notes.append(f"block {x['i']}: " + ("; ".join(x["ev"]) if x["ev"] else "no match in any runs.jsonl record or .out log") + (" -> UNATTRIBUTED" if len(x["jobs"]) != 1 else ""))
                entries.append(mk(None, None, None, notes + ["attribution not settled; nothing split"], "suspect collision", p["all_evals"], blocks[0]["start"], None, None,
                                  {"metrics": p["lines"], "step_times": None}, False))
            continue

        b = blocks[0] if blocks else None
        start = b["start"] if b else {}
        evals = b["evals"] if b else []
        notes, validity = [], "valid"
        steps_ev = [e["step"] for e in evals]
        expected = (planned // cfg["eval_interval"]) if planned and cfg.get("eval_interval") else None
        job = recs[0]["slurm_job_id"] if len(recs) == 1 else None
        if len(recs) > 1:
            notes.append(f"{len(recs)} runs.jsonl records for this run_id but one metrics block: not attributable")
        if len(steps_ev) != len(set(steps_ev)) or (planned and (p["step_rows"] or 0) > planned) or len(recs) > 1:
            validity = "suspect collision"
        elif not (b and b["done"]) or not recs or p["step_rows"] is None or (expected and len(evals) < expected):
            validity = "partial"
            if not recs:
                notes.append("no runs.jsonl record")
                for o in outs.values():  # attribute a record-less run to a job via its .out lines
                    for sg in o["segs"]:
                        if sg["run_id"] == rid and evals and curve(sg["evals"])[:len(evals)] == curve(evals):
                            job = o["job"]
                            notes.append(f"job {job} attributed from {os.path.basename(o['path'])} (first {len(evals)} eval pairs identical)")
        elif rid.startswith("smoke_") or (planned and planned <= 50):
            validity = "smoke test"
        rec = recs[0] if len(recs) == 1 else None
        if rec and evals and abs(evals[-1]["val_loss"] - rec["final_metric"]["val_loss"]) > 1e-6:
            notes.append("final val in metrics.jsonl differs from runs.jsonl")
        if rec and p["step_rows"] is not None and p["step_rows"] != rec["steps"]:
            notes.append("step_times rows differ from runs.jsonl steps")
        st = STATED_RUNS.get(rid)
        if st and rec:
            got = (rec["slurm_job_id"], rec["nodes"], rec["world_size"], cfg.get("seed"), start.get("global_batch"), planned)
            if got != st:
                flags.append(f"`{rid}`: stated (job,nodes,ws,seed,global_batch,steps) {st}, files {got}")
                notes.append("DISAGREES with stated provenance: see list at end")
        if st and not rec:
            notes.append(f"stated job {st[0]}; " + ("confirmed by a .out log" if job == st[0] else "not confirmed by any file"))
        entries.append(mk(job, rec, None, notes, validity, evals, start, b["done"] if b else None, fin,
                          {"metrics": p["lines"], "step_times": os.path.join(p["dir"], "step_times.csv") if p["step_rows"] is not None else None}, False))

    for e in entries:  # stated provenance for the 2-node runs (two jobs share these names)
        if e["split"] and e["rec"]:
            got = (e["rec"]["slurm_job_id"], e["rec"]["nodes"], e["world"], e["seed"], e["gb"], e["planned"])
            if got[0] not in TWO_NODE_JOBS or got[1:] != (2, 8, 1, 512, 2000):
                flags.append(f"`{e['id']}` (job {got[0]}): stated 2-node (nodes 2, world 8, seed 1, global batch 512, steps 2000), files {got[1:]}")

    # ---- sessions and placement ------------------------------------------------------------------------------
    jobs = defaultdict(lambda: {"runs": [], "nodelist": None, "qos": None, "csv": [], "out": None})
    for e in entries:
        if e["job"]:
            j = jobs[e["job"]]
            j["runs"].append(e["id"])
            if e["rec"]:
                j["nodelist"], j["qos"], j["nodes"] = e["rec"]["slurm_nodelist"], e["rec"]["slurm_qos"], e["rec"]["nodes"]
    for g in gpus.values():
        jobs[g["job"]]["csv"].append(g)
    for o in outs.values():
        jobs[o["job"]]["out"] = o
    for jid in STATED_JOBS:
        jobs[jid]

    def sdir(jid):
        j = jobs[jid]
        n = j.get("nodes") or STATED_JOBS.get(jid, (None,))[0] or (len(j["csv"]) or "?")
        kind = "batch" if j["out"] or (j["qos"] and "interactive" not in j["qos"]) else "interactive"
        return f"job{jid}_{n}node_{kind}"

    os.makedirs(org)
    for e in entries:
        d = os.path.join(org, "_suspect_collision" if e["validity"] == "suspect collision" else (sdir(e["job"]) if e["job"] else "_unattributed"), e["id"])
        os.makedirs(d)
        open(os.path.join(d, "metrics.jsonl"), "w").write("\n".join(e["files"]["metrics"]) + "\n")
        shutil.copy2(os.path.join(e["p"]["dir"], "config.json"), d)
        if e["files"]["step_times"]:
            shutil.copy2(e["files"]["step_times"], d)
        json.dump({"run_id": e["id"], "job": e["job"], "validity": e["validity"], "final_snapshot": e["copy"], "split_from_shared_file": e["split"],
                   "step_times_from": e["step_src"]["copy"] if e["step_src"] else None, "runs_jsonl_record": e["rec"], "notes": e["notes"],
                   "raw_file_md5": e["p"]["hash"]}, open(os.path.join(d, "_provenance.json"), "w"), indent=1)
        e["placed"] = os.path.relpath(d, out)
    for g in gpus.values():
        d = os.path.join(org, sdir(g["job"]))
        os.makedirs(d, exist_ok=True)
        shutil.copy2(g["path"], d)
    for o in outs.values():
        d = os.path.join(org, sdir(o["job"]))
        os.makedirs(d, exist_ok=True)
        shutil.copy2(o["path"], d)

    # ---- derived numbers ---------------------------------------------------------------------------------------
    for e in entries:
        e["best"] = best(e["evals"]) if e["validity"] != "suspect collision" else None
        e["final"] = e["evals"][-1] if e["evals"] and e["validity"] not in ("suspect collision", "partial") else None
        e["steps"] = e["rec"]["steps"] if e["rec"] else None
        e["clean"] = e["rec"] is not None and e["validity"] != "suspect collision"
    order = {"valid": 0, "valid (split)": 1, "smoke test": 2, "partial": 3, "suspect collision": 4}
    ents = sorted(entries, key=lambda e: (order[e["validity"]], e["job"] or "", e["id"]))

    L = ["# Results manifest", "",
         f"_Generated by `tools/organize_results.py` from `{os.path.basename(raw)}/` (snapshots oldest to newest: {', '.join(copies)}; logs: {', '.join(logdirs)}). "
         "Computed from the copied files, not from run names. Raw files are untouched; `results/organized/` holds copies._", "",
         "## Sessions (jobs)", "",
         "| session folder | job | kind | nodes | state / elapsed | GPU CSV | .out log | runs |", "|---|---|---|---|---|---|---|---|"]
    hours = []
    for jid in sorted(jobs):
        j = jobs[jid]
        sn, sel, sstate, skind = STATED_JOBS.get(jid, (None, None, None, None))
        fe = j["out"]["footer"] if j["out"] and j["out"]["footer"] else None
        elapsed = sel or (fe[0] if fe else None)
        nn = sn or (int(fe[1]) if fe else j.get("nodes"))
        csvs = ", ".join(f"`{g['node']}`" for g in j["csv"]) or "none"
        nodes_txt = j["nodelist"] or (", ".join(g["node"] for g in j["csv"]) or "?")
        el_txt = f"{sstate or '?'} / {elapsed or '?'}" + (f" (PROVISIONAL: {PROVISIONAL[jid]})" if jid in PROVISIONAL else "")
        L.append(f"| `{sdir(jid)}` | {jid} | {skind or '?'} | {nn} ({nodes_txt}) | {el_txt} | {csvs} | {'`' + os.path.basename(j['out']['path']) + '`' if j['out'] else '–'} | {', '.join(f'`{x}`' for x in j['runs']) or '–'} |")
        if elapsed and nn:
            hours.append((jid, nn, elapsed, hms(elapsed) * nn, hms(elapsed) * nn * GPUS_PER_NODE, bool(sel)))
    L += ["", "Job states and elapsed times are as stated by Aaron from `sacct` (the repo has no sacct output for interactive jobs). "
          "Job 60622124 hit its time limit (TIMEOUT).", "",
          "## Runs", "",
          "Best val loss and its step, and final val loss, are computed from `metrics.jsonl`. Median step time, comm bytes and GPU-hours come from that job's `runs.jsonl` record. "
          "`validity`: valid; valid (split) = separated from a shared file by exact match; smoke test; partial; suspect collision.", "",
          "| run_id | job | nodes | world | seed | global batch | steps | final val | best val (step) | median step | comm/step | GPU | GPU-h | validity |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for e in ents:
        r = e["rec"]
        c = e["clean"]
        best_s = f"{e['best']['val_loss']:.3f} ({e['best']['step']})" if e["best"] else "–"
        ms = f"{r['median_step_time_s'] * 1000:.1f} ms" if c else "–"
        mb = f"{r['comm_bytes_per_step'] / 1e6:.3f} MB" if c else "–"
        gh = f"{r['gpu_hours']:.3f}" if c else "–"
        L.append(f"| `{e['id']}` | {e['job'] or '–'} | {r['nodes'] if r else '–'} | {e['world'] or '–'} | {e['seed'] if e['seed'] is not None else '–'} | {e['gb'] or '–'} | "
                 f"{e['steps'] if e['steps'] else '–'} ({e['planned']}) | {f3(e['final']['val_loss']) if e['final'] else '–'} | {best_s} | {ms} | {mb} | "
                 f"{(e['device'] or '–').replace('NVIDIA ', '')} | {gh} | **{e['validity']}** |")
    L += ["", "### Notes per run", ""]
    for e in ents:
        if e["notes"]:
            L.append(f"- `{e['id']}` (job {e['job'] or 'unknown'}) [{e['validity']}], in `{e['placed']}`:")
            L += [f"  - {n}" for n in e["notes"]]

    splits = [e for e in ents if e["split"]]
    if splits or any(e["validity"] == "suspect collision" for e in ents):
        L += ["", "## Doubled files: evidence for each attribution", "",
              "Runs `gpt_ddp_2node`, `gpt_r2_2node`, `gpt_r4_2node` were written by both the interactive job 60627319 and the sbatch job 60623329, so each `metrics.jsonl` holds two runs. "
              "Each block (one per `start` line) is attributed only when it matches by exact equality: (a) its eval (val_loss, train_loss) pairs equal the eval lines printed in a batch job's `.out`; "
              "(b) its final val loss equals the final val loss in a `runs.jsonl` record. `step_times.csv` is not interleaved, so it is taken from whichever snapshot's median step time equals the block's `done` line.", "",
              "| run_id | block | attributed to | evidence | step_times.csv |", "|---|---|---|---|---|"]
        for e in ents:
            if e["split"]:
                ev = next(n for n in e["notes"] if n.startswith("attributed to")).split(": ", 1)[1]
                L.append(f"| `{e['id']}` | {e['block']} | job {e['job']} | {ev} | {'snapshot `' + e['step_src']['copy'] + '`' if e['step_src'] else 'not recoverable'} |")
            elif e["validity"] == "suspect collision" and e["id"] in ("gpt_ddp_2node", "gpt_r2_2node", "gpt_r4_2node"):
                L.append(f"| `{e['id']}` | – | UNATTRIBUTED | {'; '.join(e['notes'])} | – |")
        L.append("")
        pairs = defaultdict(dict)
        for e in splits:
            pairs[e["id"]][e["job"]] = e
        L += ["Same config, seed 1, different jobs: how close two attributed runs land.", "",
              "| run | job 60627319 final / best val | job 60623329 final / best val | final val gap | median step 60627319 | median step 60623329 |", "|---|---|---|---|---|---|"]
        for rid, d in sorted(pairs.items()):
            x, y = d.get("60627319"), d.get("60623329")
            if x and y:
                L.append(f"| `{rid}` | {x['final']['val_loss']:.4f} / {x['best']['val_loss']:.4f} ({x['best']['step']}) | {y['final']['val_loss']:.4f} / {y['best']['val_loss']:.4f} ({y['best']['step']}) | "
                         f"{abs(x['final']['val_loss'] - y['final']['val_loss']):.4f} ({100 * abs(x['final']['val_loss'] - y['final']['val_loss']) / x['final']['val_loss']:.2f}%) | "
                         f"{x['rec']['median_step_time_s'] * 1000:.2f} ms | {y['rec']['median_step_time_s'] * 1000:.2f} ms |")

    # ---- long runs ---------------------------------------------------------------------------------------------
    one = [e for e in ents if e["validity"] == "valid" and e["rec"] and e["rec"]["nodes"] == 1 and e["mode"] and e["best"]]
    ddp1 = [e for e in one if e["mode"] == "ddp" and e["planned"] >= 1000]
    long_runs = [e for e in one if e["planned"] and e["planned"] > 2000]
    if long_runs and ddp1:
        tgt = min(e["best"]["val_loss"] for e in ddp1)
        L += ["", "## Long runs: does PowerSGD reach DDP's best loss?", "",
              "1-node (A100-80GB), seed 1, global batch 256. DDP's best val loss over all 1-node DDP runs of at least 1000 steps: "
              + ", ".join(f"`{e['id']}` {e['best']['val_loss']:.4f} (step {e['best']['step']})" for e in sorted(ddp1, key=lambda e: e["id"])) + f". Target = lowest, **{tgt:.4f}**.", "",
              "| run | steps | best val (step) | wall at best | first eval ≤ target | final val | final − best | late overfitting? |", "|---|---|---|---|---|---|---|---|"]
        for e in sorted(ddp1 + long_runs, key=lambda e: (e["planned"], e["id"])):
            b = e["best"]
            hit = next((x for x in e["evals"] if x["val_loss"] <= tgt), None)
            last = e["evals"][-1]
            after = [x["val_loss"] for x in e["evals"] if x["step"] >= b["step"]]
            tr_best = next(x["train_loss"] for x in e["evals"] if x["step"] == b["step"])
            rising = len(after) > 1 and all(y > x for x, y in zip(after, after[1:])) and last["val_loss"] - b["val_loss"] > 0.01 and last["train_loss"] < tr_best
            L.append(f"| `{e['id']}` ({label(e)}) | {e['planned']} | {b['val_loss']:.4f} ({b['step']}) | {b['elapsed_s']:.0f} s | "
                     f"{('step ' + str(hit['step']) + f' ({hit['elapsed_s']:.0f} s)') if hit else 'not reached'} | {last['val_loss']:.4f} | {last['val_loss'] - b['val_loss']:+.4f} | "
                     f"{'yes: val rises after step ' + str(b['step']) + ' while train loss keeps falling' if rising else ('no (best is the last eval)' if b['step'] == last['step'] else 'val moves up and down after the best')} |")
        L += ["", "Per-eval val loss for the long runs:", ""]
        for e in long_runs:
            L.append(f"- `{e['id']}`: " + ", ".join(f"{x['step']}: {x['val_loss']:.3f}" for x in e["evals"]) + " (train loss at last eval " + f"{e['evals'][-1]['train_loss']:.3f})")
        L += ["", "Caveats: evals are every 500 steps; wall time includes evaluation. The cosine LR schedule is a function of the planned step count (`total_steps`), so a 6000-step run is a different "
              "trajectory from a 2000-step one, not a longer version of it. There is no DDP run with a 6000-step schedule, so the table does not show whether DDP would also improve with that schedule. "
              "One seed each; two identical DDP runs differ in best val loss by about 0.001 (see the repeat-run table)."]

    # ---- 2-node ------------------------------------------------------------------------------------------------
    two = [e for e in ents if e["world"] == 8 and e["rec"] and e["validity"].startswith("valid")]
    if two:
        L += ["", "## 2-node runs: step time by job", "",
              "| config | job 60627319 (interactive) | job 60623329 (sbatch) | difference | hardware 60627319 / 60623329 |", "|---|---|---|---|---|"]
        by = defaultdict(dict)
        for e in two:
            by[(e["mode"], e["rank"] if e["mode"] == "powersgd" else None)][e["job"]] = e
        for k, d in sorted(by.items(), key=lambda kv: (kv[0][0] != "ddp", kv[0][1] or 0)):
            x, y = d.get("60627319"), d.get("60623329")
            if x and y:
                a1, a2 = x["rec"]["median_step_time_s"] * 1000, y["rec"]["median_step_time_s"] * 1000
                L.append(f"| {label(x)} | {a1:.2f} ms | {a2:.2f} ms | {a2 - a1:+.2f} ms ({100 * (a2 / a1 - 1):+.2f}%) | {x['device'].replace('NVIDIA ', '')} on {x['rec']['slurm_nodelist']} / {y['device'].replace('NVIDIA ', '')} on {y['rec']['slurm_nodelist']} |")
        L += ["", "Within each job (same hardware), PowerSGD vs DDP median step time:", "", "| job | " + " | ".join(["DDP", "r2 vs DDP", "r4 vs DDP"]) + " |", "|---|---|---|---|"]
        for jid in TWO_NODE_JOBS:
            dd = by.get(("ddp", None), {}).get(jid)
            r2 = by.get(("powersgd", 2), {}).get(jid)
            r4 = by.get(("powersgd", 4), {}).get(jid)
            if dd and r2 and r4:
                m = dd["rec"]["median_step_time_s"]
                L.append(f"| {jid} | {m * 1000:.2f} ms | {r2['rec']['median_step_time_s'] * 1000:.2f} ms ({100 * (r2['rec']['median_step_time_s'] / m - 1):+.1f}%) | {r4['rec']['median_step_time_s'] * 1000:.2f} ms ({100 * (r4['rec']['median_step_time_s'] / m - 1):+.1f}%) |")
        L += ["", "Global batch 512 (world size 8) vs 256 on 1 node, same `lr_eff`: losses are not comparable between 1-node and 2-node runs. "
              "Step times across 1-node (A100-SXM4-80GB) and 2-node (A100-SXM4-40GB) are also confounded by the GPU variant."]

    # ---- GPU utilization ---------------------------------------------------------------------------------------
    L += ["", "## GPU utilization", "",
          f"Active = a 1-second GPU sample with utilization **above {BUSY_PCT}%**. `mean (all)` averages every sample in the file (idle time included); `mean (active)` only the active samples. "
          "Times are the node's local (Pacific) clock. The CSV runs from monitor start to `finish`, so batch CSVs include startup and the gaps between runs.", "",
          "| csv | job | node | GPUs | span | seconds | mean (all) | mean (active) | active share | completeness |", "|---|---|---|---|---|---|---|---|---|---|"]
    for g in sorted(gpus.values(), key=lambda g: (g["job"], g["node"])):
        j = jobs[g["job"]]
        comp = "complete batch job" if j["out"] and STATED_JOBS.get(g["job"], (0, 0, "",))[2] == "COMPLETED" else \
            "interactive allocation: end of file not verified, do not quote as final" if not j["out"] else "batch job, state unknown"
        mb = f"{g['mean_busy']:.1f}%" if g["mean_busy"] is not None else "–"
        L.append(f"| `{os.path.basename(g['path'])}`{' (identical top-level copy too)' if g.get('dups') else ''} | {g['job']} | {g['node']} | {g['gpus']} | {g['first']} to {g['last']} | {g['seconds']:.0f} | "
                 f"{g['mean_all']:.1f}% | {mb} | {100 * g['busy_frac']:.0f}% | {comp} |")
    for jid in jobs:
        if jobs[jid]["runs"] and not jobs[jid]["csv"] and jid in STATED_JOBS:
            L.append(f"\nNo GPU CSV for job {jid} ({', '.join(sorted(set(jobs[jid]['runs'])))}); its monitor failed (stated).")

    # ---- hours -------------------------------------------------------------------------------------------------
    L += ["", "## Hours", "", "From elapsed times as stated by Aaron x nodes x 4 GPUs. Job 60623330's 0:19:58 is provisional (`squeue` at COMPLETING; the in-job sacct footer said 0:19:54).", "",
          "| job | nodes | elapsed | node-hours | GPU-hours |", "|---|---|---|---|---|"]
    for jid, nn, el, nh, gh, st in hours:
        L.append(f"| {jid} | {nn} | {el}{' (provisional)' if jid in PROVISIONAL else ''} | {nh:.3f} | {gh:.3f} |")
    tnh, tgh = sum(h[3] for h in hours), sum(h[4] for h in hours)
    run_gh = sum(r["gpu_hours"] for r in {json.dumps(x, sort_keys=True): x for v in records.values() for x in v.values()}.values())
    L += [f"| **total** | | | **{tnh:.3f}** | **{tgh:.3f}** |", "",
          f"Sum of per-run `gpu_hours` in `runs.jsonl` (training wall-clock x GPUs, all records): {run_gh:.3f}. It is lower than the job totals because it excludes allocation startup, idle time and container/launch overhead.",
          "**Charged hours may differ** from elapsed x nodes (NERSC applies QOS charge factors and its own rounding); confirm in Iris or with `sacct` / `tools/update_budget.py`."]

    # ---- repeats and lr ----------------------------------------------------------------------------------------
    grp = defaultdict(list)
    for e in ents:
        if e["validity"] == "valid" and e["rec"]:
            grp[(e["mode"], e["rank"] if e["mode"] == "powersgd" else None, e["world"], e["planned"], e["gb"], e["seed"])].append(e)
    L += ["", "## Repeat runs of the same configuration (different jobs)", "", "| config | run | job | final val | best val (step) | median step |", "|---|---|---|---|---|---|"]
    for k, v in sorted(((k, v) for k, v in grp.items() if len(v) > 1), key=lambda kv: str(kv[0])):
        for e in v:
            L.append(f"| {k[0]}{' r' + str(k[1]) if k[1] else ''}, world {k[2]}, {k[3]} steps, batch {k[4]}, seed {k[5]} | `{e['id']}` | {e['job']} | {e['final']['val_loss']:.4f} | {e['best']['val_loss']:.4f} ({e['best']['step']}) | {e['rec']['median_step_time_s'] * 1000:.2f} ms |")
    lrs = defaultdict(set)
    for e in ents:
        if e["lr_eff"] is not None and e["validity"] != "suspect collision":
            lrs[e["world"]].add(e["lr_eff"])
    L += ["", "## Effective learning rate by world size (from `start` lines)", ""] + [f"- world size {w}: lr_eff {sorted(v)}" for w, v in sorted(lrs.items())]

    # ---- superseded snapshots, ambiguities ---------------------------------------------------------------------
    L += ["", "## Superseded snapshots (older copies that differ from the final one)", ""]
    if superseded or superseded_logs:
        L += ["| run | older copy | final copy | files that differ | relation |", "|---|---|---|---|---|"] + [f"| `{a}` | {b} | {c} | {d} | {e} |" for a, b, c, d, e in superseded]
        for n, d0, s0, d1, s1 in superseded_logs:
            L.append(f"| log `{n}` | {d0} ({s0} bytes) | {d1} ({s1} bytes) | content | newer file is larger (job was still running at the older copy) |")
    else:
        L.append("None.")
    L += ["", "## Provenance check against what Aaron stated", ""] + ([f"- {f}" for f in flags] or ["- No run disagrees with the stated (job, nodes, world size, seed, global batch, steps)."])
    L += ["", "## Not run (no files exist)", "", "- ResNet-18/CIFAR-10 with `train_ddp.py`, FP8 (A100 has no native FP8), tensor parallelism, LSTM in `train_ddp.py`.", ""]
    open(os.path.join(out, "MANIFEST.md"), "w").write("\n".join(L))
    print(f"wrote MANIFEST.md; {len(entries)} run entries, {len(gpus)} GPU CSVs, {len(jobs)} jobs")
    for e in ents:
        print(f"  {e['validity']:18s} {e['id']:26s} job={e['job']}  -> {e['placed']}")


if __name__ == "__main__":
    main()
