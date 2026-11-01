"""One-JSON-line-per-run logger -> results/runs.jsonl (see brief section 6)."""
import fcntl
import json
import os
import socket
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS_JSONL = Path(os.environ.get("RUNS_JSONL", ROOT / "results" / "runs.jsonl"))  # override for smoke tests


def git_commit() -> str:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = subprocess.call(["git", "diff", "--quiet"], cwd=ROOT) != 0
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def slurm_info() -> dict:
    e = os.environ
    return {
        "slurm_job_id": e.get("SLURM_JOB_ID"),
        "slurm_account": e.get("SLURM_JOB_ACCOUNT"),
        "slurm_qos": e.get("SLURM_JOB_QOS"),
        "slurm_nodelist": e.get("SLURM_JOB_NODELIST"),
        "host": socket.gethostname(),
    }


def log_run(record: dict, path: Path = RUNS_JSONL) -> None:
    """Append one record. Adds git commit, slurm info, timestamp and GPU-hours."""
    rec = dict(record)
    rec.setdefault("git_commit", git_commit())
    rec.setdefault("timestamp", time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    rec.update({k: v for k, v in slurm_info().items() if k not in rec})
    if "gpu_hours" not in rec and "gpus_used" in rec and "wall_time_s" in rec:
        rec["gpu_hours"] = rec["gpus_used"] * rec["wall_time_s"] / 3600.0
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps(rec, default=str) + "\n")
        fcntl.flock(f, fcntl.LOCK_UN)


def read_runs(path: Path = RUNS_JSONL) -> list:
    if not Path(path).exists():
        return []
    return [json.loads(l) for l in open(path) if l.strip()]
