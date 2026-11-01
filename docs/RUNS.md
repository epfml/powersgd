# What was run, and how (NanoGPT, 1 node, Sat 2026-10-31)

Everything here was run by Aaron on NERSC Perlmutter (account `m4431_g`). Results are in `results/`, summarized in `results/RESULTS.md`.
Anything not listed under "Completed" has **not** produced results.

## Completed

**Environment:** one interactive allocation, Slurm job `60622124`, node `nid200380`, 30 min, 1 node x 4 A100-SXM4-80GB, torch `2.13.0+cu130`,
repo at commit `7b066e3` (`train_ddp.py` as on `master` at the time). Run names, seed and flags below are read from each run's `config.json`.

**How:** commands were typed by hand inside the allocation using the `launch` helper from `scripts/common.sh`
(one `torchrun` per node, 4 processes/node, NCCL, c10d rendezvous), with `start_gpu_monitor` running `nvidia-smi` once a second in the background.
The equivalent of each run is:

```bash
source $SCRATCH/powersgd/scripts/common.sh
start_gpu_monitor
COMMON="--model nanogpt --seed 1 --max-steps 2000 --eval-interval 250 --start-iter 10"
launch $COMMON --mode ddp               --run-name gpt_ddp_r0_1node
launch $COMMON --mode powersgd --rank 1 --run-name gpt_powersgd_r1_1node
launch $COMMON --mode powersgd --rank 2 --run-name gpt_powersgd_r2_1node
launch $COMMON --mode powersgd --rank 4 --run-name gpt_powersgd_r4_1node
stop_gpu_monitor
```

(Reconstructed from `config.json`; the exact shell history was not saved.) Two 50-step smoke tests (`smoke_gpt_ddp`, `smoke_gpt_r2`,
`--max-steps 50 --eval-interval 25`) were run first to check the pipeline.

**Settings shared by all four sweep runs** (from `metrics.jsonl` / `runs.jsonl`): NanoGPT 6 layers / 6 heads / 384 dim / block 256 (10.8M params),
Shakespeare-char, fp32, seed 1, 64 samples per GPU (global batch 256), effective lr 0.001, 2000 steps, `start_powerSGD_iter` 10,
PyTorch built-in `powerSGD_hook` with error feedback and warm start. DDP runs through the same Python all-reduce hook (not the default C++ path)
so the logical communication bytes are measured identically.

**Files** (one directory per run in `results/runs/<run_name>/`):

| file | contents |
|---|---|
| `results/runs.jsonl` | one record per run: config summary, best/final metric, median step time, comm bytes/step, wall time, GPU-hours, Slurm job info |
| `config.json` | CLI args (`lr`/`batch_size` are `null` = trainer default; the values actually used are in the `start` line of `metrics.jsonl` and in `runs.jsonl`) |
| `metrics.jsonl` | a `start` line, one `eval` line per 250 steps (train loss, val loss, elapsed seconds), a `done` line |
| `step_times.csv` | per-step time, data-loading time, comm bytes |
| `results/gpu/gpu_60622124_nid200380.csv` | `nvidia-smi` samples, 4 GPUs at 1 Hz, for the whole allocation |

Regenerate the tables and figures with `python tools/make_results.py` (needs matplotlib).

## How to read the results

- **Do not use final val loss as the headline.** DDP overfits (best 1.48 at step 750, then rises to 2.21 while train loss falls to 0.52), and the PowerSGD
  runs were still improving at step 2000. Use best val loss and its step, val loss at fixed steps, and the curves.
- **Do not claim "PowerSGD beats DDP".** The supported statements: DDP converges fastest per step; higher rank converges faster per step; PowerSGD cuts
  logical communication ~156x / 100x / 59x (ranks 1 / 2 / 4) but costs 4-13% step time on a single NVLink node.
- `comm_bytes_per_step` is the logical payload handed to `dist.all_reduce` per rank, not wire bytes.
- One seed per config.
- GPU utilization: 96.7% while a run was active, 38.7% over the whole 30-minute allocation. The difference is idle time between hand-launched
  commands, which is an argument for `sbatch` over interactive allocations.

## Submitted, pending in the queue (NO results; do not report)

Both were submitted with `sbatch` from Perlmutter and were still PENDING (Priority) at 16:19 on 2026-10-31. The sbatch files were written on
Perlmutter and are added here (re-created from their description: diff against the Perlmutter copies).

| script | job | what | question |
|---|---|---|---|
| `scripts/nanogpt_2node.sbatch` | 60623329 `gpt_2n` | 2 nodes x 4 GPUs, 40 min; DDP, PowerSGD r2, r4, 2000 steps (`gpt_ddp_2node`, `gpt_r2_2node`, `gpt_r4_2node`) | does compression help once the all-reduce crosses nodes? |
| `scripts/nanogpt_1node_long.sbatch` | 60623330 `gpt_1n_long` | 1 node, 40 min; DDP 1000 steps, PowerSGD r2 and r4 6000 steps (`gpt_ddp_1000_1node`, `gpt_r2_6000_1node`, `gpt_r4_6000_1node`) | does PowerSGD catch up to DDP's best loss with more steps? |

These run whatever `train_ddp.py` is on Perlmutter when they start. Keep that file unchanged until they finish (see the trainer note in the PR).

## Not run

- ResNet-18/CIFAR-10 with `train_ddp.py` (`scripts/time_epoch_1node.sh`, tier1 and tier3 sbatch files were not run).
- FP8. A100 has no native FP8, so anything here would be emulated.
- Tensor parallelism: design only.
- Nayan separately ran the authors' `paper-code` (his scripts are not on this branch). His reported numbers (CIFAR-10 smoke tests, a failed LSTM/WikiText-2
  smoke test) are secondhand and unverified here.
