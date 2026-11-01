# NOTES (running log; newest first)

## 2026-10-31 (Sat)
**First real GPU results (1 node, NanoGPT):** see `results/RESULTS.md`, `docs/RUNS.md`, `figures/`. Job 60622124, 4x A100, DDP vs PowerSGD r1/r2/r4, 2000 steps, seed 1.
Headline: use *best* val loss, not final (DDP overfits after step 750; PowerSGD runs were still improving); PowerSGD costs 4-13% step time on one node and sends 59-156x fewer logical bytes. 96.7% GPU util while active, 38.7% over the allocation.
**Trainer (`train_ddp.py`) logging-only patch:** `best_step` and `lr_eff` in `runs.jsonl`; resolved batch size / lr in `config.json`. No training logic changed. NOT yet tested on a GPU node: do not copy to Perlmutter until the queued jobs (60623329, 60623330) finish and a smoke test passes on a compute node.
**Queued, no results yet:** `scripts/nanogpt_2node.sbatch` (60623329), `scripts/nanogpt_1node_long.sbatch` (60623330).
**Not run:** ResNet-18 with `train_ddp.py`, FP8, TP (design only), LSTM in `train_ddp.py`.

## 2026-10-29 (Thu)
**Done (all local, CPU/gloo only; nothing has run on Perlmutter, no GPU results exist):**
- `train_ddp.py`: torchrun, ResNet-18/CIFAR-10 and NanoGPT/Shakespeare-char, `--mode {ddp,powersgd} --rank R --precision {fp32,fp16,bf16,fp8_e4m3,fp8_e5m2}`, built-in PowerSGD hook (error feedback + warm start ON), per-run `results/runs.jsonl` line, `results/runs/<id>/{config.json,metrics.jsonl,step_times.csv}`, optional `--profile-step`.
- `common/run_logger.py`, `tools/update_budget.py` (sacct -> `results/budget.md`), `tools/prepare_data.py`, `scripts/common.sh`, `scripts/interactive_smoke.sh`, `scripts/tier1_resnet18_1node.sbatch` (`-A m4431_g -C gpu`), `docs/risks.md`.
- CPU smoke tests (2 procs, synthetic data) passed for ddp/powersgd x fp32/fp16/fp8_e4m3/fp8_e5m2 on NanoGPT, and ddp/powersgd fp32 on ResNet-18. bf16 comm hook fails on CPU by design (needs CUDA+NCCL) - unverified on GPU.
- `paper-code/` untouched.

**Run order (agreed):** (1) smoke + epoch timing on 1 node (`scripts/interactive_smoke.sh`, `scripts/time_epoch_1node.sh`); (2) 1-node sweep DDP + R1/R2/R4 ~30 epochs (`scripts/tier1_resnet18_1node.sbatch`); (3) one 2-node run, DDP vs R2 (`scripts/tier3_resnet18_2node.sbatch`). No paper-code launcher (decided against).

**Next (needs you on Perlmutter):**
1. Clone/copy repo to `$SCRATCH/powersgd`; `module avail pytorch`; check torch version has float8 + NCCL; `python tools/prepare_data.py` on a login node.
2. `salloc` 1 node, run `bash scripts/interactive_smoke.sh`; paste output back so I can cost Tier 1.
3. Tell me: allocation remaining on `m4431_g` (Iris), QOS list (`sacctmgr show assoc user=$USER format=account%12,qos%80`), Nayan's account name.
4. I then give a node-hour estimate for the Tier 1 sbatch and wait for your OK.

**Not started:** NanoGPT real runs, 2-node runs, plots/RESULTS.md generator, TP design note, LSTM/WikiText-2.
