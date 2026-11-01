# Risks (keep updated)

| # | Risk | Likelihood | Mitigation | Status |
|---|---|---|---|---|
| 1 | Single-node runs show no PowerSGD speedup (NVLink makes all-reduce cheap) | High | Run 2-node (8 GPU) comparison; report the 1-node null result honestly; profile compute vs all-reduce | Not yet measured |
| 2 | Node-hour budget burns fast; queue waits eat the weekend | Medium | Smoke-test interactively first; batch several configs per job; ask before any job > 2 node-hours; `tools/update_budget.py` after every job | Allocation size still unknown |
| 3 | FP8 is emulated on A100 (no FP8 hardware): tests numerics, not speed or wire bytes | Certain | State it on every slide/table row (`precision_note` in runs.jsonl) | Hook written, tested on CPU only |
| 4 | TP interacts non-trivially with PowerSGD's per-layer low-rank structure | High | Design note only (Tier 4); cut implementation | Not started |
| 5 | Small models (ResNet-18) show no benefit at all; NanoGPT is the scale test | Medium | Run NanoGPT Shakespeare-char; if budget allows a larger config | Not started |
| 6 | Deadline Sun 11/01: unfinished work gets cut | High | Tiered plan; anything unrun is labeled "planned" | Tier 1 code written, nothing run on GPUs |
| 7 | Code is untested on Perlmutter (only CPU/gloo smoke tests so far); e.g. bf16 hook needs CUDA+NCCL, PowerSGD QR in fp16/bf16 may fail | Medium | `scripts/interactive_smoke.sh` before any batch job | Open |
| 8 | `comm_bytes_per_step` counts logical `dist.all_reduce` payload per rank, not wire bytes; baseline goes through the Python allreduce hook (slightly slower than DDP's C++ path) | Certain | Disclosed in `runs.jsonl` and RESULTS.md; same hook path for all modes keeps comparison fair | Disclosed |
