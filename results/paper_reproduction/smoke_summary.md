# PowerSGD paper reproduction: Perlmutter smoke test

Run with `./perlmutter/submit_paper.sh smoke` on 1 node (4x A100, NCCL), 1-2 epochs.
These runs check the pipeline, not final accuracy (the paper trains 300 epochs on 16 GPUs).
MB/epoch scales with the number of steps per epoch, so 4-GPU values are 4x the 16-GPU paper values.

| Task | Algorithm | GPUs | Epochs | Test acc. (%) | Paper (300 ep) | MB/epoch | MB/epoch at 16 GPUs | Paper MB/epoch | ms/batch | vs SGD | Paper ms/batch | reduce ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cifar | SGD | 4 (nccl) | 2/2 | 49.0 | 94.3 | 4335 | 1084 | 1023 | 22.8 | +0% | 312 | 3.4 |
| cifar | Rank 2 | 4 (nccl) | 2/2 | 34.9 | 94.4 | 32 | 8 | 8 | 26.3 | +16% | 239 | 5.9 |
| cifar | Signum | 4 (nccl) | 1/1 | 51.4 | 93.6 | 135 | 34 | 32 | 24.1 | +6% | 301 | 4.7 |
| lm | Rank 4 | 4 (nccl) | 1/1 | 1303.2 (ppl) | 91 (ppl, 90 ep) | 262 | 66 | 64 | 27.8 | - | 134 | 3.2 |

Observations:
- Communication volumes match the paper (Tables 1, 3 and 6).
- On 4 NVLink-connected A100s, gradient exchange is cheap (3-6 ms), so Rank-2 PowerSGD is 16% slower
  per batch than SGD. The paper's speedups came from a 10 Gbit/s network; the 16-GPU and Gloo scaling
  runs test whether PowerSGD helps once communication is a bottleneck.
- The first LSTM run failed: the WikiText-2 host returns 403 to Python's default User-Agent, so the
  dataset download on rank 0 failed and the other ranks timed out. After the fix in
  `paper-code/train.py` the rerun completed. Perplexity after 1 epoch (still in LR warmup) is high, as
  expected; the paper's 91 is after 90 epochs.
