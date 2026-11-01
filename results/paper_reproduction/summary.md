## cifar16

| Task | Algorithm | GPUs | Seeds (done) | Epochs | Test acc. (%) ours | paper | Running-avg model | MB/epoch ours | paper | ms/batch ours | vs SGD | paper | reduce ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cifar | Rank 1 | 16 (nccl) | 3 (3) | 300/300 | 93.3 [93.1, 93.4] | 93.6 | 93.3 [93.1, 93.4] | 4 | 4 | 28.3 | +8% | 229 | 7.7 |
| cifar | Rank 2 | 16 (nccl) | 3 (3) | 300/300 | 94.4 [94.3, 94.4] | 94.4 | 94.4 [94.2, 94.4] | 8 | 8 | 29.6 | +13% | 239 | 9.2 |
| cifar | Rank 4 | 16 (nccl) | 3 (3) | 300/300 | 94.5 [94.3, 94.7] | 94.5 | 94.4 [94.3, 94.7] | 15 | 14 | 33.2 | +27% | 260 | 12.6 |
| cifar | SGD | 16 (nccl) | 3 (3) | 300/300 | 94.1 [94.1, 94.1] | 94.3 | 94.1 [94.1, 94.1] | 1073 | 1023 | 26.2 | +0% | 312 | 6.4 |

## lm16

| Task | Algorithm | GPUs | Seeds (done) | Epochs | Test ppl. ours | paper | Running-avg model | MB/epoch ours | paper | ms/batch ours | vs SGD | paper | reduce ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lm | Rank 1 | 16 (nccl) | 3 (3) | 90/90 | 111.2 [110.5, 112.0] | 102.0 | n/a | 25 | 25 | 26.5 | -8% | 131 | 1.9 |
| lm | Rank 2 | 16 (nccl) | 3 (3) | 90/90 | 101.4 [100.9, 102.1] | 93.0 | n/a | 39 | 38 | 27.1 | -6% | 141 | 2.5 |
| lm | Rank 4 | 16 (nccl) | 3 (3) | 90/90 | 98.9 [98.3, 99.3] | 91.0 | n/a | 65 | 64 | 28.1 | -3% | 134 | 3.5 |
| lm | SGD | 16 (nccl) | 3 (3) | 90/90 | 99.6 [99.4, 99.9] | 91.0 | n/a | 7882 | 7730 | 28.8 | +0% | 300 | 4.4 |

## smoke

| Task | Algorithm | GPUs | Seeds (done) | Epochs | Test acc. (%) ours | paper | Running-avg model | MB/epoch ours | paper | ms/batch ours | vs SGD | paper | reduce ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cifar | Rank 2 | 4 (nccl) | 1 (1) | 2/2 | 34.9 | 94.4 | 29.2 | 32 | 8 | 26.3 | +16% | 239 | 5.9 |
| cifar | SGD | 4 (nccl) | 1 (1) | 2/2 | 49.0 | 94.3 | 28.9 | 4335 | 1023 | 22.8 | +0% | 312 | 3.4 |
| cifar | Signum | 4 (nccl) | 1 (1) | 1/1 | 51.4 | 93.6 | 18.5 | 135 | 32 | 24.1 | +6% | 301 | 4.7 |

| Task | Algorithm | GPUs | Seeds (done) | Epochs | Test ppl. ours | paper | Running-avg model | MB/epoch ours | paper | ms/batch ours | vs SGD | paper | reduce ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lm | Rank 4 | 4 (nccl) | 1 (1) | 1/1 | 1303.2 | 91.0 | n/a | 262 | 64 | 27.8 | - | 134 | 3.2 |

