| Group | Task | Algorithm | Seed | GPUs | Epochs | Final test metric | MB/epoch | ms/batch | forward | backward | reduce | Run directory |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cifar16 | cifar | Rank 1 | 1 | 16 (nccl) | 300/300 | 93.40 | 4.4 | 28.35 | 4.34 | 9.86 | 7.79 | cifar16/cifar_rank-1_w16_nccl_s1 |
| cifar16 | cifar | Rank 1 | 2 | 16 (nccl) | 300/300 | 93.06 | 4.4 | 28.36 | 4.33 | 9.83 | 7.68 | cifar16/cifar_rank-1_w16_nccl_s2 |
| cifar16 | cifar | Rank 1 | 3 | 16 (nccl) | 300/300 | 93.36 | 4.4 | 28.09 | 4.34 | 9.82 | 7.59 | cifar16/cifar_rank-1_w16_nccl_s3 |
| cifar16 | cifar | Rank 2 | 1 | 16 (nccl) | 300/300 | 94.39 | 7.9 | 29.69 | 4.34 | 9.69 | 9.18 | cifar16/cifar_rank-2_w16_nccl_s1 |
| cifar16 | cifar | Rank 2 | 2 | 16 (nccl) | 300/300 | 94.41 | 7.9 | 29.69 | 4.36 | 9.72 | 9.29 | cifar16/cifar_rank-2_w16_nccl_s2 |
| cifar16 | cifar | Rank 2 | 3 | 16 (nccl) | 300/300 | 94.28 | 7.9 | 29.47 | 4.35 | 9.72 | 9.11 | cifar16/cifar_rank-2_w16_nccl_s3 |
| cifar16 | cifar | Rank 4 | 1 | 16 (nccl) | 300/300 | 94.28 | 14.9 | 33.45 | 4.34 | 9.69 | 12.90 | cifar16/cifar_rank-4_w16_nccl_s1 |
| cifar16 | cifar | Rank 4 | 2 | 16 (nccl) | 300/300 | 94.71 | 14.9 | 32.97 | 4.35 | 9.72 | 12.36 | cifar16/cifar_rank-4_w16_nccl_s2 |
| cifar16 | cifar | Rank 4 | 3 | 16 (nccl) | 300/300 | 94.43 | 14.9 | 33.17 | 4.33 | 9.68 | 12.68 | cifar16/cifar_rank-4_w16_nccl_s3 |
| cifar16 | cifar | SGD | 1 | 16 (nccl) | 300/300 | 94.08 | 1072.7 | 26.11 | 4.34 | 9.74 | 6.35 | cifar16/cifar_sgd_w16_nccl_s1 |
| cifar16 | cifar | SGD | 2 | 16 (nccl) | 300/300 | 94.08 | 1072.7 | 26.02 | 4.32 | 9.71 | 6.24 | cifar16/cifar_sgd_w16_nccl_s2 |
| cifar16 | cifar | SGD | 3 | 16 (nccl) | 300/300 | 94.08 | 1072.7 | 26.37 | 4.32 | 9.69 | 6.73 | cifar16/cifar_sgd_w16_nccl_s3 |
| lm16 | lm | Rank 1 | 1 | 16 (nccl) | 90/90 | 110.55 | 25.5 | 26.50 | 6.66 | 14.48 | 1.91 | lm16/lm_rank-1_w16_nccl_s1 |
| lm16 | lm | Rank 1 | 2 | 16 (nccl) | 90/90 | 111.02 | 25.5 | 26.50 | 6.66 | 14.47 | 1.91 | lm16/lm_rank-1_w16_nccl_s2 |
| lm16 | lm | Rank 1 | 3 | 16 (nccl) | 90/90 | 111.97 | 25.5 | 26.46 | 6.66 | 14.49 | 1.85 | lm16/lm_rank-1_w16_nccl_s3 |
| lm16 | lm | Rank 2 | 1 | 16 (nccl) | 90/90 | 102.10 | 38.8 | 27.13 | 6.67 | 14.51 | 2.48 | lm16/lm_rank-2_w16_nccl_s1 |
| lm16 | lm | Rank 2 | 2 | 16 (nccl) | 90/90 | 100.91 | 38.8 | 27.11 | 6.66 | 14.44 | 2.54 | lm16/lm_rank-2_w16_nccl_s2 |
| lm16 | lm | Rank 2 | 3 | 16 (nccl) | 90/90 | 101.13 | 38.8 | 27.10 | 6.67 | 14.52 | 2.42 | lm16/lm_rank-2_w16_nccl_s3 |
| lm16 | lm | Rank 4 | 1 | 16 (nccl) | 90/90 | 98.25 | 65.5 | 28.08 | 6.66 | 14.50 | 3.46 | lm16/lm_rank-4_w16_nccl_s1 |
| lm16 | lm | Rank 4 | 2 | 16 (nccl) | 90/90 | 99.09 | 65.5 | 28.05 | 6.67 | 14.47 | 3.44 | lm16/lm_rank-4_w16_nccl_s2 |
| lm16 | lm | Rank 4 | 3 | 16 (nccl) | 90/90 | 99.25 | 65.5 | 28.09 | 6.66 | 14.49 | 3.48 | lm16/lm_rank-4_w16_nccl_s3 |
| lm16 | lm | SGD | 1 | 16 (nccl) | 90/90 | 99.58 | 7882.0 | 28.86 | 6.68 | 14.59 | 4.39 | lm16/lm_sgd_w16_nccl_s1 |
| lm16 | lm | SGD | 2 | 16 (nccl) | 90/90 | 99.37 | 7882.0 | 28.69 | 6.68 | 14.57 | 4.23 | lm16/lm_sgd_w16_nccl_s2 |
| lm16 | lm | SGD | 3 | 16 (nccl) | 90/90 | 99.91 | 7882.0 | 28.90 | 6.68 | 14.58 | 4.44 | lm16/lm_sgd_w16_nccl_s3 |
