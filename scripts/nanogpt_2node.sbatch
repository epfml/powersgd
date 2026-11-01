#!/bin/bash
# NanoGPT, 2 nodes (8 GPUs): does compression help once all-reduce crosses the network?
# Cost: about 15 to 25 min on 2 nodes, roughly 0.5 to 0.8 node-hours.
#SBATCH -A m4431_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -N 2
#SBATCH --gpus-per-node=4
#SBATCH -c 128
#SBATCH -t 00:40:00
#SBATCH -J gpt_2n
#SBATCH -o logs/%x_%j.out

source "${PROJ:-$SCRATCH/powersgd}/scripts/common.sh"
COMMON="--model nanogpt --seed 1 --max-steps 2000 --eval-interval 250 --start-iter 10"

start_gpu_monitor
launch $COMMON --mode ddp --run-name gpt_ddp_2node
launch $COMMON --mode powersgd --rank 2 --run-name gpt_r2_2node
launch $COMMON --mode powersgd --rank 4 --run-name gpt_r4_2node
finish
