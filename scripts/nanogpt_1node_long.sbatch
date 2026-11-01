#!/bin/bash
# NanoGPT, 1 node: do PowerSGD ranks 2/4 catch up to DDP's best loss given more steps?
# Cost: about 20 min on 1 node, roughly 0.35 node-hours.
#SBATCH -A m4431_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH -c 128
#SBATCH -t 00:40:00
#SBATCH -J gpt_1n_long
#SBATCH -o logs/%x_%j.out

source "${PROJ:-$SCRATCH/powersgd}/scripts/common.sh"
COMMON="--model nanogpt --seed 1 --eval-interval 500 --start-iter 10"

start_gpu_monitor
launch $COMMON --mode ddp --max-steps 1000 --run-name gpt_ddp_1000_1node
launch $COMMON --mode powersgd --rank 2 --max-steps 6000 --run-name gpt_r2_6000_1node
launch $COMMON --mode powersgd --rank 4 --max-steps 6000 --run-name gpt_r4_6000_1node
finish
