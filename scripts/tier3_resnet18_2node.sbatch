#!/bin/bash
# Step 3: 2 nodes x 4 A100 = 8 GPUs, plain DDP vs PowerSGD R=2 on ResNet-18 / CIFAR-10.
# Everything except --mode/--rank matches the 1-node sweep (same seed, per-GPU batch 128, epochs). The global batch is
# 2x the 1-node one; LR scales linearly with global batch (see --lr-ref-batch), so compare 2-node DDP vs 2-node PowerSGD
# for speed, and note that 1-node vs 2-node accuracy is NOT an apples-to-apples comparison.
#
#   sbatch scripts/tier3_resnet18_2node.sbatch
#   sbatch -q debug -t 00:30:00 --export=ALL,EPOCHS=2 scripts/tier3_resnet18_2node.sbatch    # pipeline check
# COST: 2 node-hours per wall-clock hour. The time limit below is a PLACEHOLDER; set it from the measured
# s/epoch (scripts/time_epoch_1node.sh) and get an OK before submitting anything > 2 node-hours.
#SBATCH -A m4431_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -N 2
#SBATCH --gpus-per-node=4
#SBATCH -c 128
#SBATCH -t 01:00:00
#SBATCH -J t3_resnet18_2n
#SBATCH -o logs/%x_%j.out

source "${PROJ:-$SCRATCH/powersgd}/scripts/common.sh"
EPOCHS=${EPOCHS:-30}
SEED=${SEED:-1}
COMMON="--model resnet18 --epochs $EPOCHS --seed $SEED --batch-size 128 --start-iter 10"

start_gpu_monitor
launch $COMMON --mode ddp
launch $COMMON --mode powersgd --rank 2
finish
