#!/bin/bash
# Tier 1: ResNet-18 / CIFAR-10, 1 node x 4 A100, plain DDP vs PowerSGD R in {1,2,4}.
# Runs the 4 configs sequentially in ONE job (one queue wait). Identical seed, per-GPU batch,
# epochs and LR schedule across configs; only --mode/--rank change.
#
# Submit (after tools/prepare_data.py has been run on a login node):
#   sbatch scripts/tier1_resnet18_1node.sbatch                      # defaults below
#   sbatch -q debug -t 00:30:00 --export=ALL,EPOCHS=2 scripts/tier1_resnet18_1node.sbatch   # quick pipeline check
# NOTE: the time limit and EPOCHS are PLACEHOLDERS until the interactive smoke run gives real step times.
#       Cost = 1 node-hour per wall-clock hour. Ask before submitting anything > 2 node-hours.
#SBATCH -A m4431_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH -c 128
#SBATCH -t 01:00:00
#SBATCH -J t1_resnet18_1n
#SBATCH -o logs/%x_%j.out

# (sbatch runs a spooled copy of this file, so $0 is useless; use the fixed project path)
source "${PROJ:-$SCRATCH/powersgd}/scripts/common.sh"
EPOCHS=${EPOCHS:-30}
SEED=${SEED:-1}
COMMON="--model resnet18 --epochs $EPOCHS --seed $SEED --batch-size 128 --start-iter 10"

start_gpu_monitor
launch $COMMON --mode ddp
for R in 1 2 4; do
  launch $COMMON --mode powersgd --rank $R
done
# One profiled step per config (compute vs all-reduce split), 2 epochs, tagged _prof so they are not mixed with results:
# launch $COMMON --epochs 1 --mode ddp --profile-step 60 --run-name prof_ddp
# launch $COMMON --epochs 1 --mode powersgd --rank 2 --profile-step 60 --run-name prof_psgd_r2
finish
