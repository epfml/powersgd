#!/bin/bash
# Shared helpers, sourced by the *.sbatch files. UNTESTED on Perlmutter until the first interactive run.
# Layout assumption (override with PROJ=...): $SCRATCH/powersgd is the project dir (clone of this repo).
set -u
PROJ=${PROJ:-$SCRATCH/powersgd}
cd "$PROJ"
mkdir -p logs results

# NERSC PyTorch module. Version TBD: run `module avail pytorch` and pin the one we verify
# (needs float8 dtypes, torch>=2.1, and multi-node NCCL). Override: PYTORCH_MODULE=pytorch/2.x.y
module load ${PYTORCH_MODULE:-pytorch}

export DATA_DIR=${DATA_DIR:-$SCRATCH/powersgd/data}   # CIFAR-10 + Shakespeare; compute nodes have no internet
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export GPUS_PER_NODE=4

# GPU utilization logger: one CSV per node for the whole job (brief section 6).
start_gpu_monitor() {
  srun --nodes="$SLURM_NNODES" --ntasks-per-node=1 --overlap --cpu-bind=none bash -c \
    'nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw --format=csv -l 1 > logs/gpu_${SLURM_JOB_ID}_$(hostname).csv' &
  GPU_MON_PID=$!
}
stop_gpu_monitor() { kill "${GPU_MON_PID:-0}" 2>/dev/null || true; }

# launch <train_ddp.py args...>: one torchrun per node, c10d rendezvous on the first node.
# Uses a fresh rendezvous id and port per call so back-to-back runs in one job do not collide.
launch() {
  local master; master=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
  RDZV_PORT=$(( ${RDZV_PORT:-29500} + 1 ))
  srun --nodes="$SLURM_NNODES" --ntasks-per-node=1 --cpu-bind=none \
    torchrun --nnodes="$SLURM_NNODES" --nproc_per_node=$GPUS_PER_NODE \
             --rdzv_backend=c10d --rdzv_endpoint="$master:$RDZV_PORT" --rdzv_id="${SLURM_JOB_ID}_$RDZV_PORT" \
    train_ddp.py "$@"
}

# Record Slurm accounting for this job (call at the end; sacct may lag a few minutes, so
# also run `python tools/update_budget.py` later from a login node).
finish() { stop_gpu_monitor; sacct -j "$SLURM_JOB_ID" --format=JobID,Elapsed,NNodes,AllocTRES%60,State || true; }
