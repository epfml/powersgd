#!/bin/bash
#SBATCH -C gpu
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
# Runs one paper-code/run_experiment.py job. Submitted by submit_paper.sh, which sets
# REPO_DIR TASK PRESET SEED WORKERS TASKS_PER_NODE BACKEND OUT EXTRA_ARGS.
# Every rank sees all 4 GPUs of its node and picks cuda:$SLURM_LOCALID, so NCCL can use NVLink.
set -eo pipefail
source "$REPO_DIR/perlmutter/env.sh"

export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500
export OMP_NUM_THREADS=16
if [[ $BACKEND == gloo ]]; then
    export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-hsn0}
fi

echo "job $SLURM_JOB_ID on $SLURM_JOB_NODELIST: $TASK / $PRESET / seed $SEED / $WORKERS GPUs ($BACKEND)"
cd "$REPO_DIR/paper-code"
srun --ntasks="$WORKERS" --ntasks-per-node="$TASKS_PER_NODE" --cpus-per-task=32 --cpu-bind=cores \
    python -u run_experiment.py --task "$TASK" --preset "$PRESET" --seed "$SEED" \
    --backend "$BACKEND" --out "$OUT" $EXTRA_ARGS
