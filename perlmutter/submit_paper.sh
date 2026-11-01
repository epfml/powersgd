#!/bin/bash
# Submit the PowerSGD paper reproduction to Perlmutter, one Slurm job per run.
#
#   export NERSC_ACCOUNT=m1234            # your class/project allocation
#   ./perlmutter/submit_paper.sh smoke     # ~5 min debug-queue sanity check, run this first
#   ./perlmutter/submit_paper.sh table3    # then any of the experiments below
#
# Experiments (16 GPUs = 4 nodes per run unless noted; seeds from $SEEDS, default "1 2 3"):
#   smoke        1 node, 1-2 epochs of SGD / Rank 2 / Signum on CIFAR and Rank 4 on WikiText-2
#   table3       Table 3: SGD, Rank 1, 2, 4 on CIFAR-10 (300 ep) and WikiText-2 LSTM (90 ep)
#   table1       Table 1: unbiased rank 1/2 without error feedback (+ SGD, Rank 1, 2)
#   table2       Table 2: Rank 2 with vs. without warm start
#   table4       Table 4: Rank 7/2 vs Random Block, Random K, Sign+Norm, Top K (with EF)
#   table6       Table 6: Atomo and Signum vs SGD and Rank 2 on CIFAR-10
#   table7       Table 7: Signum vs SGD and Rank 4 on WikiText-2
#   scaling      Figure 3 / Table 5: 10-epoch timing runs, 1-16 GPUs, NCCL and Gloo (seed 1)
#   all          everything above except smoke
#
# Runs that already finished (DONE file) or are already queued are skipped, so it is safe to
# re-run this script after jobs fail or time out. Optional env vars: SEEDS, QOS, RESERVATION.
set -eo pipefail
REPO_DIR=$(cd "$(dirname "$0")/.." && pwd)
export REPO_DIR
source "$REPO_DIR/perlmutter/env.sh"
: "${NERSC_ACCOUNT:?export NERSC_ACCOUNT=<your NERSC project, e.g. m1234>}"
SEEDS=${SEEDS:-"1 2 3"}
LOG_DIR=$RESULTS/slurm-logs
mkdir -p "$LOG_DIR"
QUEUED=$(squeue --me -h -o %j || true)

# submit <group> <task> <preset> <gpus> <seed> <backend> <time> <qos> [run_experiment.py args]
submit() {
    local group=$1 task=$2 preset=$3 workers=$4 seed=$5 backend=$6 time=$7 qos=$8
    shift 8
    local slug name out
    slug=$(tr '[:upper:]' '[:lower:]' <<<"$preset" | sed -E 's/[^a-z0-9]+/-/g; s/^-|-$//g')
    name="${task}_${slug}_w${workers}_${backend}_s${seed}"
    out=$RESULTS/$group/$name
    if [[ -f $out/DONE ]]; then
        echo "done     $group/$name"
        return
    fi
    local job_name="${group}_$name"
    if grep -qx "$job_name" <<<"$QUEUED"; then
        echo "queued   $group/$name"
        return
    fi
    local job
    job=$(env TASK="$task" PRESET="$preset" SEED="$seed" WORKERS="$workers" \
        TASKS_PER_NODE=$((workers < 4 ? workers : 4)) BACKEND="$backend" OUT="$out" \
        EXTRA_ARGS="$*" \
        sbatch --parsable -A "$NERSC_ACCOUNT" -q "${QOS:-$qos}" -t "$time" \
        -N $(((workers + 3) / 4)) --ntasks-per-node=$((workers < 4 ? workers : 4)) \
        ${RESERVATION:+--reservation=$RESERVATION} \
        -J "$job_name" -o "$LOG_DIR/%x-%j.out" "$REPO_DIR/perlmutter/run.sbatch")
    QUEUED+=$'\n'"$job_name"
    echo "submitted $group/$name (job $job)"
}

cifar() {  # cifar <time> <preset>...
    local time=$1
    shift
    for preset in "$@"; do
        for seed in $SEEDS; do
            submit cifar16 cifar "$preset" 16 "$seed" nccl "$time" regular
        done
    done
}

lm() {  # lm <time> <preset>...
    local time=$1
    shift
    for preset in "$@"; do
        for seed in $SEEDS; do
            submit lm16 lm "$preset" 16 "$seed" nccl "$time" regular
        done
    done
}

run_experiment() {
    case $1 in
    smoke)
        submit smoke cifar "SGD" 4 1 nccl 00:15:00 debug --epochs 2
        submit smoke cifar "Rank 2" 4 1 nccl 00:15:00 debug --epochs 2
        submit smoke cifar "Signum" 4 1 nccl 00:15:00 debug --epochs 1
        submit smoke lm "Rank 4" 4 1 nccl 00:15:00 debug --epochs 1
        ;;
    table3)
        cifar 01:30:00 "SGD" "Rank 1" "Rank 2" "Rank 4"
        lm 01:00:00 "SGD" "Rank 1" "Rank 2" "Rank 4"
        ;;
    table1)
        cifar 01:30:00 "SGD" "Rank 1" "Rank 2" "Unbiased Rank 1" "Unbiased Rank 2"
        ;;
    table2)
        cifar 01:30:00 "Rank 2" "Rank 2 (no warm start)"
        ;;
    table4)
        cifar 01:30:00 "SGD" "Rank 7" "Rank 2" "Random Block (rank 7)" "Random Block (rank 2)" \
            "Sign+Norm" "Top K (medium)" "Top K (high)"
        cifar 03:00:00 "Random K (rank 7)" "Random K (rank 2)"
        ;;
    table6)
        cifar 01:30:00 "SGD" "Rank 2" "Signum"
        cifar 06:00:00 "Atomo (rank 2)"
        ;;
    table7)
        lm 01:00:00 "SGD" "Rank 4" "Signum"
        ;;
    scaling)
        for backend in nccl gloo; do
            for preset in "SGD" "Rank 2" "Signum"; do
                for workers in 1 2 4 8 16; do
                    submit scaling cifar "$preset" "$workers" 1 "$backend" 00:30:00 regular --epochs 10
                done
            done
        done
        ;;
    all)
        for experiment in table3 table1 table2 table4 table6 table7 scaling; do
            run_experiment "$experiment"
        done
        ;;
    *)
        echo "Unknown experiment '$1'. See the header of $0 for options." >&2
        exit 1
        ;;
    esac
}

[[ $# -gt 0 ]] || { sed -n '2,25p' "$0"; exit 1; }
for experiment in "$@"; do
    run_experiment "$experiment"
done
