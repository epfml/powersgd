#!/usr/bin/env python3
"""
Launch one experiment from the PowerSGD paper (Vogels et al., NeurIPS 2019).

Works under `srun` (reads SLURM_PROCID / SLURM_NTASKS / SLURM_LOCALID) and `torchrun`
(reads RANK / WORLD_SIZE / LOCAL_RANK). Hyperparameters follow hyperparameters.md.

Examples:
    srun python run_experiment.py --task cifar --preset "Rank 2" --seed 1 --out $RESULTS/x
    torchrun --nproc-per-node 2 run_experiment.py --task cifar --preset SGD --backend gloo \
        --epochs 1 --out /tmp/smoke
"""

import argparse
import json
import os
import sys
import time

CIFAR_BASE = dict(
    task="Cifar",
    task_architecture="ResNet18",
    average_reset_epoch_interval=30,
    fix_conv_weight_norm=False,
    num_epochs=300,
    num_train_tracking_batches=1,
    optimizer_batch_size=128,
    optimizer_conv_learning_rate=0.1,
    optimizer_learning_rate=0.1,
    optimizer_decay_at_epochs=[150, 250],
    optimizer_decay_with_factor=10.0,
    optimizer_momentum_type="nesterov",
    optimizer_momentum=0.9,
    optimizer_mom_before_reduce=False,
    optimizer_wd_before_reduce=False,
    optimizer_scale_lr_with_warmup_epochs=5,
    optimizer_weight_decay_conv=0.0001,
    optimizer_weight_decay_other=0.0001,
    optimizer_weight_decay_bn=0.0,
    log_verbosity=1,
)

LM_BASE = dict(
    task="LanguageModeling",
    average_reset_epoch_interval=30,
    fix_conv_weight_norm=False,
    num_epochs=90,
    num_train_tracking_batches=1,
    optimizer_batch_size=64,
    optimizer_conv_learning_rate=1.25,
    optimizer_learning_rate=1.25,
    optimizer_decay_at_epochs=[60, 80],
    optimizer_decay_with_factor=10.0,
    optimizer_momentum_type="nesterov",
    optimizer_momentum=0.0,
    optimizer_mom_before_reduce=False,
    optimizer_wd_before_reduce=False,
    optimizer_scale_lr_with_warmup_epochs=5,
    optimizer_weight_decay_conv=0.0,
    optimizer_weight_decay_other=0.0,
    optimizer_weight_decay_bn=0.0,
    log_verbosity=1,
)


def sgd():
    return dict(optimizer_reducer="ExactReducer", optimizer_memory=False)


def powersgd(rank, warm_start=True):
    return dict(
        optimizer_reducer="RankKReducer",
        optimizer_memory=True,
        optimizer_reducer_rank=rank,
        optimizer_reducer_reuse_query=warm_start,
        optimizer_reducer_n_power_iterations=0,
    )


def unbiased_rank(rank):
    return dict(
        optimizer_reducer="HalfRankKReducer",
        optimizer_memory=False,
        optimizer_reducer_rank=rank,
        optimizer_learning_rate=0.0125,
        optimizer_conv_learning_rate=0.0125,
    )


def signum(lr):
    return dict(
        optimizer_reducer="SignSGDwithMajorityVoteReducer",
        optimizer_memory=False,
        optimizer_mom_before_reduce=True,
        optimizer_momentum_type="exponential_moving_average",
        optimizer_learning_rate=lr,
        optimizer_conv_learning_rate=lr,
    )


CIFAR_PRESETS = {
    "SGD": sgd(),
    "Rank 1": powersgd(1),
    "Rank 2": powersgd(2),
    "Rank 4": powersgd(4),
    "Rank 7": powersgd(7),
    "Rank 2 (no warm start)": powersgd(2, warm_start=False),
    "Unbiased Rank 1": unbiased_rank(1),
    "Unbiased Rank 2": unbiased_rank(2),
    "Random Block (rank 7)": dict(
        optimizer_reducer="RandomSparseBlockReducer", optimizer_memory=True, optimizer_reducer_rank=7
    ),
    "Random Block (rank 2)": dict(
        optimizer_reducer="RandomSparseBlockReducer", optimizer_memory=True, optimizer_reducer_rank=2
    ),
    "Random K (rank 7)": dict(
        optimizer_reducer="RandomSparseReducer", optimizer_memory=True, optimizer_reducer_rank=7
    ),
    "Random K (rank 2)": dict(
        optimizer_reducer="RandomSparseReducer", optimizer_memory=True, optimizer_reducer_rank=2
    ),
    "Sign+Norm": dict(optimizer_reducer="SignAndNormReducer", optimizer_memory=True),
    # hyperparameters.md lists these two compression values swapped; these match the
    # 32 MB (medium) and 8 MB (high) data volumes of Table 4.
    "Top K (medium)": dict(
        optimizer_reducer="TopKReducer", optimizer_memory=True, optimizer_reducer_compression=1 / 32
    ),
    "Top K (high)": dict(
        optimizer_reducer="TopKReducer", optimizer_memory=True, optimizer_reducer_compression=1 / 136
    ),
    "Atomo (rank 2)": dict(
        optimizer_reducer="AtomoReducer", optimizer_memory=False, optimizer_reducer_rank=2
    ),
    "Signum": signum(5e-5),
}

LM_PRESETS = {
    "SGD": sgd(),
    "Rank 1": powersgd(1),
    "Rank 2": powersgd(2),
    "Rank 4": powersgd(4),
    "Signum": signum(1e-5),
}

TASKS = {"cifar": (CIFAR_BASE, CIFAR_PRESETS), "lm": (LM_BASE, LM_PRESETS)}


def distributed_env():
    """(rank, world_size, local_rank) from torchrun or Slurm, defaulting to 1 process."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        return (
            int(os.environ["RANK"]),
            int(os.environ["WORLD_SIZE"]),
            int(os.environ.get("LOCAL_RANK", 0)),
        )
    if "SLURM_PROCID" in os.environ:
        return (
            int(os.environ["SLURM_PROCID"]),
            int(os.environ["SLURM_NTASKS"]),
            int(os.environ["SLURM_LOCALID"]),
        )
    return 0, 1, 0


def parse_value(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--task", choices=TASKS, default="cifar")
    parser.add_argument("--preset", required=True, help="e.g. 'SGD', 'Rank 2'; see --list")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", required=True, help="output directory for this run")
    parser.add_argument("--epochs", type=int, default=None, help="override number of epochs")
    parser.add_argument("--backend", default="nccl", choices=["nccl", "gloo"])
    parser.add_argument(
        "--lr-scale",
        type=float,
        default=None,
        help="LR scaling factor after warmup (default: number of workers, as in the paper)",
    )
    parser.add_argument(
        "--set", nargs="*", default=[], metavar="KEY=VALUE", help="extra train.config overrides"
    )
    parser.add_argument("--force", action="store_true", help="rerun even if already finished")
    args = parser.parse_args()

    base, presets = TASKS[args.task]
    if args.preset not in presets:
        sys.exit(f"Unknown preset {args.preset!r} for task {args.task}. Options: {list(presets)}")

    rank, world_size, local_rank = distributed_env()
    done_file = os.path.join(args.out, "DONE")
    if os.path.exists(done_file) and not args.force:
        if rank == 0:
            print(f"{args.out} already finished, skipping (use --force to rerun)")
        return

    os.makedirs(args.out, exist_ok=True)
    os.environ.setdefault("DATA", os.path.join(os.environ.get("SCRATCH", "."), "powersgd"))
    if world_size == 1:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import train

    train.config.update(base)
    train.config.update(presets[args.preset])
    train.config.update(
        seed=args.seed,
        rank=rank,
        local_rank=local_rank,
        n_workers=world_size,
        distributed_backend=args.backend,
        distributed_init_method="env://",
        optimizer_scale_lr_with_factor=args.lr_scale,
        reducer_name=args.preset,
    )
    if args.epochs is not None:
        train.config["num_epochs"] = args.epochs
    for item in args.set:
        key, value = item.split("=", 1)
        train.config[key] = parse_value(value)

    train.output_dir = args.out
    metrics_path = os.path.join(args.out, "metrics.jsonl")

    if rank == 0:
        with open(os.path.join(args.out, "config.json"), "w") as fp:
            json.dump(
                {**train.config, "task_name": args.task, "world_size": world_size}, fp, indent=2
            )
        metrics_file = open(metrics_path, "w")

        def metric(name, values, tags={}):
            record = {"name": name, "time": time.time(), **values, **tags}
            metrics_file.write(json.dumps(record) + "\n")
            metrics_file.flush()
            if name != "timer":
                train.log_metric(name, values, tags)

        train.metric = metric
    else:
        train.metric = lambda *args, **kwargs: None

    start = time.time()
    train.main()

    if rank == 0:
        metrics_file.close()
        with open(done_file, "w") as fp:
            json.dump({"wallclock_seconds": time.time() - start}, fp)


if __name__ == "__main__":
    main()
