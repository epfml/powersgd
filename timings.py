#!/usr/bin/env python3

"""
This is used to measure the performance of collective communication operations
in pytorch
"""

import datetime
import os

import numpy as np
import torch

from timer import Timer

config = dict(
    distributed_backend="nccl",
    device="cuda",
    rank=1,
    n_workers=2,
    distributed_init_file=None,
    message_sizes=[
        4,
        16,
        64,
        256,
        1024,
        4096,
        16384,
        65536,
        262_144,
        1_048_576,
        4_194_304,
        16_777_216,
        67_108_864,
        268_435_456,
    ],  # in bytes
    repetitions=4,
)

output_dir = "./output.tmp"  # will be overwritten by run.py


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    timer = Timer(verbosity_level=2, log_fn=metric)

    if torch.distributed.is_available():
        if config["distributed_init_file"] is None:
            config["distributed_init_file"] = os.path.join(output_dir, "dist_init")
        print(
            "Distributed init: rank {}/{} - {}".format(
                config["rank"], config["n_workers"], config["distributed_init_file"]
            )
        )
        torch.distributed.init_process_group(
            backend=config["distributed_backend"],
            init_method="file://" + os.path.abspath(config["distributed_init_file"]),
            timeout=datetime.timedelta(0, 1800),
            world_size=config["n_workers"],
            rank=config["rank"],
        )

    # All reduce
    for message_size in config["message_sizes"]:
        # Create data
        data = torch.randn(message_size // 4, device=config["device"])
        for repetition in range(config["repetitions"]):
            # Wait until everyone is ready
            torch.distributed.barrier()
            with timer(f"all_reduce_{message_size}"):
                torch.distributed.all_reduce(data)

    # All gather
    all_gather_device = "cuda" if config["distributed_backend"] == "nccl" else "cpu"
    for message_size in config["message_sizes"]:
        # Create data
        data = torch.randn(message_size // 4, device=all_gather_device)
        other_data = [torch.empty_like(data) for i in range(config["n_workers"])]
        for repetition in range(config["repetitions"]):
            # Wait until everyone is ready
            torch.distributed.barrier()
            with timer(f"all_gather_{message_size}"):
                torch.distributed.all_gather(other_data, data)

    # Gather
    gather_device = "cuda" if config["distributed_backend"] == "nccl" else "cpu"
    if config["distributed_backend"] != "nccl":
        for message_size in config["message_sizes"]:
            # Create data
            data = torch.randn(message_size // 4, device=gather_device)
            if config["rank"] == 0:
                other_data = [torch.empty_like(data) for i in range(config["n_workers"])]
            else:
                other_data = []
            for repetition in range(config["repetitions"]):
                # Wait until everyone is ready
                torch.distributed.barrier()
                with timer(f"gather_{message_size}"):
                    torch.distributed.gather(data, other_data, dst=0)

    # Broadcast
    for message_size in config["message_sizes"]:
        # Create data
        data = torch.randn(message_size // 4, device=config["device"])
        for repetition in range(config["repetitions"]):
            # Wait until everyone is ready
            torch.distributed.barrier()
            with timer(f"broadcast_{message_size}"):
                torch.distributed.broadcast(data, src=0)

    print(timer.summary())
    if config["rank"] == 0:
        timer.save_summary(os.path.join(output_dir, "timer_summary.json"))


def log_info(info_dict):
    """Add any information to MongoDB
       This function will be overwritten when called through run.py"""
    pass


def log_metric(name, values, tags={}):
    """Log timeseries data
       This function will be overwritten when called through run.py"""
    value_list = []
    for key in sorted(values.keys()):
        value = values[key]
        value_list.append(f"{key}:{value:7.3f}")
    values = ", ".join(value_list)
    tag_list = []
    for key, tag in tags.items():
        tag_list.append(f"{key}:{tag}")
    tags = ", ".join(tag_list)
    print("{name:30s} - {values} ({tags})".format(name=name, values=values, tags=tags))


def info(*args, **kwargs):
    if config["rank"] == 0:
        log_info(*args, **kwargs)


def metric(*args, **kwargs):
    if config["rank"] == 0:
        log_metric(*args, **kwargs)


if __name__ == "__main__":
    main()
