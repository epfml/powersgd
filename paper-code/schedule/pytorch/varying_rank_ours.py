#!/usr/bin/env python3

import os

import numpy as np
import re

import shared
from jobmonitor.api import (
    kubernetes_schedule_job,
    kubernetes_schedule_job_queue,
    register_job,
    upload_code_package,
)
from jobmonitor.connections import mongo

gpus = 1
basename = os.path.basename(__file__).replace(".py", "")
experiment = f"pytorch_{basename}"
description = """
Our implementation. Should be compared to Pytorch
""".strip()

code_package = shared.upload_code()

registered_ids = []

n_workers = 16

def reducer_config(name):
    match = re.match("powersgd-rank(\d+)", name)
    if match:
        rank = int(match.group(1))
        return shared.optimizer_config(f"Rank {rank} (EF)")
    
    match = re.match("all-reduce", name)
    if match:
        return shared.optimizer_config(f"SGD")
    
    raise ValueError(f"Illegal reducer name {name}")

for seed in [1, 2]:
    for reducer in ["powersgd-rank4", "powersgd-rank2", "powersgd-rank1", "all-reduce"]:
        name = f"{n_workers:02d}workers_{reducer}_seed{seed}"
        # if mongo.job.count_documents({"job": name, "experiment": experiment}) > 0:
        #     # We have this one already
        #     continue

        job_id = register_job(
            user="vogels",
            project="powersgd",
            experiment=experiment,
            job=name,
            n_workers=n_workers,
            priority=seed * 10,
            config_overrides={
                "seed": seed,
                "distributed_backend": "nccl",
                "optimizer_scale_lr_with_factor": n_workers,
                # "num_epochs": 2,
                "log_verbosity": 1,
                **shared.sgd_config(0.1, momentum=0.9, weight_decay=0.0001),
                **reducer_config(reducer),
            },
            runtime_environment={"clone": {"code_package": code_package}, "script": "train.py"},
            annotations={"description": description},
        )
        prio = "high" if seed == 1 else "med"
        print(f"sbatch --ntasks {n_workers} --gpus-per-task 1 --cpus-per-task 4 --wrap \"srun jobrun {job_id} --mpi\" --job-name \"{name}\" -p {prio}")
        registered_ids.append(job_id)


# kubernetes_schedule_job_queue(
#     registered_ids,
#     "ic-registry.epfl.ch/mlo/vogels_experiment",
#     volumes=["pv-mlodata1"],
#     gpus=gpus,
#     parallelism=4,
#     results_dir="/pv-mlodata1/vogels/results",
#     environment_variables={"DATA": "/pv-mlodata1/vogels"},
# )
