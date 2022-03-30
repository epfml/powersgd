#!/usr/bin/env python3

import os

import numpy as np

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
experiment = f"neurips19_{basename}"
description = """
How consistently as good are we compared to SGD?
""".strip()

code_package = shared.upload_code()

registered_ids = []

reducer = "SGD"
n_workers = 16

learning_rate = 0.1

for reducer in [
    "SGD",
    "Rank 1 (EF)",
    "Rank 2 (EF)",
    "Rank 4 (EF)",
    "Rank 7 (EF)",
    "Rank 2 w/o reuse (EF)",
    "HQ Rank 2 w/o reuse (EF)",
    "Random K (rank 2) (EF)",
    "Random K (rank 7) (EF)",
    "Random Block (rank 2) (EF)",
    "Random Block (rank 7) (EF)",
    "Sign+Norm (EF)",
    "Top K (136x) (EF)",
    "Top K (32x) (EF)",
    "Top K (rank 2) (EF)",
    "Top K (rank 7) (EF)",
    "Signum",
    "Unbiased Rank 1",
    "Unbiased Rank 2",
    "Atomo (rank 2)",
    "Atomo (rank 1)",
]:
    for seed in [1, 2, 3]:
        name = f"{reducer}_{n_workers:02d}workers_seed{seed}"
        if mongo.job.count_documents({"job": name, "experiment": experiment}) > 0:
            # We have this one already
            continue

        if reducer == "Signum":
            learning_rate = 0.00005
        elif reducer == "Atomo (rank 1)":
            learning_rate = 0.1
        elif reducer == "Atomo (rank 2)":
            learning_rate = 0.1
        elif "Unbiased Rank" in reducer:
            learning_rate = 0.0125
        else:
            learning_rate = 0.1

        job_id = register_job(
            user="vogels",
            project="sgd",
            experiment=experiment,
            job=name,
            n_workers=n_workers,
            priority=seed * 10,
            config_overrides={
                "seed": seed,
                "distributed_backend": "nccl",
                "optimizer_scale_lr_with_factor": n_workers,
                "num_epochs": 300,
                "log_verbosity": 1,
                **shared.sgd_config(learning_rate, momentum=0.9, weight_decay=0.0001),
                **shared.optimizer_config(reducer),
            },
            runtime_environment={"clone": {"code_package": code_package}, "script": "train.py"},
            annotations={"description": description},
        )
        print("{} - {}".format(job_id, name))
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
