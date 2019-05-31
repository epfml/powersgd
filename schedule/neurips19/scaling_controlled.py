#!/usr/bin/env python3

import os
from time import sleep

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
Scaling plots still look weird. We need error bars (so will do three repetitions now)
and we will control device placement so that we always utilize all GPUs on a machine for each experiment
""".strip()

code_package = shared.upload_code()

registered_ids = []

priority = 5


def register_barrier():
    """So we don't have multiple jobs running simulateously"""
    register_job(
        user="vogels",
        project="sgd",
        experiment=experiment,
        job="barrier",
        priority=priority,
        n_workers=16,
        config_overrides={},
        runtime_environment={"clone": {"code_package": code_package}, "script": "barrier.py"},
        annotations={"description": description},
    )


for backend in ["nccl", "gloo"]:
    for reducer in ["SGD", "Rank 2 (EF)", "Signum"]:
        for log_level in [1, 2]:
            for seed in [1, 2, 3]:
                for n_workers in [1, 2, 4, 8, 16]:
                    name = f"{reducer}_{n_workers:02d}workers_{backend}_l{log_level}_s{seed}"
                    if mongo.job.count_documents({"job": name, "experiment": experiment}) > 0:
                        # We have this one already
                        continue
                    sleep(0.5)
                    job_id = register_job(
                        user="vogels",
                        project="sgd",
                        experiment=experiment,
                        job=name,
                        priority=seed + (100 if backend == "nccl" else 0),
                        n_workers=n_workers,
                        config_overrides={
                            "seed": 10000 + seed,
                            "optimizer_scale_lr_with_factor": n_workers,
                            "distributed_backend": backend,
                            "log_verbosity": log_level,
                            "num_epochs": 10,
                            **shared.sgd_config(0.1, momentum=0.9, weight_decay=0.0001),
                            **shared.optimizer_config(reducer),
                        },
                        runtime_environment={
                            "clone": {"code_package": code_package},
                            "script": "train.py",
                        },
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
