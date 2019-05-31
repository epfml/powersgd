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
seed = 42

basename = os.path.basename(__file__).replace(".py", "")
experiment = f"neurips19_{basename}"
description = """
Tuning Top K /w EF for Cifar10/ResNet18
""".strip()

code_package = shared.upload_code()

registered_ids = []

reducer = "Top K (EF)"
n_workers = 16
for compression in [244, 72, 32]:
    for learning_rate in [0.1, 0.05]:
        name = f"{reducer}_{n_workers:02d}workers_lr{learning_rate}_{compression}x"
        if mongo.job.count_documents({"job": name, "experiment": experiment}) > 0:
            # We have this one already
            continue
        job_id = register_job(
            user="vogels",
            project="sgd",
            experiment=experiment,
            job=name,
            n_workers=n_workers,
            priority=10,
            config_overrides={
                "seed": seed,
                "distributed_backend": "nccl",
                "optimizer_scale_lr_with_factor": n_workers,
                "num_epochs": 300,
                "log_verbosity": 1,
                **shared.sgd_config(learning_rate, momentum=0.9, weight_decay=0.0001),
                **shared.optimizer_config(reducer),
                "optimizer_reducer_compression": 1 / compression,
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
