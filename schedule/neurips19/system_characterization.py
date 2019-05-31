#!/usr/bin/env python3

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
seed = 14
n_workers = 16
num_epochs = 3

experiment = "neurips19_system_characterization_3"
description = """
Version of neurips19_system_characterization with more repetitions for smoother curves.
""".strip()

code_package = shared.upload_code()

registered_ids = []


priority = 10


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


register_barrier()
sleep(0.1)

for backend in ["nccl", "gloo"]:
    for n_workers in [2, 4, 8, 16]:
        name = f"time_{n_workers}workers_{backend}"
        if mongo.job.count_documents({"job": name, "experiment": experiment}) > 0:
            # We have this one already
            continue
        job_id = register_job(
            user="vogels",
            project="sgd",
            experiment=experiment,
            job=name,
            n_workers=n_workers,
            priority=priority,
            config_overrides={
                "distributed_backend": backend,
                "repetitions": 20,
                "device": "cuda",
                "n_workers": n_workers,
            },
            runtime_environment={"clone": {"code_package": code_package}, "script": "timings.py"},
            annotations={"description": description},
        )
        print("{} - {}".format(job_id, name))
        registered_ids.append(job_id)

        sleep(0.1)
        register_barrier()
        sleep(0.1)


# kubernetes_schedule_job_queue(
#     registered_ids,
#     "ic-registry.epfl.ch/mlo/vogels_experiment",
#     volumes=["pv-mlodata1"],
#     gpus=gpus,
#     parallelism=4,
#     results_dir="/pv-mlodata1/vogels/results",
#     environment_variables={"DATA": "/pv-mlodata1/vogels"},
# )
