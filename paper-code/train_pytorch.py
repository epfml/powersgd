#!/usr/bin/env python3

import datetime
import os
import re
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.algorithms.ddp_comm_hooks.default_hooks as default_hooks
import torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook as powerSGD

import tasks
from mean_accumulator import MeanAccumulator
from timer import Timer

"""
When you run this script, it uses the default parameters below.
To change them, you can make another script, say `experiment.py`
and write, e.g.
```
import train
train.config["num_epochs"] = 200
train.config["n_workers"] = 4
train.config["rank"] = 0
train.main()
```

The configuration overrides we used for all our experiments can be found in the folder schedule/neurips19.
"""

config = dict(
    distributed_backend="nccl",
    fix_conv_weight_norm=False,
    num_epochs=300,
    checkpoints=[],
    num_train_tracking_batches=1,
    optimizer_batch_size=128,  # per worker
    optimizer_learning_rate=0.1,  # Tuned for batch size 128 (single worker)
    optimizer_conv_learning_rate=0.1,  # tuned for batch size 128
    optimizer_decay_with_factor=10.0,
    optimizer_decay_at_epochs=[150, 250],
    optimizer_momentum_type="nesterov",
    optimizer_momentum=0.9,
    optimizer_scale_lr_with_factor=None,  # set to override world_size as a factor
    optimizer_scale_lr_with_warmup_epochs=5,  # scale lr by world size
    optimizer_mom_before_reduce=False,
    optimizer_wd_before_reduce=False,
    optimizer_weight_decay_conv=0.0001,
    optimizer_weight_decay_other=0.0001,
    optimizer_weight_decay_bn=0.0,
    start_powerSGD_iter=2,
    use_powersgd=False,
    optimizer_reducer_rank=0,
    task="Cifar",
    task_architecture="ResNet18",
    seed=42,
    rank=0,
    n_workers=1,
    distributed_init_file=None,
    log_verbosity=2,
    fp16_compression=False,
)

output_dir = "./output.tmp"  # will be overwritten by run.py


def main():
    torch.manual_seed(config["seed"] + config["rank"])
    np.random.seed(config["seed"] + config["rank"])

    assert config["optimizer_mom_before_reduce"] == False
    assert config["optimizer_wd_before_reduce"] == False

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    timer = Timer(verbosity_level=config["log_verbosity"], log_fn=metric)

    assert torch.distributed.is_available()
    if config["distributed_init_file"] is None:
        config["distributed_init_file"] = os.path.join(output_dir, "dist_init")
    print(
        "Distributed init: rank {}/{} - {}".format(
            config["rank"], config["n_workers"], config["distributed_init_file"]
        )
    )
    process_group = torch.distributed.init_process_group(
        backend=config["distributed_backend"],
        init_method="file://" + os.path.abspath(config["distributed_init_file"]),
        timeout=datetime.timedelta(seconds=120),
        world_size=config["n_workers"],
        rank=config["rank"],
    )

    if torch.distributed.get_rank() == 0:
        if config["task"] == "Cifar":
            download_cifar()
        elif config["task"] == "LSTM":
            download_wikitext2()
    torch.distributed.barrier()
    torch.cuda.synchronize()

    task = tasks.build(task_name=config["task"], device=device, timer=timer, **config)

    task._model = torch.nn.parallel.DistributedDataParallel(
        task._model, process_group=process_group
    )

    if config["use_powersgd"]:
        state = powerSGD.PowerSGDState(
            process_group=process_group,
            matrix_approximation_rank=config["optimizer_reducer_rank"],
            start_powerSGD_iter=config["start_powerSGD_iter"],
        )

        def hook(
            state: powerSGD.PowerSGDState, bucket: dist._GradBucket
        ) -> torch.futures.Future:
            start = time.time_ns() / 1_000_000_000

            def stop_the_time(fut):
                torch.cuda.synchronize()
                end = time.time_ns() / 1_000_000_000
                timer.report("batch.reduce", start, end)
                return fut.value()

            ret_val = powerSGD.powerSGD_hook(state, bucket)
            return ret_val.then(stop_the_time)

        task._model.register_comm_hook(state, hook)
    elif config["fp16_compression"]:

        def hook(
            process_group: dist.ProcessGroup, bucket: dist._GradBucket
        ) -> torch.futures.Future:
            start = time.time_ns() / 1_000_000_000

            def stop_the_time(fut):
                torch.cuda.synchronize()
                end = time.time_ns() / 1_000_000_000
                timer.report("batch.reduce", start, end)
                return fut.value()

            ret_val = default_hooks.fp16_compress_hook(process_group, bucket)
            return ret_val.then(stop_the_time)

        task._model.register_comm_hook(process_group, hook)
    else:

        def hook(
            process_group: dist.ProcessGroup, bucket: dist._GradBucket
        ) -> torch.futures.Future:
            start = time.time_ns() / 1_000_000_000

            def stop_the_time(fut):
                torch.cuda.synchronize()
                end = time.time_ns() / 1_000_000_000
                timer.report("batch.reduce", start, end)
                return fut.value()

            ret_val = default_hooks.allreduce_hook(process_group, bucket)
            return ret_val.then(stop_the_time)

        task._model.register_comm_hook(process_group, hook)

    communication = {"bits": 0}

    # Override dist.all_reduce so we can keep track of the amount communicated
    all_reduce_orig = dist.all_reduce

    def all_reduce_with_logging(*args, **kwargs):
        tensor = args[0]
        start = time.time_ns() / 1_000_000_000

        def stop_the_time(fut):
            torch.cuda.synchronize()
            end = time.time_ns() / 1_000_000_000
            timer.report("all_reduce", start, end)
            return fut.value()

        communication["bits"] += 8 * tensor.nelement() * tensor.element_size()
        ret = all_reduce_orig(*args, **kwargs)
        ret.get_future().then(stop_the_time)
        return ret

    dist.all_reduce = all_reduce_with_logging

    optimizer = torch.optim.SGD(
        [
            {
                "params": [param],
                "lr": get_learning_rate(0, param_name),
                "weight_decay": get_weight_decay(0, param_name),
            }
            for param_name, param in task._model.named_parameters()
        ],
        lr=config["optimizer_learning_rate"],
        momentum=config["optimizer_momentum"],
        nesterov=(config["optimizer_momentum_type"] == "nesterov"),
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda epoch: get_learning_rate(epoch, "dummy") / get_learning_rate(0, "dummy"),
    )

    for epoch in range(config["num_epochs"]):
        epoch_metrics = MeanAccumulator()
        info(
            {
                "state.progress": float(epoch) / config["num_epochs"],
                "state.current_epoch": epoch,
            }
        )

        train_loader = task.train_iterator(config["optimizer_batch_size"])
        for i, batch in enumerate(train_loader):
            epoch_frac = epoch + i / len(train_loader)

            with timer("batch", epoch_frac):
                _, _, metrics = task.batch_loss_and_gradient(batch)
                epoch_metrics.add(metrics)

                optimizer.step()

        scheduler.step()

        with timer("epoch_metrics.collect", epoch + 1.0, verbosity=2):
            epoch_metrics.reduce()
            for key, value in epoch_metrics.value().items():
                metric(
                    key,
                    {
                        "value": value.item(),
                        "epoch": epoch + 1.0,
                        "bits": communication["bits"],
                    },
                    tags={"split": "train"},
                )
                metric(
                    f"last_{key}",
                    {
                        "value": value.item(),
                        "epoch": epoch + 1.0,
                        "bits": communication["bits"],
                    },
                    tags={"split": "train"},
                )

        with timer("test.last", epoch):
            test_stats = task.test()
            for key, value in test_stats.items():
                metric(
                    f"last_{key}",
                    {
                        "value": value.item(),
                        "epoch": epoch + 1.0,
                        "bits": communication["bits"],
                    },
                    tags={"split": "test"},
                )

        print(timer.summary())
        if config["rank"] == 0:
            timer.save_summary(os.path.join(output_dir, "timer_summary.json"))

    info({"state.progress": 1.0})


def save(destination_path, model_state, epoch, test_stats):
    """Save a checkpoint to disk"""
    # Workaround for RuntimeError('Unknown Error -1')
    # https://github.com/pytorch/pytorch/issues/10577
    time.sleep(1)

    torch.save(
        {"epoch": epoch, "test_stats": test_stats, "model_state_dict": model_state},
        destination_path,
    )


def get_weight_decay(epoch, parameter_name):
    """Take care of differences between weight decay for parameters"""
    if is_conv_param(parameter_name):
        return config["optimizer_weight_decay_conv"]
    elif is_batchnorm_param(parameter_name):
        return config["optimizer_weight_decay_bn"]
    else:
        return config["optimizer_weight_decay_other"]


def get_learning_rate(epoch, parameter_name):
    """Apply any learning rate schedule"""
    if is_conv_param(parameter_name):
        lr = config["optimizer_conv_learning_rate"]
    else:
        lr = config["optimizer_learning_rate"]

    if config["optimizer_scale_lr_with_warmup_epochs"]:
        warmup_epochs = config["optimizer_scale_lr_with_warmup_epochs"]
        max_factor = config.get("optimizer_scale_lr_with_factor", None)
        if max_factor is None:
            max_factor = (
                torch.distributed.get_world_size()
                if torch.distributed.is_available()
                else 1.0
            )
        factor = 1.0 + (max_factor - 1.0) * min(epoch / warmup_epochs, 1.0)
        lr *= factor

    for decay_epoch in config["optimizer_decay_at_epochs"]:
        if epoch >= decay_epoch:
            lr /= config["optimizer_decay_with_factor"]
        else:
            return lr
    return lr


def is_conv_param(parameter_name):
    """
    Says whether this parameter is a conv linear layer that
    needs a different treatment from the other weights
    """
    return "conv" in parameter_name and "weight" in parameter_name


def download_cifar(data_root=os.path.join(os.getenv("DATA"), "data")):
    import torchvision

    dataset = torchvision.datasets.CIFAR10
    training_set = dataset(root=data_root, train=True, download=True)
    test_set = dataset(root=data_root, train=False, download=True)


def download_wikitext2(data_root=os.path.join(os.getenv("DATA"), "data")):
    import torchtext

    torchtext.datasets.WikiText2.splits(
        torchtext.data.Field(lower=True), root=os.path.join(data_root, "wikitext2")
    )


def is_batchnorm_param(parameter_name):
    """
    Is this parameter part of a batchnorm parameter?
    """
    return re.match(r""".*\.bn\d+\.(weight|bias)""", parameter_name)


def replace_grad_by_momentum(grad, momentum):
    """
    Inplace operation that applies momentum to a gradient.
    This distinguishes between types of momentum (heavy-ball vs nesterov)
    """
    if config["optimizer_momentum_type"] == "heavy-ball":
        grad.data[:] = momentum
    if config["optimizer_momentum_type"] == "exponential_moving_average":
        grad.data[:] = momentum
    elif config["optimizer_momentum_type"] == "nesterov":
        grad.data[:] += momentum
    else:
        raise ValueError("Unknown momentum type")


def get_reducer(device, timer):
    """Configure the reducer from the config"""
    if config["optimizer_reducer"] in ["RankKReducer"]:
        return getattr(gradient_reducers, config["optimizer_reducer"])(
            random_seed=config["seed"],
            device=device,
            timer=timer,
            n_power_iterations=config["optimizer_reducer_n_power_iterations"],
            reuse_query=config["optimizer_reducer_reuse_query"],
            rank=config["optimizer_reducer_rank"],
        )
    elif config["optimizer_reducer"] == "AtomoReducer":
        return getattr(gradient_reducers, config["optimizer_reducer"])(
            random_seed=config["seed"],
            device=device,
            timer=timer,
            rank=config["optimizer_reducer_rank"],
        )
    elif config["optimizer_reducer"] == "RandomSparseReducer":
        return getattr(gradient_reducers, config["optimizer_reducer"])(
            random_seed=config["seed"],
            device=device,
            timer=timer,
            rank=config["optimizer_reducer_rank"],
        )
    elif config["optimizer_reducer"] == "RandomSparseBlockReducer":
        return getattr(gradient_reducers, config["optimizer_reducer"])(
            random_seed=config["seed"],
            device=device,
            timer=timer,
            rank=config["optimizer_reducer_rank"],
        )
    elif (
        config["optimizer_reducer"] == "GlobalTopKReducer"
        or config["optimizer_reducer"] == "TopKReducer"
        or config["optimizer_reducer"] == "UniformRandomSparseBlockReducer"
        or config["optimizer_reducer"] == "UniformRandomSparseReducer"
    ):
        return getattr(gradient_reducers, config["optimizer_reducer"])(
            random_seed=config["seed"],
            device=device,
            timer=timer,
            compression=config["optimizer_reducer_compression"],
        )
    elif config["optimizer_reducer"] == "HalfRankKReducer":
        return getattr(gradient_reducers, config["optimizer_reducer"])(
            random_seed=config["seed"],
            device=device,
            timer=timer,
            rank=config["optimizer_reducer_rank"],
        )
    elif config["optimizer_reducer"] == "SVDReducer":
        return getattr(gradient_reducers, config["optimizer_reducer"])(
            config["seed"], device, timer, config["optimizer_reducer_rank"]
        )
    else:
        return getattr(gradient_reducers, config["optimizer_reducer"])(
            config["seed"], device, timer
        )


@torch.jit.script
def l2norm(tensor):
    """Compute the L2 Norm of a tensor in a fast and correct way"""
    # tensor.norm(p=2) is buggy in Torch 1.0.0
    # tensor.norm(p=2) is really slow in Torch 1.0.1
    return torch.sqrt(torch.sum(tensor**2))


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


def check_model_consistency_across_workers(model, epoch):
    signature = []
    for name, param in model.named_parameters():
        signature.append(param.view(-1)[0].item())

    rank = config["rank"]
    signature = ",".join(f"{x:.4f}" for x in signature)
    print(f"Model signature for epoch {epoch:04d} / worker {rank:03d}:\n{signature}")


if __name__ == "__main__":
    main()
