#!/usr/bin/env python3

import datetime
import os
import re
import time

import numpy as np
import torch

import gradient_reducers
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
    average_reset_epoch_interval=30,
    distributed_backend="nccl",
    fix_conv_weight_norm=False,
    num_epochs=300,
    checkpoints=[],
    num_train_tracking_batches=1,
    optimizer_batch_size=128,  # per worker
    optimizer_conv_learning_rate=0.1,  # tuned for batch size 128
    optimizer_decay_at_epochs=[150, 250],
    optimizer_decay_with_factor=10.0,
    optimizer_learning_rate=0.1,  # Tuned for batch size 128 (single worker)
    optimizer_memory=True,
    optimizer_momentum_type="nesterov",
    optimizer_momentum=0.9,
    optimizer_reducer="ExactReducer",
    # optimizer_reducer_compression=0.01,
    # optimizer_reducer_rank=4,
    optimizer_reducer_reuse_query=True,
    # optimizer_reducer_n_power_iterations=0,
    optimizer_scale_lr_with_factor=None,  # set to override world_size as a factor
    optimizer_scale_lr_with_warmup_epochs=5,  # scale lr by world size
    optimizer_mom_before_reduce=False,
    optimizer_wd_before_reduce=False,
    optimizer_weight_decay_conv=0.0001,
    optimizer_weight_decay_other=0.0001,
    optimizer_weight_decay_bn=0.0,
    task="Cifar",
    task_architecture="ResNet18",
    seed=42,
    rank=0,
    n_workers=1,
    distributed_init_file=None,
    log_verbosity=2,
)

output_dir = "./output.tmp"  # will be overwritten by run.py


def main():
    torch.manual_seed(config["seed"] + config["rank"])
    np.random.seed(config["seed"] + config["rank"])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    timer = Timer(verbosity_level=config["log_verbosity"], log_fn=metric)

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
            timeout=datetime.timedelta(seconds=120),
            world_size=config["n_workers"],
            rank=config["rank"],
        )

    task = tasks.build(task_name=config["task"], device=device, timer=timer, **config)
    reducer = get_reducer(device, timer)

    bits_communicated = 0
    runavg_model = MeanAccumulator()

    memories = [torch.zeros_like(param) for param in task.state]
    momenta = [torch.empty_like(param) for param in task.state]  # need initialization
    send_buffers = [torch.zeros_like(param) for param in task.state]
    for epoch in range(config["num_epochs"]):
        epoch_metrics = MeanAccumulator()
        info({"state.progress": float(epoch) / config["num_epochs"], "state.current_epoch": epoch})

        # This seems fine ...
        # check_model_consistency_across_workers(task._model, epoch)

        # Determine per-parameter optimization parameters
        wds = [get_weight_decay(epoch, name) for name in task.parameter_names]

        # Reset running average of the model
        if epoch % config["average_reset_epoch_interval"] == 0:
            runavg_model.reset()

        train_loader = task.train_iterator(config["optimizer_batch_size"])
        for i, batch in enumerate(train_loader):
            epoch_frac = epoch + i / len(train_loader)
            lrs = [get_learning_rate(epoch_frac, name) for name in task.parameter_names]

            with timer("batch", epoch_frac):
                _, grads, metrics = task.batch_loss_and_gradient(batch)
                epoch_metrics.add(metrics)

                # Compute some derived metrics from the raw gradients
                with timer("batch.reporting.lr", epoch_frac, verbosity=2):
                    for name, param, grad, lr in zip(task.parameter_names, task.state, grads, lrs):
                        if np.random.rand() < 0.001:  # with a small probability
                            tags = {"weight": name.replace("module.", "")}
                            metric(
                                "effective_lr",
                                {
                                    "epoch": epoch_frac,
                                    "value": lr / max(l2norm(param).item() ** 2, 1e-8),
                                },
                                tags,
                            )
                            metric(
                                "grad_norm",
                                {"epoch": epoch_frac, "value": l2norm(grad).item()},
                                tags,
                            )

                if config["optimizer_wd_before_reduce"]:
                    with timer("batch.weight_decay", epoch_frac, verbosity=2):
                        for grad, param, wd in zip(grads, task.state, wds):
                            if wd > 0:
                                grad.add_(param.detach(), alpha=wd)

                if config["optimizer_mom_before_reduce"]:
                    with timer("batch.momentum", epoch_frac, verbosity=2):
                        for grad, momentum in zip(grads, momenta):
                            if epoch == 0 and i == 0:
                                momentum.data = grad.clone().detach()
                            else:
                                if (
                                    config["optimizer_momentum_type"]
                                    == "exponential_moving_average"
                                ):
                                    momentum.mul_(config["optimizer_momentum"]).add_(
                                        grad, alpha=1 - config["optimizer_momentum"]
                                    )
                                else:
                                    momentum.mul_(config["optimizer_momentum"]).add_(grad)
                            replace_grad_by_momentum(grad, momentum)

                with timer("batch.accumulate", epoch_frac, verbosity=2):
                    for grad, memory, send_bfr in zip(grads, memories, send_buffers):
                        if config["optimizer_memory"]:
                            send_bfr.data[:] = grad + memory
                        else:
                            send_bfr.data[:] = grad

                with timer("batch.reduce", epoch_frac):
                    # Set 'grads' to the averaged value from the workers
                    bits_communicated += reducer.reduce(send_buffers, grads, memories)

                if config["optimizer_memory"]:
                    with timer("batch.reporting.compr_err", verbosity=2):
                        for name, memory, send_bfr in zip(
                            task.parameter_names, memories, send_buffers
                        ):
                            if np.random.rand() < 0.001:
                                tags = {"weight": name.replace("module.", "")}
                                rel_compression_error = l2norm(memory) / l2norm(send_bfr)
                                metric(
                                    "rel_compression_error",
                                    {"epoch": epoch_frac, "value": rel_compression_error.item()},
                                    tags,
                                )

                if not config["optimizer_wd_before_reduce"]:
                    with timer("batch.wd", epoch_frac, verbosity=2):
                        for grad, param, wd in zip(grads, task.state, wds):
                            if wd > 0:
                                grad.add_(param.detach(), alpha=wd)

                if not config["optimizer_mom_before_reduce"]:
                    with timer("batch.mom", epoch_frac, verbosity=2):
                        for grad, momentum in zip(grads, momenta):
                            if epoch == 0 and i == 0:
                                momentum.data = grad.clone().detach()
                            else:
                                if (
                                    config["optimizer_momentum_type"]
                                    == "exponential_moving_average"
                                ):
                                    momentum.mul_(config["optimizer_momentum"]).add_(
                                        grad, alpha=1 - config["optimizer_momentum"]
                                    )
                                else:
                                    momentum.mul_(config["optimizer_momentum"]).add_(grad)
                            replace_grad_by_momentum(grad, momentum)

                with timer("batch.step", epoch_frac, verbosity=2):
                    for param, grad, lr in zip(task.state, grads, lrs):
                        param.data.add_(grad, alpha=-lr)

                if config["fix_conv_weight_norm"]:
                    with timer("batch.normfix", epoch_frac, verbosity=2):
                        for param_name, param in zip(task.parameter_names, task.state):
                            if is_conv_param(param_name):
                                param.data[:] /= l2norm(param)

                with timer("batch.update_runavg", epoch_frac, verbosity=2):
                    runavg_model.add(task.state_dict())

                if config["optimizer_memory"]:
                    with timer("batch.reporting.memory_norm", epoch_frac, verbosity=2):
                        if np.random.rand() < 0.001:
                            sum_of_sq = 0.0
                            for parameter_name, memory in zip(task.parameter_names, memories):
                                tags = {"weight": parameter_name.replace("module.", "")}
                                sq_norm = torch.sum(memory ** 2)
                                sum_of_sq += torch.sqrt(sq_norm)
                                metric(
                                    "memory_norm",
                                    {"epoch": epoch_frac, "value": torch.sqrt(sq_norm).item()},
                                    tags,
                                )
                            metric(
                                "compression_error",
                                {"epoch": epoch_frac, "value": torch.sqrt(sum_of_sq).item()},
                            )

        with timer("epoch_metrics.collect", epoch + 1.0, verbosity=2):
            epoch_metrics.reduce()
            for key, value in epoch_metrics.value().items():
                metric(
                    key,
                    {"value": value.item(), "epoch": epoch + 1.0, "bits": bits_communicated},
                    tags={"split": "train"},
                )
                metric(
                    f"last_{key}",
                    {"value": value.item(), "epoch": epoch + 1.0, "bits": bits_communicated},
                    tags={"split": "train"},
                )

        with timer("test.last", epoch):
            test_stats = task.test()
            for key, value in test_stats.items():
                metric(
                    f"last_{key}",
                    {"value": value.item(), "epoch": epoch + 1.0, "bits": bits_communicated},
                    tags={"split": "test"},
                )

        with timer("test.runavg", epoch):
            test_stats = task.test(state_dict=runavg_model.value())
            for key, value in test_stats.items():
                metric(
                    f"runavg_{key}",
                    {"value": value.item(), "epoch": epoch + 1.0, "bits": bits_communicated},
                    tags={"split": "test"},
                )

        if epoch in config["checkpoints"] and torch.distributed.get_rank() == 0:
            with timer("checkpointing"):
                save(
                    os.path.join(output_dir, "epoch_{:03d}".format(epoch)),
                    task.state_dict(),
                    epoch + 1.0,
                    test_stats,
                )
                # Save running average model @TODO

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
                torch.distributed.get_world_size() if torch.distributed.is_available() else 1.0
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
    return torch.sqrt(torch.sum(tensor ** 2))


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
