import torch

from powersgd.powersgd import Aggregator, AllReduce, Config, PowerSGD
from powersgd.utils import params_in_optimizer


def optimizer_step(optimizer: torch.optim.Optimizer, aggregator: Aggregator):
    """
    Aggregate gradients across workers using `aggregator`,
    and then take an optimizer step using the aggregated gradient.
    """
    params = params_in_optimizer(optimizer)
    avg_grads = aggregator.aggregate()

    # Temporarily set parameter's gradients to the aggregated values
    orig_grads = [p.grad.data for p in params]
    for (p, g) in zip(params, avg_grads):
        p.grad = g

    # Run an optimizer step
    optimizer.step()

    # Put back the error buffer as the parameter's gradient
    for (p, g) in zip(params, orig_grads):
        p.grad = g
