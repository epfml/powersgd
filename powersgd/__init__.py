import torch
from .powersgd import AdaptivePowerSGD, Aggregator, Config
from .utils import params_in_optimizer


PowerSGD = AdaptivePowerSGD


def optimizer_step(optimizer: torch.optim.Optimizer, aggregator: Aggregator):
    params = params_in_optimizer(optimizer)
    grads = [p.grad.data for p in params]  # type: ignore
    avg_grads = aggregator.aggregate(grads)  # subtracts the approximation from grads

    # Temporarily set parameter's gradients to the aggregated values
    for (p, g) in zip(params, avg_grads):
        p.grad = g

    # Run an optimizer step
    optimizer.step()

    # Put back the error buffer as the parameter's gradient
    for (p, g) in zip(params, grads):
        p.grad = g
