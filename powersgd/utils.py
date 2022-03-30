import torch
from types import SimpleNamespace


def pack(tensors):
    """Packs a list of tensors into one buffer for sending to other workers"""
    buffer = torch.cat([t.view(-1) for t in tensors])  # copies
    shapes = [tensor.shape for tensor in tensors]
    return buffer, shapes


def unpack(buffer, shapes):
    """Provides pointers to tensors of original `shapes` in a flat-packed buffer."""
    idx = 0
    entries = []
    for tensor_shape in shapes:
        end = idx + tensor_shape.numel()
        entries.append(buffer[idx:end].view(size=tensor_shape))
        idx = end

    return entries


def num_bits(tensor):
    return tensor.nelement() * 8 * tensor.element_size()


def params_in_optimizer(optimizer: torch.optim.Optimizer) -> list[torch.Tensor]:
    params = []
    for group in optimizer.param_groups:
        params.extend(group["params"])
    return params


def is_distributed():
    return torch.distributed.is_available() and torch.distributed.is_initialized()  # type: ignore


def flatten(tensors: list[list[torch.Tensor]]) -> list[torch.Tensor]:
    out = []
    for list in tensors:
        out.extend(list)
    return out


def allreduce_average(data, *args, **kwargs):
    """All-reduce average if torch.distributed is available, otherwise do nothing"""
    if is_distributed():
        data /= torch.distributed.world_size()  # type: ignore
        return torch.distributed.all_reduce(data, *args, **kwargs)  # type: ignore
    else:
        return SimpleNamespace(wait=lambda: None)
