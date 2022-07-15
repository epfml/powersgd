from types import SimpleNamespace
from typing import Iterator, List, NamedTuple, Tuple

import torch


def pack(tensors: List[torch.Tensor]) -> Tuple[torch.Tensor, List[torch.Size]]:
    """Packs a list of tensors into one buffer for sending to other workers"""
    buffer = torch.cat([t.view(-1) for t in tensors])  # copies
    shapes = [tensor.shape for tensor in tensors]
    return buffer, shapes


def unpack(buffer: torch.Tensor, shapes: List[torch.Size]) -> List[torch.Tensor]:
    """Provides pointers to tensors of original `shapes` in a flat-packed buffer."""
    idx = 0
    entries = []
    for tensor_shape in shapes:
        end = idx + tensor_shape.numel()
        entries.append(buffer[idx:end].view(size=tensor_shape))
        idx = end

    return entries


def batch_unpack(
    batch_of_buffers: torch.Tensor, shapes: List[torch.Size]
) -> List[torch.Tensor]:
    """Same as unpack, but given a batch of buffers of the same type, it produces batches of tensors"""
    idx = 0
    entries = []
    batch_size = len(batch_of_buffers)
    for tensor_shape in shapes:
        end = idx + tensor_shape.numel()
        entries.append(batch_of_buffers[:, idx:end].view(batch_size, *tensor_shape))
        idx = end

    return entries


class ContiguousAllocation(NamedTuple):
    buffer: torch.Tensor
    shapes: list[torch.Size]
    tensors: list[torch.Tensor]


def allocate_contiguous(
    shapes: list[torch.Size], device: torch.device, dtype: torch.dtype
) -> ContiguousAllocation:
    numel = sum(s.numel() for s in shapes)
    buffer = torch.empty(numel, dtype=dtype, device=device)
    return ContiguousAllocation(buffer, shapes, unpack(buffer, shapes))


def params_in_optimizer(optimizer: torch.optim.Optimizer) -> List[torch.Tensor]:
    params = []
    for group in optimizer.param_groups:
        params.extend(group["params"])
    return params


def is_distributed() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()  # type: ignore


def flatten(tensors: List[List[torch.Tensor]]) -> List[torch.Tensor]:
    out = []
    for list in tensors:
        out.extend(list)
    return out


def allreduce_average(data, *args, **kwargs):
    """All-reduce average if torch.distributed is available, otherwise do nothing"""
    if is_distributed():
        data.div_(torch.distributed.get_world_size())  # type: ignore
        return torch.distributed.all_reduce(data, *args, **kwargs)  # type: ignore
    else:
        return SimpleNamespace(wait=lambda: None)
