import torch


def orthogonalize(matrix: torch.Tensor):
    matrix[:] = torch.linalg.qr(matrix).Q
