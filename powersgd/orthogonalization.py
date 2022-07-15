import torch


def orthogonalize(matrix: torch.Tensor, eps=torch.tensor(1e-16)):
    if matrix.shape[-1] == 1:
        matrix.div_(torch.maximum(matrix.norm(), eps))
    else:
        matrix.copy_(torch.linalg.qr(matrix).Q)
