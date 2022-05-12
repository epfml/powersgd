import torch


def orthogonalize(matrix: torch.Tensor, eps=1e-16):
    if matrix.shape[-1] == 1:
        matrix[:] /= max(matrix.norm(), eps)
    else:
        matrix[:] = torch.linalg.qr(matrix).Q
