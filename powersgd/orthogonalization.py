import torch


def orthogonalize(matrix: torch.Tensor):
    if matrix.shape[-1] == 1:
        matrix[:] /= matrix.norm()
    else:
        matrix[:] = torch.linalg.qr(matrix).Q
