import math
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist


def generate_random_orthogonal_matrix(feat_in, num_classes):
    rand_mat = np.random.random(size=(feat_in, num_classes))
    orth_vec, _ = np.linalg.qr(rand_mat)
    orth_vec = torch.tensor(orth_vec).float()
    assert torch.allclose(torch.matmul(orth_vec.T, orth_vec), torch.eye(num_classes), atol=1.e-7), \
        "The max irregular value is : {}".format(
            torch.max(torch.abs(torch.matmul(orth_vec.T, orth_vec) - torch.eye(num_classes))))
    return orth_vec


class ETFHead():
    def __init__(self, num_classes : int, feat_in : int) -> None:
        orth_vec = generate_random_orthogonal_matrix(feat_in, num_classes)
        i_nc_nc = torch.eye(num_classes)
        one_nc_nc: torch.Tensor = torch.mul(torch.ones(
            num_classes, num_classes), (1 / num_classes))
        etf_vec = torch.mul(torch.matmul(orth_vec, i_nc_nc - one_nc_nc),
                            math.sqrt(num_classes / (num_classes - 1)))
