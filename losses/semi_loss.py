import torch
from torch import Tensor
from torch.nn import functional as F

from ._base import LossClass
from .discreteMI import IIDSegmentationLoss


class MI(LossClass[Tensor]):

    def __init__(self, **kwargs):
        super().__init__()
        self._criterion = IIDSegmentationLoss(padding=0)

    def __call__(self, unlabeled_pred_simplex: Tensor) -> Tensor:
        loss = self._criterion(unlabeled_pred_simplex, unlabeled_pred_simplex)
        return loss


def pairwise_matrix(vec1: Tensor, vec2: Tensor):
    assert vec1.shape == vec2.shape
    assert vec1.dim() == 2
    return vec1 @ vec2.t()


def normalize(vec: Tensor, dim: int = 1):
    return F.normalize(vec, p=2, dim=dim)


class Orth(LossClass[Tensor]):
    def __init__(self, prototypes, **kwargs):
        self._prototypes = prototypes

    def __call__(self, *args, **kwargs):
        normalized_prototypes = normalize(self._prototypes)
        matrix = pairwise_matrix(normalized_prototypes.squeeze(), normalized_prototypes.squeeze())
        loss = (matrix - torch.eye(matrix.shape[0], device=matrix.device, dtype=matrix.dtype)).pow(2).mean()
        return loss
