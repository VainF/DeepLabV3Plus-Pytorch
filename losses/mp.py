# this file implements the multicore loss that lets a class having multiple prototype representation
import typing as t

import torch
from loguru import logger
from torch import nn, Tensor

from ._base import LossClass
from .kl import KL_div2, KL_div_with_ignore_index


class MultiCoreKL(nn.Module, LossClass[Tensor]):
    def __init__(self, groups: t.List[t.List[int]]):
        super().__init__()
        self._groups = groups
        self.kl = KL_div2()
        logger.trace(f"{self.__class__.__name__} created with groups: {groups}")

    def forward(self, predict_simplex: Tensor, onehot_target: Tensor):
        reduced_simplex = self.reduced_simplex(predict_simplex)
        loss = self.kl(reduced_simplex, onehot_target)
        return loss

    @property
    def groups(self) -> t.List[t.List[int]]:
        return self._groups

    def reduced_simplex(self, predict_simplex: Tensor):
        reduced_simplex = torch.cat([predict_simplex[:, i].sum(1, keepdim=True) for i in self._groups], dim=1)
        return reduced_simplex


class MultiCoreKLwithIgnoreIndex(MultiCoreKL):

    def __init__(self, groups: t.List[t.List[int]], ignore_index):
        super().__init__(groups)
        self.kl = KL_div_with_ignore_index(ignore_index=ignore_index)

    def forward(self, predict_simplex: Tensor, target_class: Tensor):
        reduced_simplex = self.reduced_simplex(predict_simplex)
        loss = self.kl(reduced_simplex, target_class)
        return loss
