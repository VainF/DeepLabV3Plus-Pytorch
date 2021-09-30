from typing import Optional, OrderedDict
from typing import TypeVar, List, Dict, Union

import numpy as np
import torch
from loguru import logger
from torch import Tensor
from torch import nn
from torch.nn import functional as F

from ._base import LossClass

__all__ = ["Entropy", "KL_div"]

A = TypeVar("A")
B = TypeVar("B")
T = TypeVar("T", Tensor, np.ndarray)


def _check_reduction_params(reduction):
    assert reduction in (
        "mean",
        "sum",
        "none",
    ), "reduction should be in ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``, given {}".format(
        reduction
    )


class Entropy(nn.Module, LossClass[Tensor]):
    r"""General Entropy interface

    the definition of Entropy is - \sum p(xi) log (p(xi))

    reduction (string, optional): Specifies the reduction to apply to the output:
    ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
    ``'mean'``: the sum of the output will be divided by the number of
    elements in the output, ``'sum'``: the output will be summed.
    """

    def __init__(self, reduction="mean", eps=1e-16):
        super().__init__()
        _check_reduction_params(reduction)
        self._eps = eps
        self._reduction = reduction

    def forward(self, input_: Tensor) -> Tensor:
        assert input_.shape.__len__() >= 2
        b, _, *s = input_.shape
        e = input_ * (input_ + self._eps).log()
        e = -1.0 * e.sum(1)
        assert e.shape == torch.Size([b, *s])
        if self._reduction == "mean":
            return e.mean()
        elif self._reduction == "sum":
            return e.sum()
        else:
            return e


class KL_div(nn.Module, LossClass[Tensor]):
    """
    KL(p,q)= -\sum p(x) * log(q(x)/p(x))
    where p, q are distributions
    p is usually the fixed one like one hot coding
    p is the target and q is the distribution to get approached.

    reduction (string, optional): Specifies the reduction to apply to the output:
    ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
    ``'mean'``: the sum of the output will be divided by the number of
    elements in the output, ``'sum'``: the output will be summed.
    """

    def __init__(self, reduction="mean", eps=1e-16, weight: Union[List[float], Tensor] = None):
        super().__init__()
        _check_reduction_params(reduction)
        self._eps = eps
        self._reduction = reduction
        self._weight: Optional[Tensor] = weight
        if weight is not None:
            assert isinstance(weight, (list, Tensor)), type(weight)
            if isinstance(weight, list):
                self._weight = torch.Tensor(weight).float()
            else:
                self._weight = weight.float()
            # normalize weight:
            self._weight = self._weight / self._weight.sum() * len(self._weight)
        logger.trace(
            f"Initialized {self.__class__.__name__} with weight={self._weight} and reduction={self._reduction}.")

    def forward(self, prob: Tensor, target: Tensor, **kwargs) -> Tensor:
        b, c, *hwd = target.shape
        kl = (-target * torch.log((prob + self._eps) / (target + self._eps)))
        if self._weight is not None:
            assert len(self._weight) == c
            weight = self._weight.expand(b, *hwd, -1).transpose(-1, 1).detach()
            kl *= weight.to(kl.device)
        kl = kl.sum(1)
        if self._reduction == "mean":
            return kl.mean()
        elif self._reduction == "sum":
            return kl.sum()
        else:
            return kl

    def __repr__(self):
        return f"{self.__class__.__name__}\n, weight={self._weight}"

    def state_dict(self, *args, **kwargs):
        save_dict = super().state_dict(*args, **kwargs)
        # save_dict["weight"] = self._weight
        # save_dict["reduction"] = self._reduction
        return save_dict

    def load_state_dict(self, state_dict: Union[Dict[str, Tensor], OrderedDict[str, Tensor]], *args, **kwargs):
        super(KL_div, self).load_state_dict(state_dict, **kwargs)
        # self._reduction = state_dict["reduction"]
        # self._weight = state_dict["weight"]


class KL_div2(nn.Module, LossClass[Tensor]):
    """
    KL(p,q)= -\sum p(x) * log(q(x)/p(x))
    where p, q are distributions
    p is usually the fixed one like one hot coding
    p is the target and q is the distribution to get approached.

    reduction (string, optional): Specifies the reduction to apply to the output:
    ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
    ``'mean'``: the sum of the output will be divided by the number of
    elements in the output, ``'sum'``: the output will be summed.
    """

    def __init__(self, reduction="mean", eps=1e-16):
        super().__init__()
        _check_reduction_params(reduction)
        self._eps = eps
        self._reduction = reduction
        self._logsoftmax = nn.LogSoftmax(dim=1)

        logger.trace(
            f"Initialized {self.__class__.__name__}  reduction={self._reduction}.")

    def forward(self, prob: Tensor, target: Tensor, **kwargs) -> Tensor:
        assert prob.shape == target.shape
        b, c, *hwd = target.shape
        kl = torch.sum(- target * self._logsoftmax(prob), 1)
        if self._reduction == "mean":
            return kl.mean()
        elif self._reduction == "sum":
            return kl.sum()
        else:
            return kl

    def __repr__(self):
        return f"{self.__class__.__name__}\n, weight={self._weight}"


class KL_div_with_ignore_index(KL_div2):
    def __init__(self, reduction="mean", eps=1e-16, ignore_index: int = None):
        super().__init__(reduction, eps, )
        self._ignore_index = ignore_index

    def forward(self, prob: Tensor, target: Tensor, **kwargs) -> Tensor:
        """

        Args:
            prob: prediction distribution where the sum reaches 1.0
            target: target classes, not one hot with ignore index.
            **kwargs: to be ignored.

        Returns: loss

        """
        mask = target != self._ignore_index
        pred_simplex_masked = prob.moveaxis(1, 3)[mask]
        target_masked = target[mask]
        _, C, *_ = prob.shape
        target_masked_one_hot = F.one_hot(target_masked, C, )

        return super(KL_div_with_ignore_index, self).forward(pred_simplex_masked, target_masked_one_hot)
