import sys
import typing as t
from itertools import repeat

import numpy as np
import torch
from loguru import logger
from termcolor import colored
from torch import Tensor
from torch import nn

try:
    from torch._six import container_abcs
except ImportError:
    import collections.abc as container_abcs
from torch.nn import functional as F

from ._base import LossClass


def average_iter(a_list):
    return sum(a_list) / float(len(a_list))


def ntuple(n):
    def parse(x):
        if isinstance(x, str):
            return tuple(repeat(x, n))
        if isinstance(x, container_abcs.Iterable):
            x = list(x)
            if len(x) == 1:
                return tuple(repeat(x[0], n))
            else:
                if len(x) != n:
                    raise RuntimeError(f"inconsistent shape between {x} and {n}")
            return x

        return tuple(repeat(x, n))

    return parse


_single = ntuple(1)
_pair = ntuple(2)


def simplex(t: Tensor, axis=1) -> bool:
    """
    check if the matrix is the probability distribution
    :param t:
    :param axis:
    :return:
    """
    _sum = t.sum(axis).type(torch.float32)
    _ones = torch.ones_like(_sum, dtype=torch.float32)
    return torch.allclose(_sum, _ones, rtol=1e-4, atol=1e-4)


class IIDLoss(nn.Module, LossClass[t.Tuple[Tensor, Tensor, Tensor]]):
    def __init__(self, lamb: float = 1.0, eps: float = sys.float_info.epsilon):
        """
        :param lamb:
        :param eps:
        """
        super().__init__()
        logger.trace(colored(f"Initialize {self.__class__.__name__}.", "green"))
        self.lamb = float(lamb)
        self.eps = float(eps)

    def forward(self, x_out: Tensor, x_tf_out: Tensor):
        """
        return the inverse of the MI. if the x_out == y_out, return the inverse of Entropy
        :param x_out:
        :param x_tf_out:
        :return:
        """
        assert len(x_out.shape) == 2, x_out.shape
        assert simplex(x_out), f"x_out not normalized."
        assert simplex(x_tf_out), f"x_tf_out not normalized."
        _, k = x_out.size()
        p_i_j = compute_joint(x_out, x_tf_out)
        assert p_i_j.size() == (k, k)

        p_i = (p_i_j.sum(dim=1).view(k, 1).expand(k, k))  # p_i should be the mean of the x_out
        p_j = p_i_j.sum(dim=0).view(1, k).expand(k, k)  # but should be same, symmetric

        loss = -p_i_j * (
            torch.log(p_i_j + 1e-10) - self.lamb * torch.log(p_j + 1e-10) - self.lamb * torch.log(p_i + 1e-10)
        )
        loss = loss.sum()
        loss_no_lamb = -p_i_j * (torch.log(p_i_j + 1e-10) - torch.log(p_j + 1e-10) - torch.log(p_i + 1e-10))
        loss_no_lamb = loss_no_lamb.sum()
        return loss, loss_no_lamb, p_i_j


class IIDSegmentationLoss(nn.Module, LossClass[Tensor]):
    def __init__(
        self, lamda=1.0, padding=7, eps: float = sys.float_info.epsilon
    ) -> None:
        super(IIDSegmentationLoss, self).__init__()
        logger.trace(f"Initialize {self.__class__.__name__} with padding = {padding}.")
        self.lamda = lamda
        self.padding = padding
        self.eps = eps

    def __call__(
        self, x_out: Tensor, x_tf_out: Tensor, mask: Tensor = None
    ) -> Tensor:
        # assert x_out.requires_grad and x_tf_out.requires_grad
        if mask is not None:
            assert not mask.requires_grad
        assert simplex(x_out)
        assert x_out.shape == x_tf_out.shape
        bn, k, h, w = x_tf_out.shape
        if mask is not None:
            x_out = x_out * mask
            x_tf_out = x_tf_out * mask

        x_out = x_out.permute(1, 0, 2, 3).contiguous()  # k, ni, h, w
        x_tf_out = x_tf_out.permute(1, 0, 2, 3).contiguous()  # k, ni, h, w
        # k, k, 2 * half_T_side_dense + 1,2 * half_T_side_dense + 1
        p_i_j = F.conv2d(x_out, weight=x_tf_out, padding=(self.padding, self.padding))
        p_i_j = p_i_j - p_i_j.min().detach() + 1e-16
        T_side_dense = self.padding * 2 + 1

        # T x T x k x k
        p_i_j = p_i_j.permute(2, 3, 0, 1)
        p_i_j = p_i_j / p_i_j.sum(dim=3, keepdim=True).sum(dim=2, keepdim=True)  # norm

        # symmetrise, transpose the k x k part
        p_i_j = (p_i_j + p_i_j.permute(0, 1, 3, 2)) / 2.0

        # T x T x k x k
        p_i_mat = p_i_j.sum(dim=2, keepdim=True).repeat(1, 1, k, 1)
        p_j_mat = p_i_j.sum(dim=3, keepdim=True).repeat(1, 1, 1, k)

        # maximise information
        loss = (-p_i_j * (torch.log(p_i_j + 1e-16) - self.lamda * torch.log(p_i_mat + 1e-16) - self.lamda * torch.log(
            p_j_mat + 1e-16))).sum() / (T_side_dense * T_side_dense)
        if torch.isnan(loss):
            raise RuntimeError(loss)
        return loss


class IIDSegmentationSmallPathLoss(IIDSegmentationLoss):

    def __init__(self, lamda=1.0, padding=7, eps: float = sys.float_info.epsilon, patch_size=32) -> None:
        super().__init__(lamda, padding, eps)
        self._patch_size = _pair(patch_size)
        self._step_size = _pair(patch_size // 2)

    def __call__(self, x_out: Tensor, x_tf_out: Tensor, mask: Tensor = None):
        assert x_out.shape == x_tf_out.shape, (x_out.shape, x_tf_out.shape)
        if mask is None:
            iic_patch_list = [super(IIDSegmentationSmallPathLoss, self).__call__(x, y) for x, y in zip(
                patch_generator(x_out, self._patch_size, self._step_size),
                patch_generator(x_tf_out, self._patch_size, self._step_size)
            )]
        else:
            iic_patch_list = [super(IIDSegmentationSmallPathLoss, self).__call__(x, y, m) for x, y, m in zip(
                patch_generator(x_out, self._patch_size, self._step_size),
                patch_generator(x_tf_out, self._patch_size, self._step_size),
                patch_generator(mask, self._patch_size, self._step_size)
            )]
        if any([torch.isnan(x) for x in iic_patch_list]):
            raise RuntimeError(iic_patch_list)
        return average_iter(iic_patch_list)

    def __repr__(self):
        return f"{self.__class__.__name__} with patch_size={self._patch_size} and padding={self.padding}."


def compute_joint(x_out: Tensor, x_tf_out: Tensor, symmetric=True) -> Tensor:
    r"""
    return joint probability
    :param x_out: p1, simplex
    :param x_tf_out: p2, simplex
    :return: joint probability

    Args:
        symmetric: if doing it symmetrically.
    """
    # produces variable that requires grad (since args require grad)
    assert simplex(x_out), f"x_out not normalized."
    assert simplex(x_tf_out), f"x_tf_out not normalized."

    bn, k = x_out.shape
    assert x_tf_out.size()[0] == bn and x_tf_out.size()[1] == k

    p_i_j = x_out.unsqueeze(2) * x_tf_out.unsqueeze(1)  # bn, k, k
    p_i_j = p_i_j.sum(dim=0)  # k, k aggregated over one batch
    if symmetric:
        p_i_j = (p_i_j + p_i_j.t()) / 2.0  # symmetric
    p_i_j /= p_i_j.sum()  # normalise

    return p_i_j


def patch_generator(feature_map, patch_size=(32, 32), step_size=(16, 16)):
    b, c, h, w = feature_map.shape
    hs = np.arange(0, h - patch_size[0], step_size[0])
    hs = np.append(hs, max(h - patch_size[0], 0))
    ws = np.arange(0, w - patch_size[1], step_size[1])
    ws = np.append(ws, max(w - patch_size[1], 0))
    for _h in hs:
        for _w in ws:
            yield feature_map[:, :, _h:min(_h + patch_size[0], h), _w:min(_w + patch_size[1], w)]
