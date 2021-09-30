import torch
from torch import Tensor, nn

from ._base import LossClass


class LossCompose(nn.Module, LossClass[Tensor]):

    def forward(self, *args, **kwargs) -> Tensor:
        if len(self._criteria) == 0:
            raise RuntimeError("no loss registered")
        losses = [criterion(*args, **kwargs) for criterion in self._criteria]
        return sum([l * w for l, w in zip(losses, self._weights)])

    def __init__(self):
        super().__init__()
        self._criteria = []  # 😀
        self._weights = []

    def __len__(self):
        return len(self._criteria)

    def register_loss(self, loss: LossClass[Tensor], weight: float):
        self._criteria.append(loss)
        self._weights.append(float(weight))


class IdenticalLoss(nn.Module, LossClass[Tensor]):

    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, *args, **kwargs):
        return torch.tensor(0, device="cuda")
