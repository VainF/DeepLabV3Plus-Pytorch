from typing import Generic, TypeVar, Tuple

from torch import Tensor

ReturnType = TypeVar('ReturnType', Tensor, Tuple[Tensor])


class LossClass(Generic[ReturnType]):

    def __call__(self, *args, **kwargs) -> ReturnType:
        pass

    def forward(self, *args, **kwargs) -> ReturnType:
        pass
