import typing as t
from abc import abstractmethod

RETURN_TYPE = t.TypeVar("RETURN_TYPE")


class Metric(t.Generic[RETURN_TYPE]):
    _initialized = False

    def __init__(self, **kwargs) -> None:
        self._initialized = True

    @abstractmethod
    def reset(self):
        pass

    @t.final
    def add(self, *args, **kwargs):
        assert self._initialized, f"{self.__class__.__name__} must be initialized by overriding __init__"
        return self._add(*args, **kwargs)

    @abstractmethod
    def _add(self, *args, **kwargs):
        pass

    @t.final
    def summary(self) -> RETURN_TYPE:
        return self._summary()

    @abstractmethod
    def _summary(self) -> RETURN_TYPE:
        pass

    @t.final
    def join(self):
        return

    @t.final
    def close(self):
        return
