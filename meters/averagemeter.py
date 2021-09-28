import typing as t
from collections import defaultdict

import numpy as np

from .metric import Metric

metric_result = t.Union[float, np.ndarray]
dictionary_metric_result = Metric[t.Union[str, metric_result]]


class AverageValueMeter(Metric[metric_result]):
    def __init__(self):
        super(AverageValueMeter, self).__init__()
        self.reset()

    def _add(self, value, n=1):
        self.sum += value * n
        self.n += n

    def reset(self):
        self.sum = 0
        self.n = 0

    def _summary(self) -> metric_result:
        # this function returns a dict and tends to aggregate the historical results.
        if self.n == 0:
            return np.nan
        return float(self.sum / self.n)


class AverageValueDictionaryMeter(Metric[dictionary_metric_result]):
    def __init__(self) -> None:
        super().__init__()
        self._meter_dicts: t.Dict[str, AverageValueMeter] = defaultdict(AverageValueMeter)

    def reset(self):
        for k, v in self._meter_dicts.items():
            v.reset()

    def _add(self, **kwargs):
        for k, v in kwargs.items():
            self._meter_dicts[k].add(v)

    def _summary(self):
        return {k: v.summary() for k, v in self._meter_dicts.items()}


class AverageValueListMeter(AverageValueDictionaryMeter):
    def _add(self, list_value: t.Iterable[float] = None, **kwargs):
        assert isinstance(list_value, t.Iterable)
        for i, v in enumerate(list_value):
            self._meter_dicts[str(i)].add(v)
