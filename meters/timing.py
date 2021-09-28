import time
from pprint import pprint
import sys

sys.path.insert(0, "../")
from meters.averagemeter import AverageValueMeter


class Timer:
    def __init__(self) -> None:
        self._data_fetch = AverageValueMeter()
        self._forward = AverageValueMeter()
        self._data_start = time.time()

    def __enter__(self):
        data_end = time.time()
        self._data_fetch.add(data_end - self._data_start)
        self._forward_start = time.time()

    def __exit__(self, *args, **kwargs):
        forward_end = time.time()
        self._forward.add(forward_end - self._forward_start)
        self._data_start = time.time()

    def summary(self):
        return {"data": self._data_fetch.summary(),
                "forward": self._forward.summary()}
