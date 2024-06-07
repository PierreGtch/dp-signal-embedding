from abc import ABCMeta, abstractmethod
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


@runtime_checkable
class SklearnTransformer(Protocol):
    def transform(self, X: NDArray[np.float32]) -> NDArray[np.float32]:
        pass


class ModelGetter(metaclass=ABCMeta):
    @abstractmethod
    def __call__(
            self,
            sfreq: float,
            input_window_seconds: float,
            chs_info: list[dict],
    ) -> SklearnTransformer:
        pass
