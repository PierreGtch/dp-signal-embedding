from dataclasses import dataclass
import numpy as np
from sklearn.preprocessing import FunctionTransformer

from signal_embedding.models.base import ModelGetter


def jumping_means(X, intervals_samples: list[tuple[int, int]]):
    """Calculate the mean of the signal in the intervals.

    Parameters
    ----------
    X: np.ndarray (batch, n_chans, n_times)
        Signal to calculate the means.
    intervals_samples: list[tuple[int, int]]
        List of tuples with the start and end of the intervals.

    Returns
    -------
    np.ndarray
        Array with the means of the signal in the intervals.
    """
    assert X.ndim == 3
    means = np.concatenate(
        [np.mean(X[:, :, start:end], axis=-1) for start, end in intervals_samples],
        axis=-1,
    )
    return means


@dataclass
class JumpingMeansTransformer(ModelGetter):
    intervals_seconds: list[tuple[float, float]]

    def __call__(
        self,
        sfreq: float,
        input_window_seconds: float,
        chs_info: list[dict],
    ):
        intervals_samples = [
            (int(start * sfreq), int(end * sfreq))
            for start, end in self.intervals_seconds
        ]
        assert all(
            (end <= input_window_seconds * sfreq) and (start < end) and (start >= 0)
            for start, end in intervals_samples
        )
        transformer = FunctionTransformer(
            jumping_means,
            kw_args=dict(intervals_samples=intervals_samples),
        )
        return transformer
