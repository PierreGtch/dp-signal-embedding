import pytest
import numpy as np

from signal_embedding.models.skorch_nn import SkorchTransformer
from signal_embedding.models.jumping_means import JumpingMeansTransformer
from signal_embedding.models.covariance import CovarianceTransformer    


@pytest.mark.parametrize(
    "model_getter", [
        SkorchTransformer(),
        JumpingMeansTransformer(intervals_seconds=[(0, .1), (.1, .2), (.2, .3)]),
        CovarianceTransformer(delays=4, estimator="scm")
    ])
def test_model_getter(model_getter):
    model = model_getter(
        sfreq=128,
        input_window_seconds=3,
        chs_info=[{'ch_name': 'C3'}, {'ch_name': 'C4'}, {'ch_name': 'Cz'}]
    )
    x = np.random.randn(1, 3, 3 * 128 + 1).astype(np.float32)
    y = model.transform(x)
    assert y.ndim == 2
    assert y.shape[0] == 1
    assert isinstance(y, np.ndarray)
    assert y.dtype == np.float32
