from dataclasses import dataclass

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
import pyriemann

from signal_embedding.models.base import ModelGetter
def format_output(X):
    return X.astype("float32")

@dataclass
class CovarianceTransformer(ModelGetter):
    """
    Output shape: n * (n + 1) / 2
    With n = n_channels * delays
    """
    delays: int = 4
    estimator: str = "scm"

    def __call__(
        self,
        sfreq: float,
        input_window_seconds: float,
        chs_info: list[dict],
    ):

        return make_pipeline(
            pyriemann.estimation.TimeDelayCovariances(
                delays=self.delays, estimator=self.estimator
            ),
            FunctionTransformer(pyriemann.utils.base.logm),
            FunctionTransformer(pyriemann.utils.tangentspace.upper),
            FunctionTransformer(format_output),
        )
