from dataclasses import dataclass

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
import pyriemann

from signal_embedding.models.base import ModelGetter


@dataclass
class CovarianceTransformer(ModelGetter):
    delays: int = 10
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
        )
