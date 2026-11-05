"""Evaluation harness, shared by the prototype and all baselines."""

from collections.abc import Iterable
from typing import Any, Protocol

import numpy as np


class CausalEstimator(Protocol):
    """Anything with the sklearn-like causal interface."""

    def fit(self, X_ob: np.ndarray, y_ob: np.ndarray) -> Any: ...
    def predict(self, X: np.ndarray) -> np.ndarray: ...
    def predict_cate(self, X: np.ndarray) -> np.ndarray: ...


def evaluate(estimator: CausalEstimator, datasets: Iterable[Any]) -> dict[str, float]:
    """Fit on each dataset's observational part and score on its query part.

    Args:
        estimator: Object with ``fit`` / ``predict`` / ``predict_cate``.
        datasets: Items as yielded by ``dopfn.data.load_data``.

    Returns:
        Aggregated metrics, e.g. ``{"outcome_nmse": ..., "cate_nmse": ...}``.
    """
    raise NotImplementedError
