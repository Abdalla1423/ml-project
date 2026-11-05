"""Metrics."""

import numpy as np


def normalized_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MSE normalized by the range of ``y_true`` (paper metric).

    ``mean(((y_true - y_pred) / (max(y_true) - min(y_true))) ** 2)``
    """
    raise NotImplementedError
