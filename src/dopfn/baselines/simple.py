"""Simple baselines from the ms1 sheet."""

from typing import Any

import numpy as np


class ObservationalRegression:
    """Regression fitted on observational data, queried with column 0 set to 0 and 1."""

    def __init__(self, estimator: Any | None = None) -> None:
        """Args:
            estimator: sklearn regressor (default chosen by owner).
        """
        raise NotImplementedError

    def fit(self, X_ob: np.ndarray, y_ob: np.ndarray) -> "ObservationalRegression":
        """Fit the regressor on observational data."""
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict outcomes at the treatment given in column 0, shape ``(n,)``."""
        raise NotImplementedError

    def predict_cate(self, X: np.ndarray) -> np.ndarray:
        """``predict(X with t=1) - predict(X with t=0)``, shape ``(n,)``."""
        raise NotImplementedError


class GroupMeanDifference:
    """Constant CATE = mean(y | t=1) - mean(y | t=0); outcome = group mean of the given t."""

    def fit(self, X_ob: np.ndarray, y_ob: np.ndarray) -> "GroupMeanDifference":
        """Compute the two group means."""
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Group mean of the treatment in column 0, shape ``(n,)``."""
        raise NotImplementedError

    def predict_cate(self, X: np.ndarray) -> np.ndarray:
        """Difference of group means for every row, shape ``(n,)``."""
        raise NotImplementedError
