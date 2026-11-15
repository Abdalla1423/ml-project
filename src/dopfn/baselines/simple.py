"""Simple baselines from the ms1 sheet."""

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression
from sklearn.utils.validation import check_array, check_is_fitted


def _features(X: np.ndarray, n_features: int | None = None) -> np.ndarray:
    """Validate a table with binary treatment in column zero."""
    X = check_array(X, dtype=float)
    if n_features is not None and X.shape[1] != n_features:
        raise ValueError("X must have the same number of columns as X_ob.")
    if not np.isin(X[:, 0], (0, 1)).all():
        raise ValueError("Treatment in column 0 must be binary (0 or 1).")
    return X

def _observations(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Validate aligned outcomes and require both treatment groups."""
    X = _features(X)
    y = np.asarray(y, dtype=float)
    if y.shape == (len(X), 1):
        y = y[:, 0]
    if y.shape != (len(X),) or not np.isfinite(y).all():
        raise ValueError("y_ob must contain one finite outcome per row of X_ob.")
    if np.unique(X[:, 0]).size != 2:
        raise ValueError("X_ob must contain both treatment groups.")
    return X, y


class ObservationalRegression(BaseEstimator):
    """Fit a linear outcome model with treatment and covariate interactions."""
    def __init__(self, estimator: Any | None = None) -> None:
        """Use linear regression by default, or a supplied sklearn regressor."""
        self.estimator = estimator
    @staticmethod
    def _design(X: np.ndarray) -> np.ndarray:
        """Add T × covariate terms so slopes can differ by treatment."""
        return np.column_stack((X, X[:, :1] * X[:, 1:]))
    def fit(self, X_ob: np.ndarray, y_ob: np.ndarray) -> "ObservationalRegression":
        """Fit the regressor on observational data."""
        X_ob, y_ob = _observations(X_ob, y_ob)
        estimator = LinearRegression() if self.estimator is None else self.estimator
        estimator.fit(self._design(X_ob), y_ob)
        self.estimator_ = estimator
        self.n_features_in_ = X_ob.shape[1]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict one outcome per row at the treatment specified in column zero."""
        check_is_fitted(self, "estimator_")
        X = _features(X, self.n_features_in_)
        return self.estimator_.predict(self._design(X))

    def predict_cate(self, X: np.ndarray) -> np.ndarray:
        """Contrast treatment and control at the same covariates.

        Observational regression may fail to estimate the total treatment effect
        because it holds post-treatment mediators fixed when computing CATE.
        """
        check_is_fitted(self, "estimator_")
        X_0 = _features(X, self.n_features_in_).copy()
        X_1 = X_0.copy() # 复制一份数据 因为接下来需要分别修改 Treatment，而且不希望修改原始的 X。
        X_0[:, 0] = 0
        X_1[:, 0] = 1
        return self.estimator_.predict(self._design(X_1)) - self.estimator_.predict(self._design(X_0))


class GroupMeanDifference(BaseEstimator):
    """Predict the observed group mean and use the mean difference as constant CATE."""
    def fit(self, X_ob: np.ndarray, y_ob: np.ndarray) -> "GroupMeanDifference":
        """Compute the control and treated group means, in that order."""
        X_ob, y_ob = _observations(X_ob, y_ob)

        self.means_ = np.array([y_ob[X_ob[:, 0] == t].mean() for t in (0, 1)])
        self.n_features_in_ = X_ob.shape[1]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return the group mean for each row's treatment."""
        check_is_fitted(self, "means_")
        X = _features(X, self.n_features_in_)
        return self.means_[X[:, 0].astype(int)]
                 #return eg： return np.array([85, 65, 85, 65])

    def predict_cate(self, X: np.ndarray) -> np.ndarray:
        """Return the difference of observed group means for every query row."""
        check_is_fitted(self, "means_")
        X = _features(X, self.n_features_in_)
        return np.full(len(X), self.means_[1] - self.means_[0])
