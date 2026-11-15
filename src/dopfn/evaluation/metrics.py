"""Point prediction metrics for interventional outcomes, CATE, and ATE."""

import numpy as np


def _vector(values: np.ndarray, name: str) -> np.ndarray:
    """Accept a flat vector or one outcome column and reject other shapes."""
    values = np.asarray(values, dtype=float)
    if values.ndim == 2 and values.shape[1] == 1:
        values = values[:, 0]
    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
        raise ValueError(f"{name} must be a nonempty vector of finite values.")
    return values


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return mean squared error for aligned target and prediction vectors."""
    y_true = _vector(y_true, "y_true")
    y_pred = _vector(y_pred, "y_pred")
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must contain the same number of values.")
    return float(np.mean((y_true - y_pred) ** 2))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return root mean squared error in the target's original units."""
    return float(np.sqrt(mse(y_true, y_pred)))


def ate_relative_error(cate_true: np.ndarray, cate_pred: np.ndarray) -> float:
    """Compare ATEs averaged over the same query covariates; zero true ATE is undefined."""
    cate_true = _vector(cate_true, "cate_true")
    cate_pred = _vector(cate_pred, "cate_pred")
    if cate_true.shape != cate_pred.shape:
        raise ValueError("cate_true and cate_pred must contain the same number of values.")
    # 先分别平均 CATE 得到 ATE，再计算相对误差，不是逐人的相对误差平均。
    truth = float(cate_true.mean())
    prediction = float(cate_pred.mean())
    return abs(prediction - truth) / abs(truth) if truth else float("nan")


def normalized_mse(
    y_true: np.ndarray, y_pred: np.ndarray, *, scale: float | None = None
) -> float:
    """Divide MSE by the squared target range, or by an explicit squared scale.

    Return NaN when the range is zero because normalization is undefined.
    """
    error = mse(y_true, y_pred)
    scale = float(np.ptp(y_true)) if scale is None else scale
    if not np.isfinite(scale) or scale < 0:
        raise ValueError("scale must be finite and nonnegative.")
    return error / scale**2 if scale else float("nan")
