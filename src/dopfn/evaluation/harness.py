"""Evaluation harness, shared by the prototype and all baselines."""

from collections.abc import Iterable
from typing import Any, Protocol

import numpy as np

from dopfn.evaluation._sample import unpack_sample
from dopfn.evaluation.metrics import ate_relative_error, mse, normalized_mse, rmse


class CausalEstimator(Protocol):
    """Anything with the sklearn-like causal interface."""

    def fit(self, X_ob: np.ndarray, y_ob: np.ndarray) -> Any: ...
    def predict(self, X: np.ndarray) -> np.ndarray: ...
    def predict_cate(self, X: np.ndarray) -> np.ndarray: ...


def evaluate(estimator: CausalEstimator, datasets: Iterable[Any]) -> dict[str, float]:
    """Score held-out query rows after fitting each dataset's observational rows.

    Each dataset has equal weight. For both targets, normalize by the range of
    interventional outcomes in that dataset. Constant-outcome datasets still
    contribute to MSE and RMSE. Zero true ATE has undefined relative error.
    Report valid counts, medians over datasets and 95% bootstrap intervals of the mean for the
    three comparison metrics.
    """
    totals = dict.fromkeys(
        (f"{target}_{metric}" for target in ("outcome", "cate")
         for metric in ("mse", "rmse", "nmse")),
        0.0,
    )
    n_datasets = n_normalized = 0
    comparison_values = {key: [] for key in ("outcome_nmse", "cate_nmse", "ate_relative_error")}
    for sample in datasets:
        X_ob, y_ob, X_in, y_in, cate = unpack_sample(sample)
        estimator.fit(X_ob, y_ob)
        predictions = {
            "outcome": (y_in, estimator.predict(X_in)),
            "cate": (cate, estimator.predict_cate(X_in)),
        }
        scale = float(np.ptp(y_in))
        for target, (truth, prediction) in predictions.items():
            totals[f"{target}_mse"] += mse(truth, prediction)
            totals[f"{target}_rmse"] += rmse(truth, prediction)
            if scale:
                value = normalized_mse(truth, prediction, scale=scale)
                totals[f"{target}_nmse"] += value
                comparison_values[f"{target}_nmse"].append(value)
        ate_error = ate_relative_error(*predictions["cate"])
        if np.isfinite(ate_error):
            comparison_values["ate_relative_error"].append(ate_error)
        n_datasets += 1
        n_normalized += bool(scale)
    if not n_datasets:
        raise ValueError("Evaluation requires at least one dataset.")
    for metric, total in totals.items():
        count = n_normalized if metric.endswith("_nmse") else n_datasets
        totals[metric] = total / count if count else float("nan")
    ate_errors = comparison_values["ate_relative_error"]
    totals["ate_relative_error"] = float(np.mean(ate_errors)) if ate_errors else float("nan")
    # 以 SCM 为重采样单位计算均值的 95% bootstrap 区间，不把查询行当独立 SCM。
    rng = np.random.default_rng(0)
    for metric, values in comparison_values.items():
        low = high = float("nan")
        if len(values) > 1:
            means = rng.choice(values, size=(2000, len(values)), replace=True).mean(axis=1)
            low, high = np.quantile(means, [0.025, 0.975])
        # The median is robust to single datasets with a near-constant outcome, where dividing by
        # the squared range makes the normalized error explode.
        totals[f"{metric}_median"] = float(np.median(values)) if values else float("nan")
        totals[f"{metric}_ci95_low"] = float(low)
        totals[f"{metric}_ci95_high"] = float(high)
    return {**totals, "n_datasets": n_datasets, "n_nmse_datasets": n_normalized,
            "n_ate_datasets": len(ate_errors)}
