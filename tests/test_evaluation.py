"""Tests for Part C metrics and estimator-independent evaluation."""

import numpy as np
import pytest

from dopfn.evaluation import ate_relative_error, evaluate, mse, normalized_mse, rmse
from dopfn.evaluation._synthetic import synthetic_sample


def test_metrics_accept_column_vectors_without_broadcasting() -> None:
    truth = np.array([0.0, 2.0])
    predictions = np.array([[2.0], [2.0]])
    assert mse(truth, predictions) == 2.0
    assert rmse(truth, predictions) == pytest.approx(np.sqrt(2))
    assert normalized_mse(truth, predictions) == 0.5
    assert normalized_mse(truth, truth) == 0.0
    with pytest.raises(ValueError, match="same number"):
        mse(truth, predictions[:1])
    with pytest.raises(ValueError, match="finite"):
        mse(truth, np.array([np.nan, 1]))


class _KnownRegressor:
    """Implement the public B interface and return the synthetic SCM's answers."""

    def fit(self, X_ob: np.ndarray, y_ob: np.ndarray) -> None:
        expected_X, expected_y, *_ = synthetic_sample()
        np.testing.assert_array_equal(X_ob, expected_X)
        np.testing.assert_array_equal(y_ob, expected_y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return 30 + 5 * X[:, 1] + X[:, 0] * self.predict_cate(X)

    def predict_cate(self, X: np.ndarray) -> np.ndarray:
        return 10 + 2 * X[:, 1]


def test_evaluate_uses_observational_context_and_query_truth() -> None:
    results = evaluate(_KnownRegressor(), (sample for sample in [synthetic_sample()]))
    assert all(results[key] == pytest.approx(0.0) for key in results
               if not key.startswith("n_") and "_ci95_" not in key)
    assert results["n_datasets"] == results["n_nmse_datasets"] == 1
    assert results["n_ate_datasets"] == 1


def test_ate_relative_error_averages_effects_before_comparison() -> None:
    assert ate_relative_error(np.array([-2., 4.]), np.array([[0.], [2.]])) == 0
    assert ate_relative_error(np.array([-2., -4.]), np.array([-1., -2.])) == 0.5
    assert np.isnan(ate_relative_error(np.array([-1., 1.]), np.array([0., 2.])))
    assert ate_relative_error(np.array([1e-12]), np.array([2e-12])) == pytest.approx(1)
    with pytest.raises(ValueError, match="same number"):
        ate_relative_error(np.ones(2), np.ones(3))


def test_undefined_ate_is_excluded_with_its_own_valid_count() -> None:
    sample = synthetic_sample()
    zero = (*sample[:-1], np.zeros_like(sample[-1]))
    scores = evaluate(_KnownRegressor(), [sample, zero])
    assert scores["n_datasets"] == 2
    assert scores["n_ate_datasets"] == 1
    assert scores["ate_relative_error"] == 0
    assert np.isnan(scores["ate_relative_error_ci95_low"])


def test_comparison_intervals_use_independent_datasets() -> None:
    sample = synthetic_sample()
    biased = (*sample[:-1], sample[-1] + 10)
    error = ate_relative_error(biased[-1], sample[-1])
    scores = evaluate(_KnownRegressor(), [sample, biased])
    assert scores["ate_relative_error"] == pytest.approx(error / 2)
    assert scores["ate_relative_error_ci95_low"] == 0
    assert scores["ate_relative_error_ci95_high"] == pytest.approx(error)


def test_cate_nmse_uses_outcome_range_even_for_constant_cate() -> None:
    X_ob, y_ob, X_in, _, y_in_ob, _ = synthetic_sample()
    model = _KnownRegressor()
    y_in = model.predict(X_in)
    cate = np.full(len(X_in), 10)
    scores = evaluate(model, [(X_ob, y_ob, X_in, y_in, y_in_ob, cate)])
    assert scores["cate_mse"] == pytest.approx(np.mean((2 * X_in[:, 1]) ** 2))
    assert scores["cate_nmse"] == pytest.approx(scores["cate_mse"] / np.ptp(y_in) ** 2)


def test_degenerate_outcome_still_contributes_to_unnormalized_scores() -> None:
    X_ob, y_ob, X_in, y_in, y_in_ob, cate = synthetic_sample()
    model = _KnownRegressor()
    regular = (X_ob, y_ob, X_in, y_in, y_in_ob, cate)
    constant = (X_ob, y_ob, X_in, np.zeros_like(y_in), y_in_ob, cate)
    scores = evaluate(model, iter([regular, constant]))
    assert scores["n_datasets"] == 2
    assert scores["n_nmse_datasets"] == 1
    assert scores["outcome_nmse"] == 0.0
    assert scores["outcome_mse"] == pytest.approx(np.mean(y_in**2) / 2)
    assert np.isnan(evaluate(model, [constant])["outcome_nmse"])
    assert np.isnan(normalized_mse(np.zeros(2), np.ones(2)))
    with pytest.raises(ValueError, match="at least one"):
        evaluate(model, [])
