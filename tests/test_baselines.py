"""Numerical checks of the Part C baselines on a known SCM."""

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from dopfn.baselines import GroupMeanDifference, ObservationalRegression
from dopfn.evaluation._synthetic import synthetic_sample


def test_group_means_use_observational_data_and_ignore_covariates() -> None:
    X_ob, y_ob, X_in, _, _, cate = synthetic_sample()
    model = GroupMeanDifference().fit(X_ob, y_ob)
    expected = np.array([y_ob[X_ob[:, 0] == t].mean() for t in (0, 1)])
    np.testing.assert_allclose(model.predict(X_in), expected[X_in[:, 0].astype(int)])
    np.testing.assert_allclose(model.predict_cate(X_in), expected[1] - expected[0])
    assert abs(model.predict_cate(X_in).mean() - cate.mean()) > 1


def test_linear_regression_recovers_heterogeneous_effect_without_mutating_x() -> None:
    X_ob, y_ob, X_in, y_in, _, cate = synthetic_sample()
    model = ObservationalRegression().fit(X_ob, y_ob[:, None])
    original = X_in.copy()
    np.testing.assert_allclose(model.predict(X_in), y_in)
    np.testing.assert_allclose(model.predict_cate(X_in), cate)
    np.testing.assert_array_equal(X_in, original)


@pytest.mark.parametrize("estimator", [GroupMeanDifference, ObservationalRegression])
def test_baseline_requires_fit_and_two_groups(estimator: type) -> None:
    model = estimator()
    with pytest.raises(NotFittedError):
        model.predict(np.array([[0, 1]]))
    with pytest.raises(ValueError, match="both treatment groups"):
        model.fit(np.array([[0, 1], [0, 2]]), np.array([1, 2]))
    with pytest.raises(ValueError, match="binary"):
        model.fit(np.array([[0, 1], [2, 2]]), np.array([1, 2]))


@pytest.mark.parametrize("estimator", [GroupMeanDifference, ObservationalRegression])
def test_treatment_only_data(estimator: type) -> None:
    model = estimator().fit(np.array([[0], [1]]), np.array([5, 9]))
    np.testing.assert_allclose(model.predict(np.array([[1], [0]])), [9, 5])
    np.testing.assert_allclose(model.predict_cate(np.array([[1], [0]])), [4, 4])
