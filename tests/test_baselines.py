"""Smoke tests for dopfn.baselines (owner: C)."""

from dopfn.baselines import GroupMeanDifference, ObservationalRegression


def test_baseline_interfaces() -> None:
    for cls in (GroupMeanDifference, ObservationalRegression):
        for method in ("fit", "predict", "predict_cate"):
            assert callable(getattr(cls, method))
