"""Smoke tests for dopfn.data (owner: A)."""

import inspect

from conftest import not_implemented

from dopfn.data import load_data


def test_load_data_signature() -> None:
    params = inspect.signature(load_data).parameters
    assert params["dataset"].default == "prior"
    for name in ("transformation", "n_train", "n_test"):
        assert params[name].default is None


@not_implemented
def test_load_data_prior_yields_dataset() -> None:
    X_ob, y_ob, X_in, y_in, y_in_ob, cate = next(iter(load_data("prior", seed=0)))
    assert set(X_ob[:, 0].tolist()) <= {0, 1}
