"""Smoke tests for dopfn.model (owner: B)."""

import inspect

import torch
from conftest import not_implemented

from dopfn.model import Model, Regressor


def test_model_interface() -> None:
    assert issubclass(Model, torch.nn.Module)
    assert list(inspect.signature(Model.forward).parameters) == ["self", "X", "y_ob"]


def test_regressor_interface() -> None:
    for method in ("fit", "predict", "predict_cate"):
        assert callable(getattr(Regressor, method))


@not_implemented
def test_model_forward_shape() -> None:
    model = Model(n_bins=10)
    X, y_ob = torch.randn(2, 30, 4), torch.randn(2, 20)
    assert model(X, y_ob).shape == (2, 10, 10)
