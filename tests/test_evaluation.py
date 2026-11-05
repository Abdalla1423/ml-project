"""Smoke tests for dopfn.evaluation (owner: C)."""

import numpy as np
from conftest import not_implemented

from dopfn.evaluation import normalized_mse


@not_implemented
def test_normalized_mse_zero_for_perfect_prediction() -> None:
    y = np.array([0.0, 1.0, 2.0])
    assert normalized_mse(y, y) == 0.0
