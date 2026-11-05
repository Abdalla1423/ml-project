"""Smoke tests for dopfn.training (owner: B)."""

from dopfn.training import TrainConfig


def test_train_config_defaults() -> None:
    assert TrainConfig().steps > 0
