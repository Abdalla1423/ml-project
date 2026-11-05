"""Smoke tests for dopfn.viz (owner: C)."""

import inspect

from dopfn.viz import show


def test_show_signature() -> None:
    params = inspect.signature(show).parameters
    assert list(params)[:2] == ["x", "outfile"]
    assert params["outfile"].default is None
