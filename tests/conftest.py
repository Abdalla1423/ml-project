"""Shared pytest config."""

import pytest

# Mark tests of not-yet-implemented deliverables with this; remove once implemented.
not_implemented = pytest.mark.xfail(raises=NotImplementedError, reason="stub", strict=True)
