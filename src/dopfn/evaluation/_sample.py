"""Read and validate the current loader sample format at one boundary."""

from typing import Any

import numpy as np

from dopfn.evaluation.metrics import _vector


def unpack_sample(sample: Any) -> tuple[np.ndarray, ...]:
    """Return observational data, intervention queries, and query CATE labels."""
    # TODO(team): D1 - adapt here if the loader's final tuple format changes.
    X_ob, y_ob, X_in, y_in, _, cate = sample
    X_ob, X_in = np.asarray(X_ob), np.asarray(X_in)
    y_ob = _vector(y_ob, "y_ob")
    y_in = _vector(y_in, "y_in")
    cate = _vector(cate, "cate")
    for name, X, y in (("X_ob", X_ob, y_ob), ("X_in", X_in, y_in)):
        if X.ndim != 2 or not X.shape[1] or len(X) != len(y):
            raise ValueError(f"{name} must be a 2D table aligned with its outcomes.")
        if not np.isfinite(X).all() or not np.isin(X[:, 0], (0, 1)).all():
            raise ValueError(f"{name} must be finite with binary treatment in column 0.")
    if X_ob.shape[1] != X_in.shape[1] or len(cate) != len(y_in):
        raise ValueError("Feature columns and query CATE labels must align.")
    return X_ob, y_ob, X_in, y_in, cate
