"""Deterministic synthetic SCM for local tests and the CLI demo only."""

import networkx as nx
import numpy as np


def synthetic_sample(seed: int = 1000) -> tuple[np.ndarray, ...]:
    """Return one sample with Z -> T, Z -> Y and T -> Y.

    Y = 30 + 5 Z + T (10 + 2 Z). T depends on Z in the observational
    sample and is assigned independently in the interventional queries.
    """
    rng = np.random.default_rng(seed)
    z_ob = rng.normal(size=120)
    X_ob = np.column_stack((rng.random(len(z_ob)) < 1 / (1 + np.exp(-2 * z_ob)), z_ob))
    z_in = rng.uniform(-2, 2, size=60)
    X_in = np.column_stack((rng.permutation(np.repeat([0, 1], 30)), z_in))
    cate = 10 + 2 * z_in
    y_ob = 30 + 5 * z_ob + X_ob[:, 0] * (10 + 2 * z_ob)
    y_in = 30 + 5 * z_in + X_in[:, 0] * cate
    t_in_ob = rng.random(len(z_in)) < 1 / (1 + np.exp(-2 * z_in))
    y_in_ob = 30 + 5 * z_in + t_in_ob * cate
    return X_ob, y_ob, X_in, y_in, y_in_ob, cate


def synthetic_graph() -> nx.DiGraph:
    """Return the test-only SCM graph."""
    return nx.DiGraph([("Z", "T"), ("Z", "Y"), ("T", "Y")])
