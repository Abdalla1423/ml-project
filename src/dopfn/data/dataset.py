"""Container for one causal dataset, as yielded by ``dopfn.data.load_data``."""

from typing import NamedTuple

import torch


class CausalDataset(NamedTuple):
    """One whole dataset: observational context plus interventional query points.

    TODO(team): D1 - this is the proposed format (see ``docs/decisions.md``), to be confirmed.

    All tensors are ``float32`` on the CPU. The treatment is column 0 of ``X_ob`` and ``X_in``,
    with values in {0, 1}. Being a ``NamedTuple``, it unpacks like a plain tuple:
    ``X_ob, y_ob, X_in, y_in, y_in_ob, cate = sample``.

    Attributes:
        X_ob: Observational covariates, shape ``(n_train, F)``.
        y_ob: Observational outcomes, shape ``(n_train,)``.
        X_in: Query covariates, shape ``(n_test, F)``. Factual (observational) covariates, with
            column 0 set to the intervened treatment value (decision D5).
        y_in: Outcome under ``do(t = X_in[:, 0])`` with the query row's own noise, ``(n_test,)``.
        y_in_ob: Factual (observational) outcome of the query rows, ``(n_test,)``. For Dont-PFN.
        cate: Ground-truth effect ``y(do(1)) - y(do(0))`` with the query row's own noise,
            ``(n_test,)``. Same definition as the Do-PFN case studies (decision D8).
    """

    X_ob: torch.Tensor
    y_ob: torch.Tensor
    X_in: torch.Tensor
    y_in: torch.Tensor
    y_in_ob: torch.Tensor
    cate: torch.Tensor

    @property
    def n_train(self) -> int:
        """Number of observational rows."""
        return self.X_ob.shape[0]

    @property
    def n_test(self) -> int:
        """Number of query rows."""
        return self.X_in.shape[0]

    @property
    def n_features(self) -> int:
        """Number of columns of X, including the treatment."""
        return self.X_ob.shape[1]

    def validate(self) -> None:
        """Check shapes, dtypes, finiteness and the binary treatment column.

        Raises:
            ValueError: If any check fails.
        """
        n_ob, n_in = self.n_train, self.n_test
        expected = {
            "X_ob": (n_ob, self.n_features),
            "y_ob": (n_ob,),
            "X_in": (n_in, self.n_features),
            "y_in": (n_in,),
            "y_in_ob": (n_in,),
            "cate": (n_in,),
        }
        for name, shape in expected.items():
            value = getattr(self, name)
            if not isinstance(value, torch.Tensor):
                raise ValueError(f"{name} must be a torch.Tensor, got {type(value).__name__}")
            if tuple(value.shape) != shape:
                raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {shape}")
            if value.dtype != torch.float32:
                raise ValueError(f"{name} has dtype {value.dtype}, expected torch.float32")
            if not torch.isfinite(value).all():
                raise ValueError(f"{name} contains NaN or inf")
        for name in ("X_ob", "X_in"):
            t = getattr(self, name)[:, 0]
            if not ((t == 0) | (t == 1)).all():
                raise ValueError(f"column 0 of {name} (treatment) must be binary in {{0, 1}}")
