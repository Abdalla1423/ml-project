"""The one thin adapter between any data source and the model.

When Person A's ``load_data`` format changes, only this file changes.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import torch

from dopfn.data.dataset import CausalDataset

Target = Literal["interventional", "observational"]


@dataclass
class ModelBatch:
    """Model-ready batch.

    Attributes:
        X: Observational rows followed by query rows, shape ``(batch, rows, features)``. Feature
            columns padded with NaN (decision D2).
        y_ob: Observational outcomes, shape ``(batch, n_ob)``.
        y_target: Targets for the query rows (original scale), shape ``(batch, rows - n_ob)``.
        n_ob: Index where query rows start.
        cate: Ground-truth CATE of the query rows, shape ``(batch, rows - n_ob)``.
    """

    X: torch.Tensor
    y_ob: torch.Tensor
    y_target: torch.Tensor
    n_ob: int
    cate: torch.Tensor


def to_model_batch(
    samples: Sequence[CausalDataset],
    device: str | torch.device = "cpu",
    target: Target = "interventional",
) -> ModelBatch:
    """Convert a list of datasets (as yielded by ``load_data``) to a ``ModelBatch``.

    TODO(team): D2 - all datasets of a batch must have the same ``n_train`` and ``n_test``;
    feature columns are padded with NaN to the largest feature count.

    Args:
        samples: Datasets from ``dopfn.data.load_data``.
        device: Target device.
        target: ``"interventional"`` trains on ``y_in`` (Do-PFN). ``"observational"`` trains on
            the factual outcomes ``y_in_ob`` of the query rows (plain regression, a sanity check
            and Dont-PFN in ms2).

    Returns:
        The batch.

    Raises:
        ValueError: Empty list, unequal sizes or unknown target.
    """
    if not samples:
        raise ValueError("samples is empty")
    if target not in ("interventional", "observational"):
        raise ValueError(f"Unknown target {target!r}")
    n_ob, n_in = samples[0].n_train, samples[0].n_test
    if any(s.n_train != n_ob or s.n_test != n_in for s in samples):
        raise ValueError("All datasets in a batch need the same n_train and n_test (decision D2).")

    n_features = max(s.n_features for s in samples)
    X = torch.full((len(samples), n_ob + n_in, n_features), float("nan"))
    for i, s in enumerate(samples):
        X[i, :, : s.n_features] = torch.cat([s.X_ob, s.X_in])
    y_target = torch.stack([s.y_in if target == "interventional" else s.y_in_ob for s in samples])
    return ModelBatch(
        X=X.to(device),
        y_ob=torch.stack([s.y_ob for s in samples]).to(device),
        y_target=y_target.to(device),
        n_ob=n_ob,
        cate=torch.stack([s.cate for s in samples]).to(device),
    )
