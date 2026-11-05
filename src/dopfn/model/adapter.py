"""The one thin adapter between any data source and the model.

When Person A's ``load_data`` format changes, only this file changes.
"""

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class ModelBatch:
    """Model-ready batch.

    Attributes:
        X: Observational rows followed by query rows, shape ``(batch, rows, features)``.
        y_ob: Observational outcomes, shape ``(batch, n_ob)``.
        y_target: Targets for the query rows, shape ``(batch, rows - n_ob)``.
        n_ob: Index where query rows start.
    """

    X: torch.Tensor
    y_ob: torch.Tensor
    y_target: torch.Tensor
    n_ob: int


def to_model_batch(samples: list[Any], device: str | torch.device = "cpu") -> ModelBatch:
    """Convert a list of datasets (as yielded by ``load_data``) to a ``ModelBatch``.

    TODO(team): D1 (input format), D2 (padding), D4 (normalization).

    Args:
        samples: Datasets from ``dopfn.data.load_data``.
        device: Target device.

    Returns:
        The batch.
    """
    raise NotImplementedError
