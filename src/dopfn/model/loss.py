"""Training loss: NLL of interventional outcomes under the predicted bar distribution."""

import torch

from dopfn.model.bar_distribution import BarDistribution


def nll_loss(logits: torch.Tensor, y_target: torch.Tensor, dist: BarDistribution) -> torch.Tensor:
    """Mean negative log-likelihood over all query rows in the batch.

    Args:
        logits: Model output, shape ``(batch, n_query, n_bins)``.
        y_target: Interventional outcomes of the query rows on the normalized scale (see
            ``dopfn.model.transformer.normalize_y``), shape ``(batch, n_query)``.
        dist: Bar distribution defining the bins.

    Returns:
        Scalar loss.
    """
    return -dist.log_prob(logits, y_target).mean()
