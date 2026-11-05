"""Bar distribution (histogram over bins of y) as in Müller et al. (2022)."""

import torch


class BarDistribution(torch.nn.Module):
    """Piecewise-constant density over fixed bin borders, parametrized by logits."""

    def __init__(self, borders: torch.Tensor) -> None:
        """Args:
            borders: Increasing bin borders, shape ``(n_bins + 1,)``.
                TODO(team): D4 - borders depend on how y is normalized.
        """
        super().__init__()
        raise NotImplementedError

    def log_prob(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Log density of ``y`` under the distribution given by ``logits``.

        Args:
            logits: Shape ``(..., n_bins)``.
            y: Targets, shape ``(...)``.

        Returns:
            Log densities, shape ``(...)``.
        """
        raise NotImplementedError

    def mean(self, logits: torch.Tensor) -> torch.Tensor:
        """Mean of the distribution, shape ``(...)``."""
        raise NotImplementedError
