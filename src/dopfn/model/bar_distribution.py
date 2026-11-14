"""Bar distribution (histogram over bins of y) as in Müller et al. (2022).

The model outputs one logit per bin. The density is piecewise constant: bin ``k`` has probability
``softmax(logits)[k]`` spread uniformly over ``[borders[k], borders[k + 1]]``. Values outside the
borders are clamped into the edge bins (TODO(team): D4 - a full-support version with half-normal
tails, as in TabPFN, would handle them exactly).
"""

import torch


class BarDistribution(torch.nn.Module):
    """Piecewise-constant density over fixed bin borders, parametrized by logits.

    Args:
        borders: Strictly increasing bin borders, shape ``(n_bins + 1,)``. They live on the
            normalized y scale (see ``dopfn.model.transformer.normalize_y``).
    """

    def __init__(self, borders: torch.Tensor) -> None:
        super().__init__()
        borders = torch.as_tensor(borders, dtype=torch.float32)
        if borders.ndim != 1 or borders.numel() < 2:
            raise ValueError("borders must be a 1-D tensor with at least 2 entries")
        if not torch.all(borders[1:] > borders[:-1]):
            raise ValueError("borders must be strictly increasing")
        self.register_buffer("borders", borders)

    @property
    def n_bins(self) -> int:
        """Number of bins."""
        return self.borders.numel() - 1

    @property
    def widths(self) -> torch.Tensor:
        """Bin widths, shape ``(n_bins,)``."""
        return self.borders[1:] - self.borders[:-1]

    @property
    def mids(self) -> torch.Tensor:
        """Bin centers, shape ``(n_bins,)``."""
        return (self.borders[1:] + self.borders[:-1]) / 2

    def bin_index(self, y: torch.Tensor) -> torch.Tensor:
        """Index of the bin containing ``y`` (clamped to the edge bins), same shape as ``y``."""
        idx = torch.searchsorted(self.borders, y.contiguous(), right=True) - 1
        return idx.clamp(0, self.n_bins - 1)

    def log_prob(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Log density of ``y`` under the distribution given by ``logits``.

        Args:
            logits: Shape ``(..., n_bins)``.
            y: Targets, shape ``(...)``.

        Returns:
            Log densities, shape ``(...)``.
        """
        idx = self.bin_index(y)
        log_p = torch.log_softmax(logits, dim=-1).gather(-1, idx.unsqueeze(-1)).squeeze(-1)
        return log_p - torch.log(self.widths)[idx]

    def mean(self, logits: torch.Tensor) -> torch.Tensor:
        """Mean of the distribution, shape ``(...)``."""
        return torch.softmax(logits, dim=-1) @ self.mids


def make_borders(n_bins: int, y_samples: torch.Tensor, margin: float = 0.5) -> torch.Tensor:
    """Bin borders with equal prior mass per bin (quantiles of ``y_samples``), as in PFNs.

    The outer borders are pushed out by ``margin`` times the range of the samples, so rare large
    values still fall into proper bins. Duplicate quantiles are spread apart to keep the borders
    strictly increasing.

    Args:
        n_bins: Number of bins.
        y_samples: Normalized target values sampled from the prior, shape ``(n,)``.
        margin: Relative extension of the outermost borders.

    Returns:
        Borders, shape ``(n_bins + 1,)``.
    """
    y = y_samples.flatten().float()
    y = y[torch.isfinite(y)]
    if y.numel() < n_bins:
        raise ValueError(f"need at least n_bins={n_bins} samples, got {y.numel()}")
    q = torch.linspace(0, 1, n_bins + 1, dtype=torch.float64)
    borders = torch.quantile(y.double(), q)
    span = (borders[-1] - borders[0]).clamp_min(1e-6)
    borders[0] -= margin * span
    borders[-1] += margin * span
    # Enforce strict monotonicity (many equal quantiles happen e.g. for discrete outcomes).
    # b_i - i * gap made non-decreasing by cummax, then shifted back: gaps are at least `gap` and
    # borders that already had larger gaps are unchanged.
    gap = span * 1e-6
    offsets = torch.arange(borders.numel(), dtype=torch.float64) * gap
    borders = torch.cummax(borders - offsets, dim=0).values + offsets
    return borders.float()
