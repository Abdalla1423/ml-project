"""Simplified Do-PFN transformer (nanoTabPFN style) with a treatment indicator on column 0."""

import torch


class Model(torch.nn.Module):
    """Prior-data fitted network predicting p(y | do(t), x, D_ob) for query rows."""

    def __init__(
        self,
        n_bins: int = 100,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 256,
        dropout: float = 0.0,
    ) -> None:
        """Build the model.

        Args:
            n_bins: Number of bins of the bar distribution output head.
            d_model: Embedding size.
            n_heads: Attention heads.
            n_layers: Transformer layers.
            d_ff: Hidden size of the feed-forward blocks.
            dropout: Dropout probability.
        """
        super().__init__()
        raise NotImplementedError

    def forward(self, X: torch.Tensor, y_ob: torch.Tensor) -> torch.Tensor:
        """Predict a distribution over the outcomes of the query rows.

        Args:
            X: Batch of tables, observational rows followed by query rows, treatment in column 0.
                Shape ``(batch, rows, features)``.
            y_ob: Observational outcomes, shape ``(batch, n_ob)``. The query rows of ``X`` are
                ``X[:, n_ob:]``. TODO(team): D3 - is the split index implied by ``y_ob`` enough?

        Returns:
            Bar distribution logits for every query row, shape ``(batch, rows - n_ob, n_bins)``.
        """
        raise NotImplementedError
