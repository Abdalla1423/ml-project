"""Simplified Do-PFN transformer (nanoTabPFN style) with a treatment indicator on column 0.

Adapted from nanoTabPFN (https://github.com/automl/nanoTabPFN, ``model.py`` at commit ``6071eec``,
Copyright the nanoTabPFN authors, Apache License 2.0, see ``docs/third-party.md``). Changes:

* Column 0 (the treatment) gets a learned treatment embedding added to its cell embeddings, so the
  model knows which column is the treatment. This is Do-PFN's only architectural change to TabPFN
  (M5). The treatment is normalized and encoded like the other features: a separate treatment
  encoder (raw or normalized) made the model stay about twice as long on the initial plateau where
  it ignores the context (synthetic test, 3 seeds each).
* All-NaN feature columns are padding: they are masked in the attention between features, so a
  batch can mix datasets with different numbers of features (decision D2).
* ``y_ob`` is z-normalized per dataset inside the model; the output logits are over bins of the
  normalized outcome (decision D4).
* One attention call for the rows: every row attends to the observational rows only, which is the
  same as nanoTabPFN's separate train/test calls.
* Configurable dropout; output head with ``n_bins`` logits for a bar distribution.
"""

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import nn

from dopfn.model.bar_distribution import BarDistribution

# Small constant to avoid division by zero for constant columns / outcomes.
STD_EPS = 1e-6
# Normalized feature values are clipped to this range (as in nanoTabPFN).
FEATURE_CLIP = 100.0
# Range of the placeholder bin borders before ``set_borders`` is called (normalized scale).
DEFAULT_BORDER_RANGE = 5.0


@dataclass
class ModelConfig:
    """Model hyperparameters (``model`` section of ``configs/train/*.yaml``)."""

    n_bins: int = 1000
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 256
    dropout: float = 0.0


def y_stats(y_ob: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean and std of the observational outcomes per dataset.

    Args:
        y_ob: Shape ``(batch, n_ob)``.

    Returns:
        Mean and std, each of shape ``(batch, 1)``. The std is at least ``STD_EPS``.
    """
    mean = y_ob.mean(dim=-1, keepdim=True)
    std = y_ob.std(dim=-1, correction=0, keepdim=True).clamp_min(STD_EPS)
    return mean, std


def normalize_y(y: torch.Tensor, y_ob: torch.Tensor) -> torch.Tensor:
    """Map outcomes ``y`` of shape ``(batch, n)`` to the model's normalized scale (D4)."""
    mean, std = y_stats(y_ob)
    return (y - mean) / std


def denormalize_y(y_norm: torch.Tensor, y_ob: torch.Tensor) -> torch.Tensor:
    """Inverse of ``normalize_y``."""
    mean, std = y_stats(y_ob)
    return y_norm * std + mean


class TransformerLayer(nn.Module):
    """Attention between features, attention between rows, MLP. Post-norm, as in nanoTabPFN."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.feature_attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.row_attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, h: torch.Tensor, n_ob: int, column_padding: torch.Tensor) -> torch.Tensor:
        """Apply the layer.

        Args:
            h: Cell embeddings, shape ``(batch, rows, columns, d_model)``.
            n_ob: Number of observational rows (the first rows).
            column_padding: ``True`` for padded columns, shape ``(batch, columns)``.

        Returns:
            Updated embeddings, same shape as ``h``.
        """
        b, r, c, e = h.shape
        # Attention between the cells of one row (features + target); padded columns are no keys.
        x = h.reshape(b * r, c, e)
        mask = column_padding.repeat_interleave(r, dim=0)
        x = self.feature_attention(x, x, x, key_padding_mask=mask, need_weights=False)[0] + x
        h = self.norm1(x.reshape(b, r, c, e))

        # Attention between rows of one column: every row attends to the observational rows.
        x = h.transpose(1, 2).reshape(b * c, r, e)
        context = x[:, :n_ob]
        x = self.row_attention(x, context, context, need_weights=False)[0] + x
        h = self.norm2(x.reshape(b, c, r, e).transpose(1, 2))

        h = self.norm3(self.dropout(self.linear2(F.gelu(self.linear1(h)))) + h)
        return h


class Model(torch.nn.Module):
    """Prior-data fitted network predicting p(y | do(t), x, D_ob) for query rows.

    Args:
        n_bins: Number of bins of the bar distribution output head.
        d_model: Embedding size.
        n_heads: Attention heads.
        n_layers: Transformer layers.
        d_ff: Hidden size of the feed-forward blocks.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        n_bins: int = 1000,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 256,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.config = ModelConfig(n_bins, d_model, n_heads, n_layers, d_ff, dropout)
        self.feature_encoder = nn.Linear(1, d_model)
        self.treatment_embedding = nn.Parameter(torch.randn(d_model) * 0.02)
        self.target_encoder = nn.Linear(1, d_model)
        self.layers = nn.ModuleList(
            TransformerLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        )
        self.decoder = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, n_bins))
        # Bin borders are part of the trained model (saved in the state dict). Uniform placeholder
        # until training sets prior-based borders with ``set_borders``.
        self.bar = BarDistribution(
            torch.linspace(-DEFAULT_BORDER_RANGE, DEFAULT_BORDER_RANGE, n_bins + 1)
        )

    def set_borders(self, borders: torch.Tensor) -> None:
        """Set the bin borders of the output distribution, shape ``(n_bins + 1,)``."""
        borders = BarDistribution(borders).borders  # validates
        if borders.shape != self.bar.borders.shape:
            raise ValueError(f"Expected {self.bar.borders.numel()} borders, got {borders.numel()}")
        self.bar.borders.copy_(borders)

    @classmethod
    def from_config(cls, config: ModelConfig) -> "Model":
        """Build a model from a ``ModelConfig``."""
        return cls(**asdict(config))

    def forward(self, X: torch.Tensor, y_ob: torch.Tensor) -> torch.Tensor:
        """Predict a distribution over the outcomes of the query rows.

        Args:
            X: Batch of tables, observational rows followed by query rows, treatment in column 0.
                Shape ``(batch, rows, features)``. All-NaN columns are padding.
            y_ob: Observational outcomes, shape ``(batch, n_ob)``. The query rows of ``X`` are
                ``X[:, n_ob:]`` (decision D3).

        Returns:
            Bar distribution logits on the normalized outcome scale for every query row, shape
            ``(batch, rows - n_ob, n_bins)``.
        """
        b, r, f = X.shape
        n_ob = y_ob.shape[1]
        if y_ob.shape[0] != b or not 0 < n_ob < r:
            raise ValueError(f"Need y_ob of shape (batch, n_ob) with 0 < n_ob < rows, got {tuple(y_ob.shape)}")

        padding = torch.isnan(X).all(dim=1)  # (b, f)
        X = torch.nan_to_num(X, nan=0.0)

        # Features (incl. the treatment): z-normalized with observational statistics, clipped (as
        # nanoTabPFN), one shared encoder. The treatment column gets the treatment embedding (M5).
        mean = X[:, :n_ob].mean(dim=1, keepdim=True)
        std = X[:, :n_ob].std(dim=1, correction=0, keepdim=True).clamp_min(STD_EPS)
        x = self.feature_encoder(((X - mean) / std).clamp(-FEATURE_CLIP, FEATURE_CLIP).unsqueeze(-1))
        is_treatment = torch.zeros(f, 1, device=X.device, dtype=x.dtype)
        is_treatment[0] = 1.0
        x = x + is_treatment * self.treatment_embedding

        # Target column: normalized y_ob, query rows padded with 0 (= the observational mean).
        y = torch.cat([normalize_y(y_ob, y_ob), y_ob.new_zeros(b, r - n_ob)], dim=1)
        y = self.target_encoder(y.unsqueeze(-1).unsqueeze(-1))

        h = torch.cat([x, y], dim=2)  # (b, r, f + 1, d_model)
        column_padding = torch.cat([padding, padding.new_zeros(b, 1)], dim=1)
        for layer in self.layers:
            h = layer(h, n_ob, column_padding)
        return self.decoder(h[:, n_ob:, -1])
