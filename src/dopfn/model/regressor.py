"""Scikit-learn-like wrapper around a trained ``Model`` (like ``DoPFNRegressor``)."""

from pathlib import Path

import numpy as np
import torch

from dopfn.model.transformer import Model, denormalize_y, y_stats

ArrayLike = np.ndarray | torch.Tensor


def _as_tensor(a: ArrayLike, device: torch.device) -> torch.Tensor:
    """Float32 tensor on ``device`` from a numpy array or tensor."""
    if not isinstance(a, torch.Tensor):
        a = np.asarray(a)
    return torch.as_tensor(a, dtype=torch.float32).to(device)


class Regressor:
    """In-context regressor: ``fit`` stores data, ``predict`` runs the model.

    Args:
        model: A trained ``Model`` (its bin borders are part of the model).
        device: Device for inference.
    """

    def __init__(self, model: Model, device: str | torch.device = "cpu") -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.X_ob: torch.Tensor | None = None
        self.y_ob: torch.Tensor | None = None

    @classmethod
    def from_checkpoint(cls, path: str | Path, device: str | torch.device = "cpu") -> "Regressor":
        """Load a ``Regressor`` from a checkpoint written by ``dopfn.training``."""
        from dopfn.training.loop import load_checkpoint  # avoid an import cycle

        return cls(load_checkpoint(path, map_location=device), device=device)

    def fit(self, X_ob: ArrayLike, y_ob: ArrayLike) -> "Regressor":
        """Only store the observational data (no training).

        Args:
            X_ob: Observational covariates, treatment in column 0, shape ``(n_ob, features)``.
            y_ob: Observational outcomes, shape ``(n_ob,)``.

        Returns:
            ``self``.
        """
        X_ob, y_ob = _as_tensor(X_ob, self.device), _as_tensor(y_ob, self.device).flatten()
        if X_ob.ndim != 2 or X_ob.shape[0] != y_ob.shape[0] or X_ob.shape[0] < 1:
            raise ValueError(f"Need X_ob (n, F) and y_ob (n,), got {tuple(X_ob.shape)}, {tuple(y_ob.shape)}")
        self.X_ob, self.y_ob = X_ob, y_ob
        return self

    @torch.no_grad()
    def _logits(self, X_queries: torch.Tensor) -> torch.Tensor:
        """Logits for a batch of query tables sharing the stored context.

        Args:
            X_queries: Shape ``(batch, n, features)``.

        Returns:
            Logits of shape ``(batch, n, n_bins)`` on the normalized scale.
        """
        if self.X_ob is None or self.y_ob is None:
            raise RuntimeError("Call fit before predict.")
        b = X_queries.shape[0]
        context = self.X_ob.unsqueeze(0).expand(b, -1, -1)
        X = torch.cat([context, X_queries], dim=1)
        return self.model(X, self.y_ob.unsqueeze(0).expand(b, -1))

    def predict_distribution(self, X: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        """Predicted bar distribution on the original outcome scale (for plots and overlays).

        Args:
            X: Query rows, shape ``(n, features)``.

        Returns:
            Probabilities of shape ``(n, n_bins)`` and bin borders of shape ``(n, n_bins + 1)``
            (borders differ per dataset only by the outcome scaling).
        """
        logits = self._logits(_as_tensor(X, self.device).unsqueeze(0))[0]
        probs = torch.softmax(logits, dim=-1)
        mean, std = y_stats(self.y_ob.unsqueeze(0))
        borders = self.model.bar.borders * std[0] + mean[0]
        return probs.cpu().numpy(), borders.expand(probs.shape[0], -1).cpu().numpy()

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Predicted outcome under the treatment given in column 0 of ``X``.

        Args:
            X: Query rows, shape ``(n, features)``.

        Returns:
            Predicted means, shape ``(n,)``.
        """
        logits = self._logits(_as_tensor(X, self.device).unsqueeze(0))
        mean = denormalize_y(self.model.bar.mean(logits), self.y_ob.unsqueeze(0))
        return mean[0].cpu().numpy()

    def predict_cate(self, X: ArrayLike) -> np.ndarray:
        """Prediction with column 0 set to 1 minus prediction with column 0 set to 0.

        Args:
            X: Query rows, shape ``(n, features)``; column 0 is overwritten (on a copy).

        Returns:
            CATE estimates, shape ``(n,)``.
        """
        X = _as_tensor(X, self.device)
        X1, X0 = X.clone(), X.clone()
        X1[:, 0], X0[:, 0] = 1.0, 0.0
        means = denormalize_y(self.model.bar.mean(self._logits(torch.stack([X1, X0]))), self.y_ob.unsqueeze(0))
        return (means[0] - means[1]).cpu().numpy()
