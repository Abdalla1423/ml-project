"""Scikit-learn-like wrapper around a trained ``Model`` (like ``DoPFNRegressor``)."""

import numpy as np

from dopfn.model.transformer import Model


class Regressor:
    """In-context regressor: ``fit`` stores data, ``predict`` runs the model."""

    def __init__(self, model: Model, device: str = "cpu") -> None:
        """Args:
            model: A trained ``Model``.
            device: Device for inference.
        """
        raise NotImplementedError

    def fit(self, X_ob: np.ndarray, y_ob: np.ndarray) -> "Regressor":
        """Only store the observational data (no training).

        Args:
            X_ob: Observational covariates, treatment in column 0, shape ``(n_ob, features)``.
            y_ob: Observational outcomes, shape ``(n_ob,)``.

        Returns:
            ``self``.
        """
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicted outcome under the treatment given in column 0 of ``X``.

        Args:
            X: Query rows, shape ``(n, features)``.

        Returns:
            Predicted means, shape ``(n,)``.
        """
        raise NotImplementedError

    def predict_cate(self, X: np.ndarray) -> np.ndarray:
        """Prediction with column 0 set to 1 minus prediction with column 0 set to 0.

        Args:
            X: Query rows, shape ``(n, features)``; column 0 is overwritten.

        Returns:
            CATE estimates, shape ``(n,)``.
        """
        raise NotImplementedError
