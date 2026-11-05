"""``show``: causal graph next to observational and interventional data."""

from pathlib import Path
from typing import Any


def show(x: list[Any], outfile: str | Path | None = None, **kwargs: Any) -> Any:
    """Render a list of data points as one image.

    Args:
        x: List of datasets as yielded by ``dopfn.data.load_data``.
        outfile: If given, save the image to this path.
        **kwargs: Extensions, e.g. overlays of model predictions.

    Returns:
        The matplotlib figure.
    """
    raise NotImplementedError
