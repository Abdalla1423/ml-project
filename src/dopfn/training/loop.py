"""Pre-training loop (Algorithm 1): sample prior datasets, NLL on interventional targets, Adam."""

from dataclasses import dataclass
from pathlib import Path

from dopfn.model.transformer import Model


@dataclass
class TrainConfig:
    """Training hyperparameters, filled from ``configs/train/*.yaml``."""

    steps: int = 10_000
    batch_size: int = 8
    lr: float = 1e-4
    val_every: int = 500
    n_val_datasets: int = 256
    checkpoint_every: int = 1_000
    out_dir: str = "outputs/train"
    seed: int = 0
    device: str = "cuda"


def train(model: Model, config: TrainConfig) -> dict[str, list[float]]:
    """Pre-train ``model`` on prior data until the loss converges.

    Logs training and validation loss and writes checkpoints to ``config.out_dir``.

    Args:
        model: Model to train (modified in place).
        config: Training config.

    Returns:
        Loss history, e.g. ``{"train": [...], "val": [...]}``.
    """
    raise NotImplementedError


def save_checkpoint(model: Model, path: str | Path, step: int) -> None:
    """Save model weights and step to ``path``."""
    raise NotImplementedError


def load_checkpoint(path: str | Path) -> Model:
    """Load a model from a checkpoint written by ``save_checkpoint``."""
    raise NotImplementedError
