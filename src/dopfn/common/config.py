"""Config loading. Configs are YAML files in ``configs/`` mapped onto dataclasses."""

from pathlib import Path
from typing import Any


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file into a dict.

    Args:
        path: Path to the YAML file.

    Returns:
        The parsed config.
    """
    raise NotImplementedError
