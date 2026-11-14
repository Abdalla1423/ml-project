"""Config loading. Configs are YAML files in ``configs/`` mapped onto dataclasses."""

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file into a dict.

    Args:
        path: Path to the YAML file.

    Returns:
        The parsed config (an empty dict for an empty file).

    Raises:
        ValueError: If the top level of the file is not a mapping.
    """
    with Path(path).open() as f:
        config = yaml.safe_load(f)
    if config is None:
        return {}
    if not isinstance(config, dict):
        raise ValueError(f"Top level of {path} must be a mapping, got {type(config).__name__}")
    return config
