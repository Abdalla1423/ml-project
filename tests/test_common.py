"""Tests for dopfn.common (shared) and repo-wide rules."""

import random
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from dopfn.common import load_yaml, set_seed

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_set_seed_makes_all_rngs_reproducible() -> None:
    set_seed(123)
    a = (random.random(), np.random.rand(), torch.rand(1).item())
    set_seed(123)
    b = (random.random(), np.random.rand(), torch.rand(1).item())
    assert a == b


def test_load_yaml(tmp_path: Path) -> None:
    assert load_yaml(REPO_ROOT / "configs/data/prior.yaml")["seed"] == 0
    (tmp_path / "empty.yaml").write_text("")
    assert load_yaml(tmp_path / "empty.yaml") == {}
    (tmp_path / "list.yaml").write_text("- 1\n- 2\n")
    with pytest.raises(ValueError):
        load_yaml(tmp_path / "list.yaml")


def test_no_notebooks_in_repo() -> None:  # req C1
    out = subprocess.run(
        ["git", "ls-files", "*.ipynb"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == ""
