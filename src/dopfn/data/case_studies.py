"""Do-PFN case-study datasets on disk.

Do-PFN ships its case studies as ``dill`` pickles of its own ``InterventionalDataset`` class
(``Do-PFN/data/prior_sampling/<Case_Study>/<Case_Study>_<i>.pkl``, 100 per case study). Loading
those needs the Do-PFN code on ``sys.path``, ``dill`` and ``pandas``, may break across Python
versions, and executes code stored in the file. We therefore convert them once
(``python scripts/generate_data.py case-studies ...``) into plain tensor files that
``load_data`` reads with ``torch.load(weights_only=True)``.

Converted layout: ``<root>/<Case_Study>/<Case_Study>_<i>.pt``, each a dict of float32 tensors
``x_obs (n, F)``, ``y_obs (n,)``, ``x_int (n, F)``, ``y_int (n,)``, ``cate (n,)``.

In each Do-PFN file, row ``k`` is one unit: ``x_obs[k], y_obs[k]`` its factual values and
``x_int[k], y_int[k]`` the values under the flipped treatment ``do(t = x_int[k, 0])`` with the same
noise, ``cate[k] = y(do(1)) - y(do(0))``. ``split_case_study`` follows Do-PFN's own evaluation
(``TabularDataset.generate_valid_split``): a random subset of units is the observational context,
the rest are queries with factual covariates and the intervened treatment.
"""

import os
import re
import sys
from pathlib import Path

import torch

from dopfn.data.dataset import CausalDataset

# Case-study name accepted by load_data -> directory name in the Do-PFN repo.
CASE_STUDY_DIRECTORIES: dict[str, str] = {
    "observed_confounder": "Observed_Confounder",
    "observed_mediator": "Observed_Mediator",
    "confounder_mediator": "Observed_Mediator_and_Confounder",
    "unobserved_confounder": "Unobserved_Confounder",
    "backdoor_criterion": "Backdoor_Criterion",
    "frontdoor_criterion": "Frontdoor_Criterion",
    "common_effect": "Common_Effect",
    "observed_confounder_small": "Observed_Confounder_Small",
}

# TODO(team): D9 - agree on the cluster path.
DEFAULT_CASE_STUDY_ROOT = Path("/home/space/datasets/dopfn_case_studies")
CASE_STUDY_ROOT_ENV = "DOPFN_CASE_STUDY_ROOT"

# Fraction of units used as observational context when neither n_train nor n_test is given.
# Do-PFN evaluates with 5-fold splits, i.e. 80 % context and 20 % queries.
DEFAULT_TRAIN_FRACTION = 0.8

RAW_KEYS = ("x_obs", "y_obs", "x_int", "y_int", "cate")

_DO_PFN_ROOT = Path(__file__).resolve().parents[3] / "Do-PFN"


def case_study_root(data_root: str | Path | None = None) -> Path:
    """Directory holding the converted case studies.

    Args:
        data_root: Explicit root. ``None``: ``$DOPFN_CASE_STUDY_ROOT`` or the cluster default.
    """
    if data_root is not None:
        return Path(data_root)
    return Path(os.environ.get(CASE_STUDY_ROOT_ENV, DEFAULT_CASE_STUDY_ROOT))


def _index(path: Path) -> int:
    """Trailing integer of ``<Name>_<i>.<ext>``, for natural sorting."""
    match = re.search(r"_(\d+)$", path.stem)
    return int(match.group(1)) if match else -1


def get_case_study_files(name: str, data_root: str | Path | None = None) -> list[Path]:
    """All converted files of one case study, sorted by their index (1, 2, ..., 100).

    Args:
        name: Key of ``CASE_STUDY_DIRECTORIES``.
        data_root: See ``case_study_root``.

    Raises:
        ValueError: Unknown case study.
        FileNotFoundError: Directory missing or empty.
    """
    if name not in CASE_STUDY_DIRECTORIES:
        raise ValueError(f"Unknown case study {name!r}. Choose one of {list(CASE_STUDY_DIRECTORIES)}.")
    directory = case_study_root(data_root) / CASE_STUDY_DIRECTORIES[name]
    files = sorted(directory.glob("*.pt"), key=lambda p: (_index(p), p.name))
    if not files:
        raise FileNotFoundError(
            f"No converted case-study files (*.pt) in {directory}. Convert them with "
            f"'python scripts/generate_data.py case-studies --source <Do-PFN>/data/prior_sampling "
            f"--out {case_study_root(data_root)}' or set ${CASE_STUDY_ROOT_ENV}."
        )
    return files


def load_case_study_file(path: str | Path) -> dict[str, torch.Tensor]:
    """Read one converted case-study file (no code execution).

    Args:
        path: ``.pt`` file written by ``save_case_study_file``.

    Returns:
        Dict with the keys ``RAW_KEYS``.
    """
    raw = torch.load(Path(path), map_location="cpu", weights_only=True)
    _check_raw(raw, str(path))
    return raw


def save_case_study_file(raw: dict[str, torch.Tensor], path: str | Path) -> None:
    """Write one converted case-study file."""
    _check_raw(raw, str(path))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({k: raw[k] for k in RAW_KEYS}, path)


def _check_raw(raw: dict[str, torch.Tensor], where: str) -> None:
    """Validate keys and shapes of a raw case-study dict."""
    missing = [k for k in RAW_KEYS if k not in raw]
    if missing:
        raise ValueError(f"{where}: missing keys {missing}")
    n, f = raw["x_obs"].shape
    if raw["x_int"].shape != (n, f):
        raise ValueError(f"{where}: x_int has shape {tuple(raw['x_int'].shape)}, expected {(n, f)}")
    for k in ("y_obs", "y_int", "cate"):
        if raw[k].shape != (n,):
            raise ValueError(f"{where}: {k} has shape {tuple(raw[k].shape)}, expected {(n,)}")


def split_case_study(
    raw: dict[str, torch.Tensor],
    generator: torch.Generator,
    n_train: int | None = None,
    n_test: int | None = None,
) -> CausalDataset:
    """Split the units of one case-study file into context and query rows.

    Args:
        raw: Dict from ``load_case_study_file``.
        generator: Torch generator for the random split.
        n_train: Context rows. ``None``: 80 % of the units, or all units not used as queries.
        n_test: Query rows. ``None``: all units not used as context.

    Returns:
        The dataset. Query covariates are the factual ``x_obs`` with column 0 replaced by the
        intervened treatment ``x_int[:, 0]`` (mediators keep their factual values, decision D5).

    Raises:
        ValueError: If ``n_train + n_test`` exceeds the number of units.
    """
    n = raw["x_obs"].shape[0]
    if n_train is None and n_test is None:
        n_train = int(round(DEFAULT_TRAIN_FRACTION * n))
    if n_train is None:
        n_train = n - n_test
    if n_test is None:
        n_test = n - n_train
    if n_train < 1 or n_test < 1 or n_train + n_test > n:
        raise ValueError(f"Cannot split {n} units into n_train={n_train} and n_test={n_test}.")

    perm = torch.randperm(n, generator=generator)
    ctx, qry = perm[:n_train], perm[n_train : n_train + n_test]
    X_in = raw["x_obs"][qry].clone()
    X_in[:, 0] = raw["x_int"][qry, 0]
    return CausalDataset(
        X_ob=raw["x_obs"][ctx].float(),
        y_ob=raw["y_obs"][ctx].float(),
        X_in=X_in.float(),
        y_in=raw["y_int"][qry].float(),
        y_in_ob=raw["y_obs"][qry].float(),
        cate=raw["cate"][qry].float(),
    )


def convert_do_pfn_pickle(
    pkl_path: str | Path, do_pfn_root: str | Path = _DO_PFN_ROOT
) -> dict[str, torch.Tensor]:
    """Read one original Do-PFN case-study pickle into a raw tensor dict.

    Only used by ``scripts/generate_data.py``. **Executes code stored in the pickle**: only use it
    on files from the Do-PFN repository. Needs ``pip install -e ".[case-studies]"``.

    Args:
        pkl_path: Path to ``<Case_Study>_<i>.pkl``.
        do_pfn_root: Directory with the Do-PFN code (its ``datasets`` and ``priors`` modules must
            be importable for unpickling).

    Returns:
        Dict with the keys ``RAW_KEYS`` (float32).
    """
    import dill  # optional dependency

    root = str(Path(do_pfn_root).resolve())
    added = root not in sys.path
    if added:
        sys.path.insert(0, root)
    try:
        with Path(pkl_path).open("rb") as f:
            ds = dill.load(f)
    finally:
        if added:
            sys.path.remove(root)
    raw = {k: torch.as_tensor(getattr(ds, k)).detach().float().cpu().contiguous() for k in RAW_KEYS}
    _check_raw(raw, str(pkl_path))
    return raw
