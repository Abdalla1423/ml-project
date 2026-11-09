"""Load Do-PFN case-study datasets from disk."""

import os
import pickle
from pathlib import Path



CASE_STUDY_DIRECTORIES = {
    "observed_confounder": "Observed_Confounder",
    "observed_mediator": "Observed_Mediator",
    "confounder_mediator": "Observed_Mediator_and_Confounder",
    "unobserved_confounder": "Unobserved_Confounder",
    "backdoor_criterion": "Backdoor_Criterion",
    "frontdoor_criterion": "Frontdoor_Criterion",
    "common_effect": "Common_Effect",
    "observed_confounder_small": "Observed_Confounder_Small",
}


def get_case_study_files(
    dataset: str,
    data_root: str | Path | None = None,
) -> list[Path]:
    """Find all pickle files for a case study."""

    if dataset not in CASE_STUDY_DIRECTORIES:
        raise ValueError(f"Unknown case study: {dataset!r}")

    if data_root is None:
        data_root = os.environ.get(
            "DOPFN_DATA_ROOT",
            "/home/space/datasets",
        )

    dataset_dir = Path(data_root) / CASE_STUDY_DIRECTORIES[dataset]

    if not dataset_dir.is_dir():
        raise FileNotFoundError(
            f"Dataset directory does not exist: {dataset_dir}"
        )

    files = sorted(dataset_dir.glob("*.pkl"))

    if not files:
        raise FileNotFoundError(
            f"No pickle files found in: {dataset_dir}"
        )

    return files


def load_case_study(file_path: str | Path):
    """Load one case-study pickle file."""

    with Path(file_path).open("rb") as f:
        data = pickle.load(f)

    Xob = data.x_obs
    yob = data.y_obs
    Xin = data.x_int
    yin = data.y_int
    y_in_ob = data.y_obs
    cate = data.cate

    return Xob, yob, Xin, yin, y_in_ob, cate
