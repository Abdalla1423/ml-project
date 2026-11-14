"""Data loading entry point required by the ms1 sheet."""

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import torch

from dopfn.data.case_studies import (
    CASE_STUDY_DIRECTORIES,
    get_case_study_files,
    load_case_study_file,
    split_case_study,
)
from dopfn.data.dataset import CausalDataset
from dopfn.data.prior import PriorConfig, derive_seed, iter_prior_samples, load_prior_config

# Case-study names accepted by ``load_data(dataset=...)``, as in the Do-PFN repo
# (see dopfn.data.case_studies.CASE_STUDY_DIRECTORIES).
CASE_STUDIES: tuple[str, ...] = tuple(CASE_STUDY_DIRECTORIES)


def load_data(
    dataset: str = "prior",
    transformation: Callable[[CausalDataset], Any] | None = None,
    n_train: int | None = None,
    n_test: int | None = None,
    seed: int = 0,
    n_datasets: int | None = None,
    config: PriorConfig | str | Path | None = None,
    data_root: str | Path | None = None,
) -> Iterator[Any]:
    """Return a lazy iterator of datasets. One item is one whole dataset.

    Arguments are validated immediately; datasets are only produced while iterating.

    Args:
        dataset: ``'prior'`` samples a fresh SCM per item, with observational data and
            interventional query points from the same SCM. Any name in ``CASE_STUDIES`` reads the
            converted case study from disk (see ``dopfn.data.case_studies``).
        transformation: Applied to every dataset before it is yielded.
        n_train: Observational rows per dataset (``None``: sampled for the prior, 80 % of the
            units for case studies).
        n_test: Query rows per dataset (``None``: the remaining rows).
        seed: Prior: base seed of the stream (disjoint streams for different seeds). Case studies:
            seed of the random context/query split.
        n_datasets: Maximum number of datasets (``None``: infinite for the prior, all files for a
            case study).
        config: Prior config, or a path to a YAML like ``configs/data/prior.yaml``. Ignored for
            case studies.
        data_root: Root of the converted case studies (default ``$DOPFN_CASE_STUDY_ROOT`` or
            ``/home/space/datasets/dopfn_case_studies``). Ignored for the prior.

    Returns:
        Iterator over ``CausalDataset(X_ob, y_ob, X_in, y_in, y_in_ob, cate)`` (or whatever
        ``transformation`` returns). The treatment is column 0 of ``X_ob`` and ``X_in``.

    Raises:
        ValueError: Unknown dataset or invalid sizes.
        FileNotFoundError: Case study not found on disk.
    """
    for name, value in (("n_train", n_train), ("n_test", n_test), ("n_datasets", n_datasets)):
        if value is not None and value < 1:
            raise ValueError(f"{name} must be positive, got {value}")
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")

    if dataset == "prior":
        if isinstance(config, (str, Path)):
            config = load_prior_config(config)
        samples: Iterator[CausalDataset] = (
            s.data
            for s in iter_prior_samples(
                seed=seed, config=config, n_train=n_train, n_test=n_test, n_datasets=n_datasets
            )
        )
    elif dataset in CASE_STUDIES:
        files = get_case_study_files(dataset, data_root)[:n_datasets]
        samples = _iter_case_study(files, seed=seed, n_train=n_train, n_test=n_test)
    else:
        raise ValueError(f"Unknown dataset {dataset!r}. Choose 'prior' or one of {CASE_STUDIES}.")

    return _apply(samples, transformation)


def _iter_case_study(
    files: list[Path], seed: int, n_train: int | None, n_test: int | None
) -> Iterator[CausalDataset]:
    """Lazily read and split case-study files. File ``i`` is split with ``derive_seed(seed, i)``."""
    for i, path in enumerate(files):
        generator = torch.Generator().manual_seed(derive_seed(seed, i))
        yield split_case_study(load_case_study_file(path), generator, n_train=n_train, n_test=n_test)


def _apply(
    samples: Iterator[CausalDataset], transformation: Callable[[CausalDataset], Any] | None
) -> Iterator[Any]:
    """Apply ``transformation`` lazily to every dataset."""
    for sample in samples:
        yield sample if transformation is None else transformation(sample)

