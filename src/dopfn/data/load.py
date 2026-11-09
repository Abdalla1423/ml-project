"""Data loading entry point required by the ms1 sheet."""

from collections.abc import Callable, Iterable
from typing import Any
from .prior import generate_prior_samples
from .case_studies import get_case_study_files, load_case_study

# Case-study names accepted by ``load_data(dataset=...)``.
# Sheet example: 'observed_confounder'. TODO(team): D1 - confirm names against the Do-PFN repo.
CASE_STUDIES: tuple[str, ...] = (
    "observed_confounder",
    "observed_mediator",
    "confounder_mediator",
    "unobserved_confounder",
    "backdoor_criterion",
    "frontdoor_criterion",
    "common_effect",
    "observed_confounder_small",
)


def load_data(
    dataset: str = "prior",
    transformation: Callable[[Any], Any] | None = None,
    n_train: int | None = None,
    n_test: int | None = None,
    seed: int | None = None,
    **kwargs: Any,
) -> Iterable[Any]:
    """Return a lazy iterable of datasets. One sample is one whole dataset.

    Args:
        dataset: ``'prior'`` samples a fresh SCM per item, with observational data and
            interventional query points from the same SCM. Any name in ``CASE_STUDIES`` reads
            that case study from disk (``/home/space/datasets/<dataset>`` on the cluster).
        transformation: Applied to every dataset before it is yielded.
        n_train: Number of observational rows per dataset (``None``: sampled / as on disk).
        n_test: Number of interventional query rows per dataset (``None``: sampled / as on disk).
        seed: Seed for the prior sampler.
        **kwargs: Further options (e.g. prior config).

    Yields:
        Per dataset ``(X_ob, y_ob, X_in, y_in, y_in_ob, cate)`` with the treatment in column 0
        of ``X_ob`` and ``X_in``. ``y_in_ob`` are the observational outcomes of the query points
        (for Dont-PFN), ``cate`` the ground-truth CATE of the query points.
        TODO(team): D1 - exact tuple format, dtypes and shapes.
    """
    if dataset == "prior":
        samples = generate_prior_samples(n_train=n_train, n_test=n_test, seed=seed)
        
    elif dataset in CASE_STUDIES:
        data_root = kwargs.pop("data_root", None)
        files = get_case_study_files(dataset=dataset, data_root=data_root)
        samples = (load_case_study(file_path) for file_path in files)

    else:
        raise ValueError(
            f"Unknown dataset: {dataset!r}. "
            f"Choose 'prior' or one of {CASE_STUDIES}."
        )

    for sample in samples:
        if transformation is not None:
            sample = transformation(sample)

        yield sample

