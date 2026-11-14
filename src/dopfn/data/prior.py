"""SCM prior with a designated binary treatment and do(t) interventions.

Extends Do-PFN-prior (https://github.com/oossen/Do-PFN-prior, copied to
``Do-PFN-prior/``). Do-PFN-prior samples a random DAG and SCM with target ``y`` but no
treatment and only observational data. On top of it we

1. add a binary treatment node ``t`` to the sampled graph (``graph_extension``),
2. build the SCM on the extended graph with Do-PFN-prior's ``SCMBuilder``,
3. sample observational rows, and interventional query rows following Algorithm 1 of the paper:
   factual covariates, a random treatment ``t_in``, and ``y_in`` under ``do(t_in)`` with the same
   noise (``intervention``),
4. compute the ground-truth effect ``y(do(1)) - y(do(0))`` of each query row with its own noise.

Every dataset is fully determined by one integer seed.
"""

import copy
from collections.abc import Iterator
from dataclasses import dataclass, field
from itertools import count
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import torch
from dopfnprior.causal_graph.graph_builder import GraphBuilder
from dopfnprior.configs.default_config import prior_config as DOPFNPRIOR_DEFAULT_CONFIG
from dopfnprior.scm.scm import SCM
from dopfnprior.scm.scm_builder import SCMBuilder
from dopfnprior.utils.sampling import build_samplers, sample_parameters

from dopfn.common.config import load_yaml
from dopfn.data.dataset import CausalDataset
from dopfn.data.graph_extension import OUTCOME, TREATMENT, add_treatment_node, get_input_variables
from dopfn.data.intervention import propagate, sample_exogenous


def derive_seed(*keys: int) -> int:
    """Derive an independent 63-bit seed from a tuple of non-negative integers.

    Uses NumPy's ``SeedSequence``, so ``derive_seed(0, 1)`` and ``derive_seed(1, 0)`` give
    unrelated streams. Prior streams with different base seeds therefore never share datasets
    (unlike ``seed + i``), see decision D6.

    Args:
        keys: Non-negative integers, e.g. ``(base_seed, dataset_index)``.

    Returns:
        A seed usable with ``torch.Generator.manual_seed``.
    """
    state = np.random.SeedSequence(list(keys)).generate_state(1, dtype=np.uint64)[0]
    return int(state >> np.uint64(1))


@dataclass
class PriorConfig:
    """Prior hyperparameters. Defaults equal ``configs/data/prior.yaml``.

    Attributes:
        graph: Overrides of Do-PFN-prior's ``graph_config`` (same format).
        scm: Overrides of Do-PFN-prior's ``scm_config`` (same format).
        m_min: Minimum number of observational rows when ``n_train`` is not given.
        m_max: Total rows per dataset (observational + query) when sizes are not given.
        min_test: Minimum number of query rows when sizes are sampled.
        max_treatment_children: Maximum number of children of ``t``.
        treated_fraction: Accepted range of the observational treated fraction; SCMs outside are
            resampled. Off by default (``(0, 1)``); e.g. ``(0.05, 0.95)`` drops datasets without
            treated or untreated units.
        max_attempts: Resampling attempts per dataset.
    """

    graph: dict[str, Any] = field(default_factory=dict)
    scm: dict[str, Any] = field(default_factory=dict)
    m_min: int = 10
    m_max: int = 2200
    min_test: int = 10
    max_treatment_children: int = 3
    treated_fraction: tuple[float, float] = (0.0, 1.0)
    max_attempts: int = 100

    def __post_init__(self) -> None:
        self.treated_fraction = tuple(self.treated_fraction)
        if not 1 <= self.m_min <= self.m_max - self.min_test:
            raise ValueError("Need 1 <= m_min <= m_max - min_test.")
        if self.min_test < 1:
            raise ValueError("min_test must be at least 1.")
        low, high = self.treated_fraction
        if not 0.0 <= low <= high <= 1.0:
            raise ValueError("treated_fraction must be a range within [0, 1].")

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "PriorConfig":
        """Build from the nested layout of ``configs/data/prior.yaml``.

        Args:
            config: Parsed YAML. The top-level ``seed`` key is ignored here.

        Returns:
            The config.

        Raises:
            ValueError: On unknown keys.
        """
        config = dict(config)
        config.pop("seed", None)
        dataset = dict(config.pop("dataset", {}) or {})
        treatment = dict(config.pop("treatment", {}) or {})
        kwargs: dict[str, Any] = {
            "graph": config.pop("graph", {}) or {},
            "scm": config.pop("scm", {}) or {},
        }
        for key in ("m_min", "m_max", "min_test"):
            if key in dataset:
                kwargs[key] = dataset.pop(key)
        if "max_children" in treatment:
            kwargs["max_treatment_children"] = treatment.pop("max_children")
        if "treated_fraction" in treatment:
            kwargs["treated_fraction"] = treatment.pop("treated_fraction")
        if "max_attempts" in config:
            kwargs["max_attempts"] = config.pop("max_attempts")
        unknown = list(config) + [f"dataset.{k}" for k in dataset] + [f"treatment.{k}" for k in treatment]
        if unknown:
            raise ValueError(f"Unknown prior config keys: {unknown}")
        return cls(**kwargs)

    def dopfnprior_config(self) -> dict[str, Any]:
        """Do-PFN-prior's default config with the ``graph`` and ``scm`` overrides applied."""
        merged = copy.deepcopy(DOPFNPRIOR_DEFAULT_CONFIG)
        for section, overrides in (("graph_config", self.graph), ("scm_config", self.scm)):
            unknown = set(overrides) - set(merged[section])
            if unknown:
                raise ValueError(f"Unknown Do-PFN-prior {section} keys: {sorted(unknown)}")
            merged[section].update(copy.deepcopy(overrides))
        return merged


def load_prior_config(path: str | Path) -> PriorConfig:
    """Load a prior config YAML (e.g. ``configs/data/prior.yaml``)."""
    return PriorConfig.from_dict(load_yaml(path))


class PriorSampler:
    """Samples datasets from the prior. Builds the Do-PFN-prior samplers once.

    Args:
        config: Prior config. ``None`` uses the defaults.
    """

    def __init__(self, config: PriorConfig | None = None) -> None:
        self.config = config if config is not None else PriorConfig()
        dopfnprior_config = self.config.dopfnprior_config()
        graph_samplers = build_samplers(dopfnprior_config["graph_config"], "graph")
        self._features_per_node = graph_samplers.pop("features_per_node")
        self._graph_samplers = graph_samplers
        self._scm_samplers = build_samplers(dopfnprior_config["scm_config"], "scm")

    def sample_scm(self, generator: torch.Generator) -> tuple[SCM, nx.DiGraph]:
        """Sample a DAG from Do-PFN-prior, add the treatment node and build the SCM.

        Args:
            generator: Torch generator, the only source of randomness.

        Returns:
            The SCM and its extended graph.
        """
        graph_params = sample_parameters(self._graph_samplers, generator)
        graph = GraphBuilder(**graph_params, features_per_node_dist=self._features_per_node).sample(
            generator
        )
        graph = add_treatment_node(graph, generator, self.config.max_treatment_children)
        scm = SCMBuilder(graph, **self._scm_samplers).sample(generator)
        return scm, graph

    def _sizes(
        self, n_train: int | None, n_test: int | None, generator: torch.Generator
    ) -> tuple[int, int]:
        """Resolve the numbers of observational and query rows."""
        c = self.config
        if n_train is None:
            high = max(c.m_min, c.m_max - (n_test if n_test is not None else c.min_test))
            n_train = int(torch.randint(c.m_min, high + 1, (1,), generator=generator).item())
        if n_test is None:
            n_test = max(c.m_max - n_train, c.min_test)
        if n_train < 1 or n_test < 1:
            raise ValueError(f"n_train and n_test must be positive, got {n_train}, {n_test}")
        return n_train, n_test

    def sample(
        self, seed: int, n_train: int | None = None, n_test: int | None = None
    ) -> "PriorSample":
        """Sample one dataset. The result is fully determined by the arguments.

        Args:
            seed: Dataset seed (non-negative).
            n_train: Observational rows. ``None``: sampled as in Algorithm 1.
            n_test: Query rows. ``None``: ``m_max - n_train`` (at least ``min_test``).

        Returns:
            The dataset plus the SCM and graph it came from.

        Raises:
            RuntimeError: If no valid dataset is found within ``max_attempts``.
        """
        low, high = self.config.treated_fraction
        for attempt in range(self.config.max_attempts):
            generator = torch.Generator().manual_seed(derive_seed(seed, attempt))
            scm, graph = self.sample_scm(generator)
            input_vars = get_input_variables(graph)
            n_ob, n_in = self._sizes(n_train, n_test, generator)

            # Observational context.
            observational = propagate(scm, sample_exogenous(scm, n_ob, generator))
            X_ob, y_ob = build_features(observational, graph, input_vars)
            treated = X_ob[:, 0].mean().item()
            if not low <= treated <= high:
                continue

            # Interventional queries: factual covariates, y under do(t_in) with the same noise.
            exogenous = sample_exogenous(scm, n_in, generator)
            t_in = torch.randint(0, 2, (n_in,), generator=generator).to(X_ob.dtype)
            X_in, y_in_ob = build_features(propagate(scm, exogenous), graph, input_vars)
            X_in[:, 0] = t_in
            _, y0 = build_features(propagate(scm, exogenous, do_t=0.0), graph, input_vars)
            _, y1 = build_features(propagate(scm, exogenous, do_t=1.0), graph, input_vars)
            y_in = torch.where(t_in == 1, y1, y0)

            data = CausalDataset(
                X_ob=X_ob.float(),
                y_ob=y_ob.float(),
                X_in=X_in.float(),
                y_in=y_in.float(),
                y_in_ob=y_in_ob.float(),
                cate=(y1 - y0).float(),
            )
            if not all(torch.isfinite(v).all() for v in data):
                continue
            return PriorSample(data=data, scm=scm, graph=graph, input_vars=input_vars, seed=seed)
        raise RuntimeError(
            f"No valid prior dataset for seed {seed} after {self.config.max_attempts} attempts."
        )


@dataclass
class PriorSample:
    """A prior dataset together with where it came from (for visualization and tests).

    Attributes:
        data: The dataset.
        scm: The SCM (on the extended graph).
        graph: The extended causal graph.
        input_vars: Covariate nodes in column order (``t`` first in X).
        seed: Seed that reproduces this sample with ``PriorSampler.sample``.
    """

    data: CausalDataset
    scm: SCM
    graph: nx.DiGraph
    input_vars: list[str]
    seed: int


def build_features(
    values: dict[str, torch.Tensor], graph: nx.DiGraph, input_vars: list[str]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Assemble ``X`` (treatment in column 0) and ``y`` from node values.

    Columns of ``X``: ``t``, then the visible features of ``input_vars`` (except ``t``) in order,
    then the remaining visible features of the ``y`` node (as in Do-PFN-prior). ``y`` is the first
    visible feature of the ``y`` node.

    Args:
        values: Node values from ``propagate``, each of shape ``(n, dimension_v)``.
        graph: Graph with ``n_visible`` node attributes.
        input_vars: Covariate nodes from ``get_input_variables``.

    Returns:
        ``X`` of shape ``(n, F)`` and ``y`` of shape ``(n,)``.
    """
    visible = {v: values[v][..., : graph.nodes[v]["n_visible"]] for v in values}
    features = [visible[TREATMENT]] + [visible[v] for v in input_vars if v != TREATMENT]
    y_visible = visible[OUTCOME]
    features.append(y_visible[..., 1:])
    return torch.cat(features, dim=-1), y_visible[..., 0]


def iter_prior_samples(
    seed: int = 0,
    config: PriorConfig | None = None,
    n_train: int | None = None,
    n_test: int | None = None,
    n_datasets: int | None = None,
) -> Iterator[PriorSample]:
    """Lazily sample prior datasets. Dataset ``i`` uses seed ``derive_seed(seed, i)``.

    Args:
        seed: Base seed of the stream (non-negative). Different base seeds give disjoint streams.
        config: Prior config. ``None`` uses the defaults.
        n_train: Observational rows per dataset (``None``: sampled).
        n_test: Query rows per dataset (``None``: ``m_max - n_train``).
        n_datasets: Number of datasets. ``None``: infinite stream.

    Yields:
        One ``PriorSample`` per dataset.
    """
    sampler = PriorSampler(config)
    indices = count() if n_datasets is None else range(n_datasets)
    for i in indices:
        yield sampler.sample(derive_seed(seed, i), n_train=n_train, n_test=n_test)
