"""Tests for dopfn.data (owner: A). Requirement IDs refer to docs/milestones/ms1/requirements.md."""

import hashlib
import inspect
import os
import subprocess
import sys
from collections.abc import Iterator
from itertools import islice
from pathlib import Path

import networkx as nx
import pytest
import torch
import torch.distributions as dist
from dopfnprior.scm.scm import SCM
from dopfnprior.utils.sampling import FixedSampler, TorchDistributionSampler
from torch import nn

from dopfn.data import CASE_STUDIES, CausalDataset, PriorConfig, load_data, load_prior_config
from dopfn.data.case_studies import (
    CASE_STUDY_DIRECTORIES,
    case_study_root,
    convert_do_pfn_pickle,
    get_case_study_files,
    save_case_study_file,
    split_case_study,
)
from dopfn.data.graph_extension import (
    OUTCOME,
    TREATMENT,
    add_treatment_node,
    get_input_variables,
    get_pre_treatment_variables,
)
from dopfn.data.intervention import propagate, sample_exogenous
from dopfn.data.prior import PriorSampler, derive_seed, iter_prior_samples

REPO_ROOT = Path(__file__).resolve().parents[1]
SMALL = {"n_train": 40, "n_test": 30}


def _digest(data: CausalDataset) -> str:
    """Hash of all tensors of a dataset."""
    h = hashlib.sha256()
    for t in data:
        h.update(t.numpy().tobytes())
    return h.hexdigest()


# --- D1-D5: interface ---------------------------------------------------------------------------


def test_load_data_signature() -> None:  # req D1
    params = inspect.signature(load_data).parameters
    assert list(params)[:4] == ["dataset", "transformation", "n_train", "n_test"]
    assert params["dataset"].default == "prior"
    for name in ("transformation", "n_train", "n_test"):
        assert params[name].default is None


def test_load_data_is_lazy() -> None:  # req D3
    it = load_data("prior", seed=0)  # infinite stream: must not materialize anything
    assert isinstance(it, Iterator)
    assert not isinstance(it, (list, tuple))
    assert len(list(islice(it, 3))) == 3


def test_prior_yields_whole_valid_datasets() -> None:  # req D2, D4, D5
    for data in load_data("prior", seed=0, n_datasets=10, **SMALL):
        assert isinstance(data, CausalDataset)
        data.validate()  # shapes, float32, finite, binary treatment column
        X_ob, y_ob, X_in, y_in, y_in_ob, cate = data  # plain tuple unpacking works
        assert X_ob.shape == (40, X_in.shape[1]) and y_ob.shape == (40,)
        assert X_in.shape[0] == y_in.shape[0] == y_in_ob.shape[0] == cate.shape[0] == 30


def test_treated_fraction_filter() -> None:
    config = PriorConfig(treated_fraction=(0.2, 0.8))
    for data in load_data("prior", seed=3, n_datasets=10, n_train=200, n_test=10, config=config):
        assert 0.2 <= data.X_ob[:, 0].mean().item() <= 0.8


def test_unknown_dataset_and_bad_sizes_raise_immediately() -> None:
    with pytest.raises(ValueError):
        load_data("does_not_exist")
    with pytest.raises(ValueError):
        load_data("prior", n_train=0)
    with pytest.raises(ValueError):
        load_data("prior", seed=-1)


# --- D6: seeding and reproducibility --------------------------------------------------------------


def test_prior_is_reproducible_and_datasets_differ() -> None:  # req D6
    a = [_digest(d) for d in load_data("prior", seed=0, n_datasets=5, **SMALL)]
    b = [_digest(d) for d in load_data("prior", seed=0, n_datasets=5, **SMALL)]
    c = [_digest(d) for d in load_data("prior", seed=1, n_datasets=5, **SMALL)]
    assert a == b
    assert len(set(a)) == 5, "every dataset in a stream must come from a different SCM"
    assert not set(a) & set(c), "streams with different seeds must not share datasets"


def test_prior_sample_reproducible_from_its_seed() -> None:  # req D6
    sample = next(iter_prior_samples(seed=7, n_datasets=1, **SMALL))
    again = PriorSampler().sample(sample.seed, **SMALL)
    assert _digest(sample.data) == _digest(again.data)


def test_prior_independent_of_python_hash_seed() -> None:  # req D6
    """Set iteration order of strings changes with PYTHONHASHSEED; the prior must not depend on it."""
    code = (
        "import hashlib\n"
        "from dopfn.data import load_data\n"
        "h = hashlib.sha256()\n"
        "for d in load_data('prior', seed=0, n_datasets=5, n_train=20, n_test=5):\n"
        "    for t in d: h.update(t.numpy().tobytes())\n"
        "print(h.hexdigest())\n"
    )
    digests = set()
    for hash_seed in ("1", "2"):
        env = {**os.environ, "PYTHONHASHSEED": hash_seed}
        out = subprocess.run(
            [sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True
        )
        digests.add(out.stdout.strip())
    assert len(digests) == 1


def test_derive_seed_streams_do_not_overlap() -> None:
    seeds = {derive_seed(s, i) for s in range(5) for i in range(200)}
    assert len(seeds) == 1000


def test_prior_queries_consistent_with_scm() -> None:  # req D6, D7
    """Re-propagating the SCM reproduces y_in, y_in_ob and cate of every query row."""
    sample = next(iter_prior_samples(seed=11, n_datasets=1, **SMALL))
    data = sample.data
    # Where the intervened treatment equals the factual one, do(t) changes nothing.
    # The factual outcome follows from the intervened one and the effect:
    # y_in_ob = y(do(t_fact)) and cate = y(do(1)) - y(do(0)).
    t_in = data.X_in[:, 0]
    y0 = torch.where(t_in == 0, data.y_in, data.y_in - data.cate)
    y1 = y0 + data.cate
    assert torch.all(torch.isclose(data.y_in_ob, y0, atol=1e-5) | torch.isclose(data.y_in_ob, y1, atol=1e-5))


# --- D7: hand-built SCM with a known effect -------------------------------------------------------


class LinearMechanism(nn.Module):
    """Deterministic ``sum_p w_p * x_p`` over the parents."""

    def __init__(self, weights: dict[str, float]) -> None:
        super().__init__()
        self.weights = weights

    def forward(self, parents: dict[str, torch.Tensor]) -> torch.Tensor:
        return sum(self.weights[p] * v for p, v in parents.items())


def _hand_built_scm(with_mediator: bool) -> tuple[SCM, nx.DiGraph]:
    """x -> t, x -> y, and t -> y (effect 2) or t -> m -> y (effect 3, noise-free m and y)."""
    g = nx.DiGraph()
    nodes = ["x", TREATMENT, "m", OUTCOME] if with_mediator else ["x", TREATMENT, OUTCOME]
    for v in nodes:
        g.add_node(v, dimension=1, n_hidden=0, n_visible=1)
    g.add_edges_from([("x", TREATMENT), ("x", OUTCOME)])
    normal = TorchDistributionSampler(dist.Normal(0.0, 1.0))
    noise = {"x": normal, TREATMENT: normal, OUTCOME: FixedSampler(0.0)}
    mechanisms = {TREATMENT: LinearMechanism({"x": 2.0})}
    if with_mediator:
        g.add_edges_from([(TREATMENT, "m"), ("m", OUTCOME)])
        noise["m"] = FixedSampler(0.0)
        mechanisms["m"] = LinearMechanism({TREATMENT: 3.0})
        mechanisms[OUTCOME] = LinearMechanism({"m": 1.0, "x": 1.0})
    else:
        g.add_edge(TREATMENT, OUTCOME)
        mechanisms[OUTCOME] = LinearMechanism({TREATMENT: 2.0, "x": 1.0})
    scm = SCM(g, mechanisms, noise, post_activations={}, device=torch.device("cpu"))
    return scm, g


@pytest.fixture
def hand_built_prior(monkeypatch: pytest.MonkeyPatch):
    """PriorSampler whose sample_scm returns a hand-built SCM."""

    def make(with_mediator: bool) -> PriorSampler:
        scm, graph = _hand_built_scm(with_mediator)
        monkeypatch.setattr(PriorSampler, "sample_scm", lambda self, generator: (scm, graph))
        return PriorSampler(PriorConfig(treated_fraction=(0.0, 1.0)))

    return make


def test_known_effect_with_confounder(hand_built_prior) -> None:  # req D7
    data = hand_built_prior(with_mediator=False).sample(seed=0, n_train=2000, n_test=500).data
    data.validate()
    torch.testing.assert_close(data.cate, torch.full_like(data.cate, 2.0))
    # X = [t, x]; y_in = 2 * t_in + x exactly (no outcome noise).
    torch.testing.assert_close(data.y_in, 2.0 * data.X_in[:, 0] + data.X_in[:, 1])
    # The confounder x drives the observational treatment: treated units have larger x.
    t, x = data.X_ob[:, 0], data.X_ob[:, 1]
    assert x[t == 1].mean() > x[t == 0].mean() + 0.5


def test_mediator_keeps_factual_value_in_queries(hand_built_prior) -> None:  # req D5, D7
    data = hand_built_prior(with_mediator=True).sample(seed=0, n_train=500, n_test=500).data
    t_in, x_in, m_in = data.X_in[:, 0], data.X_in[:, 1], data.X_in[:, 2]  # X = [t, x, m]
    torch.testing.assert_close(data.cate, torch.full_like(data.cate, 3.0))
    # y under do(t_in) uses the recomputed mediator 3 * t_in.
    torch.testing.assert_close(data.y_in, 3.0 * t_in + x_in)
    # The covariate m is the factual value 3 * t_fact, so y_in_ob = m + x.
    torch.testing.assert_close(data.y_in_ob, m_in + x_in)
    flipped = m_in != 3.0 * t_in
    assert flipped.float().mean() > 0.2, "some queries must have t_in != t_fact"


def test_propagate_do_cuts_incoming_edges() -> None:  # req D7
    scm, _ = _hand_built_scm(with_mediator=False)
    ex = sample_exogenous(scm, 100, torch.Generator().manual_seed(0))
    for value in (0.0, 1.0):
        assert torch.all(propagate(scm, ex, do_t=value)[TREATMENT] == value)
    obs = propagate(scm, ex)
    assert set(obs[TREATMENT].unique().tolist()) <= {0.0, 1.0}
    torch.testing.assert_close(propagate(scm, ex)["y"], obs["y"])  # deterministic given the noise


# --- graph extension --------------------------------------------------------------------------------


def test_add_treatment_node() -> None:
    sampler = PriorSampler()
    for seed in range(30):
        _, graph = sampler.sample_scm(torch.Generator().manual_seed(seed))
        assert nx.is_directed_acyclic_graph(graph)
        assert nx.has_path(graph, TREATMENT, OUTCOME)
        children = list(graph.successors(TREATMENT))
        assert 1 <= len(children) <= PriorConfig().max_treatment_children
        for parent in graph.predecessors(TREATMENT):
            assert all(nx.has_path(graph, parent, c) for c in children)
        inputs = get_input_variables(graph)
        assert TREATMENT in inputs and OUTCOME not in inputs
        assert TREATMENT not in get_pre_treatment_variables(graph)


def test_add_treatment_node_rejects_bad_graphs_and_copies() -> None:
    g = nx.DiGraph([("a", OUTCOME)])
    nx.set_node_attributes(g, 1, "dimension")
    extended = add_treatment_node(g, torch.Generator().manual_seed(0))
    assert TREATMENT not in g and TREATMENT in extended
    with pytest.raises(ValueError):
        add_treatment_node(extended, torch.Generator())
    with pytest.raises(ValueError):
        add_treatment_node(nx.DiGraph([("a", "b")]), torch.Generator())


# --- D9, D10: transformation and sizes --------------------------------------------------------------


def test_transformation_applied_to_every_dataset() -> None:  # req D9
    calls = []

    def transformation(data: CausalDataset) -> tuple[str, int]:
        calls.append(1)
        return ("transformed", data.n_train)

    out = list(load_data("prior", transformation=transformation, seed=0, n_datasets=4, **SMALL))
    assert out == [("transformed", 40)] * 4 and len(calls) == 4


def test_sizes_respected() -> None:  # req D10
    datasets = list(load_data("prior", seed=0, n_datasets=3, n_train=17, n_test=9))
    assert len(datasets) == 3
    assert all(d.n_train == 17 and d.n_test == 9 for d in datasets)


def test_sampled_sizes_follow_algorithm_1() -> None:  # req D10
    config = PriorConfig(m_min=10, m_max=200, min_test=5)
    for data in load_data("prior", seed=0, n_datasets=5, config=config):
        assert 10 <= data.n_train <= 195 and data.n_train + data.n_test == 200


# --- config ----------------------------------------------------------------------------------------


def test_yaml_config_equals_defaults() -> None:
    from_yaml = load_prior_config(REPO_ROOT / "configs/data/prior.yaml")
    defaults = PriorConfig()
    def without_activations(config: dict) -> dict:  # activation factories are objects
        config["scm_config"].pop("activation_dist")
        return config

    assert without_activations(from_yaml.dopfnprior_config()) == without_activations(
        defaults.dopfnprior_config()
    )
    for name in ("m_min", "m_max", "min_test", "max_treatment_children", "treated_fraction", "max_attempts"):
        assert getattr(from_yaml, name) == getattr(defaults, name)


def test_config_path_accepted_and_unknown_keys_rejected(tmp_path: Path) -> None:
    data = next(load_data("prior", config=REPO_ROOT / "configs/data/prior.yaml", **SMALL))
    data.validate()
    with pytest.raises(ValueError):
        PriorConfig.from_dict({"dataset": {"m_mni": 3}})
    with pytest.raises(ValueError):
        PriorConfig(graph={"num_node": {"value": 3}}).dopfnprior_config()


# --- D8: case studies --------------------------------------------------------------------------------


def _fake_raw(n: int, f: int = 3, seed: int = 0) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    x_obs = torch.randn(n, f, generator=g)
    x_obs[:, 0] = torch.randint(0, 2, (n,), generator=g).float()
    x_int = x_obs + 0.5  # covariates of x_int differ (recomputed mediators)
    x_int[:, 0] = 1 - x_obs[:, 0]
    return {
        "x_obs": x_obs,
        "y_obs": torch.randn(n, generator=g),
        "x_int": x_int,
        "y_int": torch.randn(n, generator=g),
        "cate": torch.randn(n, generator=g),
    }


@pytest.fixture
def fake_case_study_root(tmp_path: Path) -> Path:
    """Converted-format case studies: 3 files each, with indices 1, 2, 10 (natural order)."""
    for directory in CASE_STUDY_DIRECTORIES.values():
        for i, n in ((10, 50), (1, 20), (2, 30)):
            save_case_study_file(_fake_raw(n, seed=i), tmp_path / directory / f"{directory}_{i}.pt")
    return tmp_path


def test_case_study_names_match_do_pfn() -> None:  # req D8
    assert set(CASE_STUDIES) == set(CASE_STUDY_DIRECTORIES)
    assert "observed_confounder" in CASE_STUDIES


def test_every_case_study_loads(fake_case_study_root: Path) -> None:  # req D8
    for name in CASE_STUDIES:
        datasets = list(load_data(name, data_root=fake_case_study_root))
        assert [d.n_train + d.n_test for d in datasets] == [20, 30, 50]  # files 1, 2, 10
        for d in datasets:
            d.validate()


def test_case_study_split_follows_do_pfn() -> None:  # req D5, D8
    raw = _fake_raw(100)
    data = split_case_study(raw, torch.Generator().manual_seed(0))
    assert (data.n_train, data.n_test) == (80, 20)
    # Every query row is one unit: factual covariates, intervened treatment, its y_int / y_obs / cate.
    for k in range(data.n_test):
        unit = int(torch.nonzero((raw["y_int"] == data.y_in[k])).item())
        torch.testing.assert_close(data.X_in[k, 1:], raw["x_obs"][unit, 1:])
        assert data.X_in[k, 0] == raw["x_int"][unit, 0]
        assert data.y_in_ob[k] == raw["y_obs"][unit] and data.cate[k] == raw["cate"][unit]
    # Context and query units are disjoint and come from x_obs / y_obs.
    assert not set(data.y_ob.tolist()) & set(data.y_in_ob.tolist())


def test_case_study_sizes_seed_and_limits(fake_case_study_root: Path) -> None:  # req D6, D10
    kwargs = {"data_root": fake_case_study_root, "n_train": 12, "n_test": 5}
    a = list(load_data("observed_mediator", seed=0, **kwargs))
    b = list(load_data("observed_mediator", seed=0, **kwargs))
    c = list(load_data("observed_mediator", seed=1, **kwargs))
    assert all(d.n_train == 12 and d.n_test == 5 for d in a)
    assert [_digest(d) for d in a] == [_digest(d) for d in b] != [_digest(d) for d in c]
    assert len(list(load_data("common_effect", data_root=fake_case_study_root, n_datasets=2))) == 2
    with pytest.raises(ValueError):
        list(load_data("common_effect", data_root=fake_case_study_root, n_train=30, n_test=30))


def test_case_study_missing_directory_raises(tmp_path: Path) -> None:  # req D8
    with pytest.raises(FileNotFoundError):
        get_case_study_files("observed_confounder", data_root=tmp_path)


@pytest.mark.skipif(
    not (case_study_root() / "Observed_Confounder").is_dir(),
    reason="converted case studies not available (set $DOPFN_CASE_STUDY_ROOT)",
)
def test_real_case_studies_load() -> None:  # req D8
    for name in CASE_STUDIES:
        files = get_case_study_files(name)
        assert len(files) == 100
        for data in load_data(name, n_datasets=3):
            data.validate()


DO_PFN_DATA = os.environ.get("DOPFN_REPO_DATA")  # <Do-PFN clone>/data/prior_sampling


@pytest.mark.skipif(DO_PFN_DATA is None, reason="set $DOPFN_REPO_DATA to test the pickle conversion")
def test_convert_do_pfn_pickle() -> None:
    pytest.importorskip("dill")
    pickle = Path(DO_PFN_DATA) / "Observed_Confounder" / "Observed_Confounder_1.pkl"
    raw = convert_do_pfn_pickle(pickle)
    data = split_case_study(raw, torch.Generator().manual_seed(0))
    data.validate()
    # Do-PFN's cate is the shared-noise effect: y_int - y_obs = (t_int - t_obs) * cate.
    torch.testing.assert_close(raw["y_int"] - raw["y_obs"], (raw["x_int"][:, 0] - raw["x_obs"][:, 0]) * raw["cate"])


def test_validate_rejects_bad_datasets() -> None:
    good = next(load_data("prior", seed=0, **SMALL))
    bad_t = good.X_ob.clone()
    bad_t[0, 0] = 0.5
    with pytest.raises(ValueError):
        good._replace(X_ob=bad_t).validate()
    with pytest.raises(ValueError):
        good._replace(y_in=good.y_in[:-1]).validate()
    with pytest.raises(ValueError):
        good._replace(cate=good.cate.double()).validate()
