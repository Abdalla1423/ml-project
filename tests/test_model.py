"""Tests for dopfn.model (owner: B). Requirement IDs refer to docs/milestones/ms1/requirements.md."""

import inspect

import numpy as np
import pytest
import torch

from dopfn.data.prior import PriorSampler, derive_seed
from dopfn.model import (
    BarDistribution,
    Model,
    Regressor,
    denormalize_y,
    make_borders,
    nll_loss,
    normalize_y,
    to_model_batch,
)

SMALL = {"n_bins": 20, "d_model": 16, "n_heads": 2, "n_layers": 2, "d_ff": 32}


@pytest.fixture
def model() -> Model:
    torch.manual_seed(0)
    return Model(**SMALL).eval()


@pytest.fixture(scope="module")
def datasets():
    sampler = PriorSampler()
    return [sampler.sample(derive_seed(3, i), n_train=30, n_test=10).data for i in range(4)]


def _table(b: int = 2, n_ob: int = 20, n_in: int = 10, f: int = 4, seed: int = 0):
    """Random batch with a binary treatment column."""
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(b, n_ob + n_in, f, generator=g)
    X[..., 0] = torch.randint(0, 2, (b, n_ob + n_in), generator=g).float()
    return X, torch.randn(b, n_ob, generator=g)


# --- Model: M1-M5 -----------------------------------------------------------------------------------


def test_model_interface() -> None:  # req M1
    assert issubclass(Model, torch.nn.Module)
    assert list(inspect.signature(Model.forward).parameters) == ["self", "X", "y_ob"]


def test_model_forward_shape(model: Model) -> None:  # req M2, M3
    X, y_ob = _table()
    logits = model(X, y_ob)
    assert logits.shape == (2, 10, SMALL["n_bins"])
    probs = torch.softmax(logits, dim=-1)
    torch.testing.assert_close(probs.sum(-1), torch.ones(2, 10))


def test_model_rejects_bad_split(model: Model) -> None:
    X, _ = _table()
    with pytest.raises(ValueError):
        model(X, torch.randn(2, 30))  # no query rows


def test_all_parameters_get_gradients(model: Model) -> None:  # req M4
    model.train()
    X, y_ob = _table()
    nll_loss(model(X, y_ob), torch.randn(2, 10), model.bar).backward()
    for name, p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
        assert p.grad.abs().sum() > 0, name


def test_treatment_column_is_special(model: Model) -> None:  # req M5
    X, y_ob = _table()
    out = model(X, y_ob)
    flipped = X.clone()
    flipped[:, 20:, 0] = 1 - flipped[:, 20:, 0]
    assert not torch.allclose(out, model(flipped, y_ob))
    # Swapping column 0 with another column changes the output (other columns are interchangeable).
    swapped = X[..., [1, 0, 2, 3]]
    assert not torch.allclose(out, model(swapped, y_ob), atol=1e-5)
    torch.testing.assert_close(out, model(X[..., [0, 2, 1, 3]], y_ob), atol=1e-5, rtol=1e-4)


def test_nan_padded_columns_are_ignored(model: Model) -> None:  # decision D2
    X, y_ob = _table()
    padded = torch.cat([X, torch.full((2, 30, 3), float("nan"))], dim=2)
    torch.testing.assert_close(model(padded, y_ob), model(X, y_ob), atol=1e-5, rtol=1e-4)


def test_context_order_and_other_queries_do_not_matter(model: Model) -> None:
    X, y_ob = _table()
    out = model(X, y_ob)
    perm = torch.randperm(20, generator=torch.Generator().manual_seed(1))
    X_perm = torch.cat([X[:, perm], X[:, 20:]], dim=1)
    torch.testing.assert_close(model(X_perm, y_ob[:, perm]), out, atol=1e-5, rtol=1e-4)
    changed = X.clone()
    changed[:, 25:] = torch.randn_like(changed[:, 25:])
    torch.testing.assert_close(model(changed, y_ob)[:, :5], out[:, :5], atol=1e-5, rtol=1e-4)


def test_output_is_invariant_to_outcome_scale(model: Model) -> None:  # decision D4
    X, y_ob = _table()
    torch.testing.assert_close(model(X, 3.0 * y_ob + 7.0), model(X, y_ob), atol=1e-4, rtol=1e-4)


def test_normalize_roundtrip() -> None:
    y_ob, y = torch.randn(3, 10) * 4 + 2, torch.randn(3, 5)
    torch.testing.assert_close(denormalize_y(normalize_y(y, y_ob), y_ob), y)


# --- Bar distribution and loss: M3, M6 ---------------------------------------------------------------


def test_bar_distribution_density_integrates_to_one() -> None:  # req M3
    dist = BarDistribution(torch.tensor([-2.0, -0.5, 0.0, 1.0, 3.0]))
    logits = torch.randn(7, 4)
    probs = torch.softmax(logits, -1)
    density = torch.stack(
        [dist.log_prob(logits, mid.expand(7)).exp() for mid in dist.mids], dim=-1
    )  # density at each bin center, shape (7, 4)
    torch.testing.assert_close((density * dist.widths).sum(-1), torch.ones(7))
    torch.testing.assert_close(density * dist.widths, probs)


def test_bar_distribution_mean_and_clamping() -> None:
    dist = BarDistribution(torch.tensor([0.0, 1.0, 2.0, 4.0]))
    one_hot = torch.tensor([[-1e4, -1e4, 0.0]])
    torch.testing.assert_close(dist.mean(one_hot), torch.tensor([3.0]))
    assert dist.bin_index(torch.tensor([-5.0, 0.5, 10.0])).tolist() == [0, 0, 2]
    with pytest.raises(ValueError):
        BarDistribution(torch.tensor([0.0, 0.0, 1.0]))


def test_make_borders() -> None:
    borders = make_borders(50, torch.randn(10_000))
    assert borders.shape == (51,) and torch.all(borders[1:] > borders[:-1])
    discrete = make_borders(10, torch.tensor([0.0, 1.0] * 500))  # many equal quantiles
    assert torch.all(discrete[1:] > discrete[:-1])


def test_nll_loss_prefers_the_right_target(model: Model) -> None:  # req M6
    dist = BarDistribution(torch.linspace(-3, 3, 21))
    logits = torch.full((2, 5, 20), -10.0)
    logits[..., 15] = 10.0  # mass in the bin [1.5, 1.8]
    right, wrong = torch.full((2, 5), 1.6), torch.full((2, 5), -2.0)
    loss_right, loss_wrong = nll_loss(logits, right, dist), nll_loss(logits, wrong, dist)
    assert loss_right.ndim == 0 and torch.isfinite(loss_right)
    assert loss_right < loss_wrong


def test_set_borders(model: Model) -> None:
    model.set_borders(torch.linspace(-1, 1, SMALL["n_bins"] + 1))
    assert "bar.borders" in model.state_dict()
    with pytest.raises(ValueError):
        model.set_borders(torch.linspace(-1, 1, 5))


# --- Adapter -------------------------------------------------------------------------------------------


def test_adapter_stacks_and_pads(datasets) -> None:
    batch = to_model_batch(datasets)
    n_features = max(d.n_features for d in datasets)
    assert batch.X.shape == (4, 40, n_features) and batch.n_ob == 30
    assert batch.y_ob.shape == (4, 30) and batch.y_target.shape == batch.cate.shape == (4, 10)
    for i, d in enumerate(datasets):
        torch.testing.assert_close(batch.X[i, :30, : d.n_features], d.X_ob)
        torch.testing.assert_close(batch.X[i, 30:, : d.n_features], d.X_in)
        assert torch.isnan(batch.X[i, :, d.n_features :]).all()
        torch.testing.assert_close(batch.y_target[i], d.y_in)
    observational = to_model_batch(datasets, target="observational")
    torch.testing.assert_close(observational.y_target[0], datasets[0].y_in_ob)


def test_adapter_rejects_mixed_sizes(datasets) -> None:  # decision D2
    other = PriorSampler().sample(1, n_train=31, n_test=10).data
    with pytest.raises(ValueError):
        to_model_batch([datasets[0], other])
    with pytest.raises(ValueError):
        to_model_batch(datasets, target="counterfactual")


def test_model_runs_on_padded_prior_batch(model: Model, datasets) -> None:
    batch = to_model_batch(datasets)
    logits = model(batch.X, batch.y_ob)
    assert logits.shape == (4, 10, SMALL["n_bins"]) and torch.isfinite(logits).all()


# --- Regressor: R1-R5 -----------------------------------------------------------------------------------


def test_regressor_interface() -> None:  # req R1
    assert list(inspect.signature(Regressor.__init__).parameters)[:2] == ["self", "model"]
    for method in ("fit", "predict", "predict_cate"):
        assert callable(getattr(Regressor, method))


def test_fit_only_stores_data(model: Model, datasets) -> None:  # req R2
    before = {k: v.clone() for k, v in model.state_dict().items()}
    reg = Regressor(model).fit(datasets[0].X_ob, datasets[0].y_ob)
    assert reg.fit(datasets[0].X_ob.numpy(), datasets[0].y_ob.numpy()) is reg
    for k, v in model.state_dict().items():
        assert torch.equal(v, before[k]), k


def test_predict_shapes_and_cate(model: Model, datasets) -> None:  # req R3, R4
    d = datasets[0]
    reg = Regressor(model).fit(d.X_ob.numpy(), d.y_ob.numpy())
    pred = reg.predict(d.X_in.numpy())
    assert isinstance(pred, np.ndarray) and pred.shape == (d.n_test,)
    X1, X0 = d.X_in.clone(), d.X_in.clone()
    X1[:, 0], X0[:, 0] = 1, 0
    np.testing.assert_allclose(reg.predict_cate(d.X_in), reg.predict(X1) - reg.predict(X0), atol=1e-5)
    assert torch.equal(d.X_in[:, 0], datasets[0].X_in[:, 0])  # input not modified


def test_predict_is_on_the_original_scale(model: Model, datasets) -> None:
    d = datasets[0]
    reg = Regressor(model).fit(d.X_ob, d.y_ob)
    shifted = Regressor(model).fit(d.X_ob, 10 * d.y_ob + 5)
    np.testing.assert_allclose(shifted.predict(d.X_in), 10 * reg.predict(d.X_in) + 5, rtol=1e-4, atol=1e-3)
    np.testing.assert_allclose(shifted.predict_cate(d.X_in), 10 * reg.predict_cate(d.X_in), rtol=1e-4, atol=1e-3)
    probs, borders = reg.predict_distribution(d.X_in)
    assert probs.shape == (d.n_test, SMALL["n_bins"]) and borders.shape == (d.n_test, SMALL["n_bins"] + 1)
    np.testing.assert_allclose(probs.sum(1), 1, atol=1e-5)


def test_regressor_in_a_harness_loop(model: Model, datasets) -> None:  # req R5
    """The fit / predict / predict_cate protocol that dopfn.evaluation uses."""
    reg = Regressor(model)
    for d in datasets:
        reg.fit(d.X_ob.numpy(), d.y_ob.numpy())
        assert np.isfinite(reg.predict(d.X_in.numpy())).all()
        assert np.isfinite(reg.predict_cate(d.X_in.numpy())).all()


def test_predict_before_fit_raises(model: Model) -> None:
    with pytest.raises(RuntimeError):
        Regressor(model).predict(np.zeros((3, 2)))
