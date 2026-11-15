"""Tests for dopfn.training (owner: B). Requirement IDs refer to docs/milestones/ms1/requirements.md."""

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from dopfn.data.prior import PriorSampler, derive_seed
from dopfn.model import Model, Regressor, nll_loss, normalize_y, to_model_batch
from dopfn.training import TrainConfig, load_checkpoint, load_train_config, train
from dopfn.training.loop import read_metrics, sample_batch

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL = {"n_bins": 20, "d_model": 16, "n_heads": 2, "n_layers": 1, "d_ff": 32}


def _config(out_dir: Path, **kwargs) -> TrainConfig:
    base = TrainConfig(
        steps=6, batch_size=2, n_train_min=10, n_train_max=30, n_test=8, warmup_steps=2,
        val_every=3, n_val_datasets=4, val_batch_size=2, n_border_datasets=10, log_every=2,
        checkpoint_every=3, out_dir=str(out_dir), device="cpu",
    )
    return replace(base, **kwargs)


def test_train_config_defaults() -> None:
    assert TrainConfig().steps > 0
    with pytest.raises(ValueError):
        TrainConfig(n_train_min=10, n_train_max=5)


def test_yaml_configs_load() -> None:
    for name in ("default", "debug"):
        model_config, config = load_train_config(REPO_ROOT / "configs/train" / f"{name}.yaml")
        Model.from_config(model_config)
        assert config.steps > 0 and config.prior.m_max > 0
    _, config = load_train_config(REPO_ROOT / "configs/train/debug.yaml", steps=7, device=None)
    assert config.steps == 7 and config.device == "auto"


def test_batches_are_deterministic() -> None:  # req D6 (seeding)
    config = _config(Path("unused"))
    a, b = (sample_batch(PriorSampler(), 0, 5, config) for _ in range(2))
    assert torch.equal(torch.nan_to_num(a.X), torch.nan_to_num(b.X)) and torch.equal(a.y_target, b.y_target)
    c = sample_batch(PriorSampler(), 0, 6, config)
    assert not torch.equal(a.y_ob, c.y_ob) or a.y_ob.shape != c.y_ob.shape


def test_train_writes_outputs_and_checkpoint_roundtrip(tmp_path: Path) -> None:  # req M7
    torch.manual_seed(0)
    model = Model(**MODEL)
    history = train(model, _config(tmp_path))
    assert history["step"] == [2, 4, 6] and history["val_step"] == [3, 6]
    for name in ("metrics.csv", "loss_curve.png", "checkpoint_latest.pt", "checkpoint_best.pt"):
        assert (tmp_path / name).exists(), name
    rows = read_metrics(tmp_path)
    assert any("val_nll" in r and "val_cate_mse_group_means" in r for r in rows)

    loaded = load_checkpoint(tmp_path / "checkpoint_latest.pt")
    X = torch.randn(1, 15, 3)
    X[..., 0] = torch.randint(0, 2, (1, 15)).float()
    y_ob = torch.randn(1, 10)
    torch.testing.assert_close(loaded(X, y_ob), model.cpu().eval()(X, y_ob))
    reg = Regressor.from_checkpoint(tmp_path / "checkpoint_best.pt")
    assert reg.fit(X[0, :10], y_ob[0]).predict(X[0, 10:]).shape == (5,)


def test_same_seed_same_losses(tmp_path: Path) -> None:
    runs = []
    for name in ("a", "b"):
        torch.manual_seed(123)
        runs.append(train(Model(**MODEL), _config(tmp_path / name)))
    assert runs[0] == runs[1]


def test_resume_continues_the_run(tmp_path: Path) -> None:
    torch.manual_seed(0)
    full = train(Model(**MODEL), _config(tmp_path / "full"))
    torch.manual_seed(0)
    train(Model(**MODEL), _config(tmp_path / "split", steps=6, max_minutes=0.0))  # stops after step 1
    resumed = train(Model(**MODEL), _config(tmp_path / "split"), resume=True)
    assert resumed["step"] == [2, 4, 6]
    for a, b in zip(resumed["val"], full["val"], strict=True):
        assert a == pytest.approx(b, rel=1e-4)


def test_loss_decreases_on_a_fixed_batch() -> None:  # req M6/M7: the model can fit
    torch.manual_seed(0)
    sampler = PriorSampler()
    batch = to_model_batch([sampler.sample(derive_seed(9, i), n_train=40, n_test=20).data for i in range(4)])
    model = Model(**{**MODEL, "n_bins": 50, "d_model": 32, "n_layers": 2})
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    y = normalize_y(batch.y_target, batch.y_ob)
    losses = []
    for _ in range(60):
        loss = nll_loss(model(batch.X, batch.y_ob), y, model.bar)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] - 0.5
