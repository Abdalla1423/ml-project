"""Exercise the formal evaluation entry point with stand-ins for A and B."""

import json
import runpy
import sys
from pathlib import Path
from types import ModuleType

import pytest
import yaml

import dopfn.model
import dopfn.training.loop
import dopfn.viz
from dopfn.baselines import ObservationalRegression
from dopfn.evaluation._synthetic import synthetic_sample


def test_cli_scores_baselines_and_checkpoint_by_dataset(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Use the documented loader, checkpoint and estimator interfaces."""
    cases = []
    sample = synthetic_sample()
    case_sample = synthetic_sample(seed=321)

    def load_data(dataset: str, *, seed: int | None = None, n_datasets: int | None = None):
        cases.append((dataset, seed, n_datasets))
        data = sample if dataset == "prior" else case_sample
        return iter([data, data][:n_datasets])

    seen_results = []
    original_show_metrics = dopfn.viz.show_metrics

    def show_metrics(results, *args, **kwargs):
        seen_results.append(results)
        return original_show_metrics(results, *args, **kwargs)

    class CheckpointModel:
        def eval(self) -> None:
            pass

    checkpoint = tmp_path / "model.pt"
    loaded_paths = []

    def load_checkpoint(path: Path) -> CheckpointModel:
        loaded_paths.append(path)
        return CheckpointModel()

    def regressor(model: CheckpointModel, device: object = "cpu") -> ObservationalRegression:
        assert isinstance(model, CheckpointModel)
        return ObservationalRegression()

    data_module = ModuleType("dopfn.data")
    data_module.CASE_STUDIES = ("observed_confounder",)
    data_module.load_data = load_data
    monkeypatch.setitem(sys.modules, "dopfn.data", data_module)
    monkeypatch.setattr(dopfn.training.loop, "load_checkpoint", load_checkpoint)
    monkeypatch.setattr(dopfn.model, "Regressor", regressor)
    monkeypatch.setattr(dopfn.viz, "show_metrics", show_metrics)
    config = tmp_path / "eval.yaml"
    config.write_text(yaml.safe_dump({
        "seed": 123,
        "n_prior_datasets": 1,
        "case_studies": ["observed_confounder"],
        "baselines": ["observational_regression", "group_mean_difference"],
        "out_dir": str(tmp_path),
    }))
    script = Path(__file__).resolve().parents[1] / "scripts" / "evaluate.py"
    plot = tmp_path / "comparison.png"
    monkeypatch.setattr(sys, "argv", [str(script), "--config", str(config),
                                      "--checkpoint", str(checkpoint), "--plot", str(plot)])
    runpy.run_path(str(script), run_name="__main__")
    scores = json.loads((tmp_path / "metrics.json").read_text())
    assert set(scores["prior"]) == {
        "observational_regression", "group_mean_difference", "dopfn"
    }
    assert cases == [("prior", 123, 1), ("observed_confounder", None, None)]
    assert loaded_paths == [checkpoint]
    assert len(seen_results) == 1
    assert list(seen_results[0]) == ["observed_confounder"]
    assert plot.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert list(tmp_path.glob("*.png")) == [plot]
    assert scores["prior"]["dopfn"]["n_datasets"] == 1
    assert scores["observed_confounder"]["dopfn"]["n_datasets"] == 2
    assert scores["prior"]["dopfn"]["cate_rmse"] == pytest.approx(0.0)
    assert scores["prior"]["group_mean_difference"]["cate_rmse"] > 1
    assert scores["observed_confounder"]["dopfn"]["ate_relative_error"] == pytest.approx(0.0)
