"""Evaluate baselines and a Do-PFN checkpoint on prior data and case studies."""

import argparse
import json
import math
from itertools import islice
from pathlib import Path

import matplotlib.pyplot as plt
import yaml

from dopfn.baselines import GroupMeanDifference, ObservationalRegression
from dopfn.evaluation import CausalEstimator, evaluate
from dopfn.viz import show_metrics


def main() -> None:
    """Run the requested datasets and save a score table as JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/eval/default.yaml")
    parser.add_argument("--checkpoint", type=Path, help="Checkpoint saved by dopfn.training.")
    parser.add_argument("--synthetic", action="store_true", help="Use the test-only SCM.")
    # 只写 --plot 时保存到项目的上一级目录，本机对应桌面的 PML 文件夹。
    # 现在统一放在该目录下的 case_studies/case_study_metrics.png。
    parser.add_argument("--plot", type=Path, nargs="?",
                        const=Path(__file__).resolve().parents[2] / "case_studies" / "case_study_metrics.png",
                        help="Save the three-metric comparison to this path.")
    args = parser.parse_args()
    if args.synthetic and args.checkpoint:
        parser.error("--synthetic uses the test-only SCM and baseline estimators.")

    config = yaml.safe_load(Path(args.config).read_text())
    baseline_types = {
        "observational_regression": ObservationalRegression,
        "group_mean_difference": GroupMeanDifference,
    }
    estimators: dict[str, CausalEstimator] = {
        name: baseline_types[name]() for name in config["baselines"]
    }
    if args.checkpoint:
        from dopfn.model import Regressor
        from dopfn.training.loop import load_checkpoint

        try:
            model = load_checkpoint(args.checkpoint)
            model.eval()
            estimators["dopfn"] = Regressor(model)
        except NotImplementedError:
            parser.exit(1, "Checkpoint loading or Regressor is still a stub.\n")
    if not estimators:
        parser.error("Configure at least one baseline or provide --checkpoint.")

    seed = int(config["seed"])
    if args.synthetic:
        from dopfn.evaluation._synthetic import synthetic_sample

        datasets = {"synthetic_test_only": [synthetic_sample(seed)]}
    else:
        from dopfn.data import CASE_STUDIES, load_data

        n_prior = int(config["n_prior_datasets"])
        cases = config["case_studies"]
        if cases == "all":
            cases = CASE_STUDIES
        if not isinstance(cases, (list, tuple)) or any(case not in CASE_STUDIES for case in cases):
            parser.error("case_studies must be 'all' or a list of supported case-study names.")
        try:
            # TODO(team): D6 - coordinate this seed with the training and validation split.
            # 同时取得 A 生成的数据和对应的 SCM 图；评估时使用其中的数据。
            prior_samples = list(
                islice(load_data("prior", seed=seed, with_graph=True), n_prior)
            )
            datasets = {"prior": [sample for sample, _ in prior_samples]}
            for case in cases:
                datasets[case] = list(load_data(case))
        except NotImplementedError:
            parser.exit(1, "load_data is still a stub. Use --synthetic for a local test.\n")

    # 按数据集分别评估，同一数据集下的所有模型都使用同一批 samples。
    results = {}
    for dataset_name, samples in datasets.items():
        results[dataset_name] = {}
        for name, estimator in estimators.items():
            scores = evaluate(estimator, samples)
            # 常量 outcome 的 NMSE 可能为 NaN，转成 None 后会保存为 JSON 的 null。
            results[dataset_name][name] = {
                key: value if math.isfinite(value) else None
                for key, value in scores.items()
            }

    # 统一比较三项指标：CID MSE、CATE MSE 和 ATE 相对误差。
    if args.plot:
        args.plot.parent.mkdir(parents=True, exist_ok=True)
        # 有 case study 时每类一列；只评估 prior 时展示 prior。
        comparison = {name: scores for name, scores in results.items() if name != "prior"} or results
        figure = show_metrics(comparison, args.plot,
                              title="CID, CATE and ATE comparison")
        plt.close(figure)

    out_dir = Path(config["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    outfile = out_dir / ("synthetic_test_only.json" if args.synthetic else "metrics.json")
    report = json.dumps(results, indent=2, allow_nan=False)
    outfile.write_text(report + "\n")
    print(report)
    print(f"Saved {outfile}")


if __name__ == "__main__":
    main()
