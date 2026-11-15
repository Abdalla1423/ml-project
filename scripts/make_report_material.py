"""Create the figures and tables of the ms1 report from the stored cluster results.

Usage:
    python scripts/make_report_material.py
    python scripts/make_report_material.py --limit 5      # quick test with 5 datasets per set

Reads ``metrics.json``, ``train_metrics.csv`` and ``checkpoint_best.pt`` from ``--results`` and the
converted case studies from ``$DOPFN_CASE_STUDY_ROOT``. Writes PNG and PDF figures, LaTeX tables and
an index (``README.md``) with captions and key numbers to ``--out``. Scores per dataset are cached in
``per_dataset_scores.csv``; delete the file to recompute them.
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import yaml
from sklearn.linear_model import LogisticRegression

from dopfn.baselines import GroupMeanDifference, ObservationalRegression
from dopfn.data import load_data
from dopfn.data.prior import PriorSampler, derive_seed
from dopfn.evaluation._sample import unpack_sample
from dopfn.evaluation.metrics import normalized_mse
from dopfn.model import Regressor
from dopfn.viz import show

MS1 = Path(__file__).resolve().parents[1] / "docs/milestones/ms1"
CASES = ("observed_confounder", "observed_mediator", "confounder_mediator",
         "unobserved_confounder", "frontdoor_criterion", "backdoor_criterion")
MODELS = ("dopfn", "observational_regression", "group_mean_difference")
LABELS = {"dopfn": "Our model", "observational_regression": "Regression",
          "group_mean_difference": "Group means"}
COLORS = {"dopfn": "#0072B2", "observational_regression": "#D55E00",
          "group_mean_difference": "#009E73"}
TITLES = {"prior": "Prior (held out)", "observed_confounder": "Observed Confounder",
          "observed_mediator": "Observed Mediator", "confounder_mediator": "Confounder + Mediator",
          "unobserved_confounder": "Unobserved Confounder",
          "frontdoor_criterion": "Front-Door", "backdoor_criterion": "Back-Door"}
METRICS = ("outcome_nmse", "cate_nmse")
EXAMPLE_SEED, STATS_SEED = 2026, 4242  # streams used by neither training (0) nor evaluation (1000)


def save(fig: plt.Figure, out: Path, name: str) -> None:
    """Save a figure as PNG (300 dpi) and PDF, then close it."""
    for suffix in ("png", "pdf"):
        fig.savefig(out / f"{name}.{suffix}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fmt(value: float | None) -> str:
    """Three significant digits; plain decimals for the usual range."""
    if value is None or not np.isfinite(value):
        return "--"
    return f"{value:.4f}" if abs(value) < 1 else f"{value:.3g}"


# --- Training curve ------------------------------------------------------------------------------


def plot_training(results: Path, out: Path) -> dict[str, float]:
    """Loss curve and validation CATE error against the difference of group means."""
    rows = list(csv.DictReader((results / "train_metrics.csv").open()))
    step = np.array([int(r["step"]) for r in rows])
    train = np.array([float(r["train_nll"]) if r["train_nll"] else np.nan for r in rows])
    val = [(int(r["step"]), float(r["val_nll"]), float(r["val_cate_mse"]),
            float(r["val_cate_mse_group_means"])) for r in rows if r["val_nll"]]
    v_step, v_nll, v_cate, v_ref = map(np.array, zip(*val, strict=True))

    fig, (left, right) = plt.subplots(1, 2, figsize=(10.5, 3.6), layout="constrained")
    left.plot(step, train, color="#9ecae1", linewidth=0.9, label="training (mean of 50 steps)")
    left.plot(v_step, v_nll, color="#D55E00", marker="o", markersize=2.5, linewidth=1.2,
              label="validation (256 held-out datasets)")
    left.set(xlabel="training step", ylabel="NLL of the normalized outcome", title="Pre-training loss")
    right.plot(v_step, v_cate, color="#0072B2", marker="o", markersize=2.5, linewidth=1.2,
               label="our model")
    right.axhline(v_ref[-1], color="#009E73", linestyle="--", linewidth=1.2,
                  label="difference of group means")
    right.set(xlabel="training step", ylabel="CATE MSE (normalized outcome)",
              title="Validation CATE error", ylim=(0, 0.45))
    for ax in (left, right):
        ax.grid(alpha=0.2)
        ax.legend(frameon=False, fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    save(fig, out, "fig_training")
    return {"steps": int(step[-1]), "minutes": float(rows[-1]["minutes"]),
            "val_nll_first": float(v_nll[0]), "val_nll_last": float(v_nll[-1]),
            "val_nll_3000_before": float(v_nll[-7]), "train_nll_last": float(np.nanmean(train[-20:])),
            "val_cate_last": float(v_cate[-1]), "val_cate_ref": float(v_ref[-1])}


# --- Prior exploration ---------------------------------------------------------------------------


def no_overlap_share(X_ob: np.ndarray) -> float:
    """Share of context rows whose estimated propensity P(t=1 | covariates) is below 0.05 or above 0.95.

    The propensity comes from a logistic regression on the standardized covariates. For such rows the
    data hold (almost) no comparable units of the other treatment group.
    """
    t, covariates = X_ob[:, 0], X_ob[:, 1:]
    if covariates.shape[1] == 0 or t.min() == t.max():
        return float("nan")
    scale = covariates.std(axis=0)
    covariates = np.clip((covariates - covariates.mean(axis=0)) / np.where(scale > 0, scale, 1.0), -10, 10)
    propensity = LogisticRegression(max_iter=500).fit(covariates, t).predict_proba(covariates)[:, 1]
    return float(np.mean((propensity < 0.05) | (propensity > 0.95)))


def prior_statistics(n_datasets: int) -> dict[str, Any]:
    """Describe the SCMs and datasets that the prior generates (sizes as in training)."""
    sampler = PriorSampler()
    stats = defaultdict(list)
    for i in range(n_datasets):
        sample = sampler.sample(derive_seed(STATS_SEED, i), n_train=None, n_test=None)
        data, graph = sample.data, sample.graph
        visible = set(sample.input_vars) - {"t"}
        ancestors_t, ancestors_y = nx.ancestors(graph, "t"), nx.ancestors(graph, "y")
        descendants_t = nx.descendants(graph, "t")
        confounders = ancestors_t & ancestors_y
        mediators = descendants_t & ancestors_y
        t_ob = data.X_ob[:, 0]
        std = float(data.y_ob.std()) or 1.0
        stats["nodes"].append(graph.number_of_nodes())
        stats["features"].append(data.X_ob.shape[1])
        stats["rows_context"].append(len(data.X_ob))
        stats["rows_query"].append(len(data.X_in))
        stats["treated_fraction"].append(float(t_ob.mean()))
        stats["one_group"].append(float(t_ob.min() == t_ob.max()))
        stats["few_treated"].append(float(min(t_ob.mean(), 1 - t_ob.mean()) < 0.05))
        stats["no_effect"].append(float("y" not in descendants_t))
        stats["abs_ate_in_std"].append(abs(float(data.cate.mean())) / std)
        stats["heterogeneous"].append(float(data.cate.std() > 1e-6))
        stats["has_confounder"].append(float(bool(confounders)))
        stats["hidden_confounder"].append(float(bool(confounders - visible)))
        stats["has_mediator"].append(float(bool(mediators)))
        stats["observed_mediator"].append(float(bool(mediators & visible)))
        stats["only_treatment"].append(float(data.X_ob.shape[1] == 1))
        stats["no_overlap"].append(no_overlap_share(data.X_ob.numpy().astype(float)))
    summary: dict[str, Any] = {"n_datasets": n_datasets}
    for key in ("nodes", "features", "rows_context", "rows_query", "treated_fraction", "abs_ate_in_std",
                "no_overlap"):
        summary[key] = [float(np.nanquantile(stats[key], q)) for q in (0.0, 0.25, 0.5, 0.75, 1.0)]
    summary["mostly_no_overlap"] = float(np.nanmean(np.array(stats["no_overlap"]) > 0.5))
    for key in ("one_group", "few_treated", "no_effect", "heterogeneous", "has_confounder",
                "hidden_confounder", "has_mediator", "observed_mediator", "only_treatment"):
        summary[key] = float(np.mean(stats[key]))
    return summary


def write_prior_table(stats: dict[str, Any], out: Path) -> None:
    """LaTeX table of the prior statistics."""
    quantile_rows = (("nodes", "Nodes of the SCM (incl. $t$, $y$)", "{:.0f}"),
                     ("features", "Columns of $X$ (incl. treatment)", "{:.0f}"),
                     ("rows_context", "Context rows $n_{ob}$", "{:.0f}"),
                     ("rows_query", "Query rows $n_{in}$", "{:.0f}"),
                     ("treated_fraction", "Treated fraction in the context", "{:.2f}"),
                     ("abs_ate_in_std", r"$|\mathrm{ATE}| / \mathrm{std}(y_{ob})$", "{:.2f}"),
                     ("no_overlap", "Context rows without overlap (propensity outside $[0.05, 0.95]$)", "{:.2f}"))
    share_rows = (("has_confounder", "Common cause of $t$ and $y$ exists"),
                  ("hidden_confounder", "\\quad at least one of them hidden"),
                  ("has_mediator", "Mediator between $t$ and $y$ exists"),
                  ("observed_mediator", "\\quad at least one of them observed"),
                  ("no_effect", "No causal path $t \\to y$ (true effect is 0)"),
                  ("heterogeneous", "Effect differs between units"),
                  ("mostly_no_overlap", "More than half of the context rows without overlap"),
                  ("few_treated", "Smaller treatment group below 5\\,\\% of the context"),
                  ("one_group", "Only one treatment group in the context"),
                  ("only_treatment", "No covariates ($X$ is only the treatment)"))
    lines = [r"\begin{tabular}{lrrrrr}", r"\toprule",
             r"Quantity & min & 25\,\% & median & 75\,\% & max \\", r"\midrule"]
    lines += [f"{label} & " + " & ".join(f.format(v) for v in stats[key]) + r" \\"
              for key, label, f in quantile_rows]
    lines += [r"\midrule", r"Property & \multicolumn{5}{r}{share of datasets} \\", r"\midrule"]
    lines += [f"{label} & \\multicolumn{{5}}{{r}}{{{100 * stats[key]:.1f}\\,\\%}} \\\\"
              for key, label in share_rows]
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "tab_prior_stats.tex").write_text("\n".join(lines) + "\n")


def pick_examples(n: int) -> list[Any]:
    """Prior datasets that make readable example figures (small graph, both groups, clear effect)."""
    sampler, chosen = PriorSampler(), []
    for i in range(2000):
        sample = sampler.sample(derive_seed(EXAMPLE_SEED, i), n_train=300, n_test=150)
        data = sample.data
        if data.X_ob.shape[1] < 2 or sample.graph.number_of_nodes() > 7:
            continue
        treated = float(data.X_ob[:, 0].mean())
        effect = abs(float(data.cate.mean())) / (float(data.y_ob.std()) or 1.0)
        corr = abs(np.corrcoef(data.X_ob[:, 1].numpy(), data.y_ob.numpy())[0, 1])
        if 0.25 < treated < 0.75 and effect > 0.4 and np.isfinite(corr) and corr > 0.3:
            chosen.append(sample)
        if len(chosen) == n:
            break
    return chosen


def plot_examples(model: Regressor, out: Path) -> int:
    """Example datasets with their causal graph, without and with the model's predictions."""
    examples = pick_examples(3)
    datasets = [e.data for e in examples]
    graphs = [e.graph for e in examples]
    fig = show(datasets, graphs=graphs,
               title="Datasets sampled from our prior: causal graph, observational and interventional data")
    save(fig, out, "fig_prior_examples")
    predictions = [model.fit(d.X_ob, d.y_ob).predict(d.X_in) for d in datasets]
    fig = show(datasets, graphs=graphs, predictions=predictions,
               title="Predictions of the trained model for the interventional outcomes (crosses)")
    save(fig, out, "fig_predictions")
    return len(examples)


def plot_case_studies(model: Regressor, out: Path) -> None:
    """One dataset of every case study (first file), in two figures of three; with model predictions."""
    datasets = {case: next(iter(load_data(case, n_datasets=1))) for case in CASES}
    for part, cases in (("a", CASES[:3]), ("b", CASES[3:])):
        chosen = [datasets[c] for c in cases]
        names = [TITLES[c] for c in cases]
        save(show(chosen, names=names), out, f"fig_case_studies_{part}")
        predictions = [model.fit(d.X_ob, d.y_ob).predict(d.X_in) for d in chosen]
        save(show(chosen, names=names, predictions=predictions), out, f"fig_case_studies_{part}_predictions")


# --- Scores per dataset --------------------------------------------------------------------------


def score_all(model: Regressor, results: Path, out: Path, limit: int | None) -> list[dict[str, Any]]:
    """Normalized MSE of every model on every dataset (cached as CSV)."""
    cache = out / "per_dataset_scores.csv"
    if cache.exists() and limit is None:
        return [{**r, "outcome_nmse": float(r["outcome_nmse"]), "cate_nmse": float(r["cate_nmse"])}
                for r in csv.DictReader(cache.open())]
    config = yaml.safe_load((MS1.parents[2] / "configs/eval/default.yaml").read_text())
    n_prior = limit or int(config["n_prior_datasets"])
    sets = {"prior": load_data("prior", seed=int(config["seed"]), n_datasets=n_prior)}
    sets.update({case: load_data(case, n_datasets=limit) for case in CASES})
    estimators = {"dopfn": model, "observational_regression": ObservationalRegression(),
                  "group_mean_difference": GroupMeanDifference()}
    rows = []
    for name, samples in sets.items():
        for i, sample in enumerate(samples):
            X_ob, y_ob, X_in, y_in, cate = unpack_sample(sample)
            scale = float(np.ptp(y_in))
            if not scale:
                continue
            try:
                scored = []
                for key, estimator in estimators.items():
                    estimator.fit(X_ob, y_ob)
                    scored.append({
                        "dataset": name, "index": i, "model": key,
                        "outcome_nmse": normalized_mse(y_in, estimator.predict(X_in), scale=scale),
                        "cate_nmse": normalized_mse(cate, estimator.predict_cate(X_in), scale=scale),
                    })
            except ValueError as error:  # e.g. a context with a single treatment group
                print(f"skipped {name} {i}: {error}", flush=True)
                continue
            rows += scored
        print(f"scored {name}", flush=True)
    if limit is None:
        with cache.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return rows


def by_dataset(rows: list[dict[str, Any]], metric: str) -> dict[str, dict[str, np.ndarray]]:
    """``{dataset: {model: scores aligned by dataset index}}``."""
    table: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in sorted(rows, key=lambda r: int(r["index"])):
        table[r["dataset"]][r["model"]].append(r[metric])
    return {d: {m: np.array(v) for m, v in models.items()} for d, models in table.items()}


def plot_boxplots(rows: list[dict[str, Any]], out: Path) -> None:
    """Distribution of the normalized MSE over datasets (log scale), per dataset and model."""
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.4), layout="constrained", sharex=True)
    names = ["prior", *CASES]
    for ax, metric, heading in zip(axes, METRICS, ("Interventional outcome", "CATE"), strict=True):
        scores = by_dataset(rows, metric)
        for d, name in enumerate(names):
            for m, key in enumerate(MODELS):
                values = np.maximum(scores[name][key], 1e-7)
                box = ax.boxplot(values, positions=[d + (m - 1) * 0.26], widths=0.22, patch_artist=True,
                                 flierprops={"markersize": 2, "alpha": 0.5}, medianprops={"color": "black"})
                box["boxes"][0].set(facecolor=COLORS[key], alpha=0.85)
        ax.set_yscale("log")
        ax.set_ylabel(f"normalized MSE, {heading}")
        ax.grid(axis="y", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    axes[1].set_xticks(range(len(names)), [TITLES[n].replace(" ", "\n", 1) for n in names], fontsize=9)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[k], alpha=0.85) for k in MODELS]
    axes[0].legend(handles, [LABELS[k] for k in MODELS], frameon=False, ncol=3, fontsize=9,
                   loc="lower left", bbox_to_anchor=(0, 1.0))
    save(fig, out, "fig_nmse_boxplots")


def win_rates(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Share of datasets on which our model has a lower error than both baselines."""
    rates: dict[str, dict[str, float]] = defaultdict(dict)
    for metric in METRICS:
        for name, scores in by_dataset(rows, metric).items():
            best_baseline = np.minimum(scores["observational_regression"], scores["group_mean_difference"])
            rates[name][metric] = float(np.mean(scores["dopfn"] < best_baseline))
    return rates


# --- Tables --------------------------------------------------------------------------------------


def write_results_tables(metrics: dict[str, Any], rates: dict[str, dict[str, float]], out: Path) -> None:
    """Results per dataset and model as LaTeX and Markdown (numbers from the cluster evaluation)."""
    tex = [r"\begin{tabular}{llrrrr}", r"\toprule",
           r" & & \multicolumn{2}{c}{Outcome NMSE} & \multicolumn{2}{c}{CATE NMSE} \\",
           r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
           r"Dataset & Model & mean & median & mean & median \\", r"\midrule"]
    md = ["| Dataset | Model | Outcome NMSE mean (95% CI) | median | CATE NMSE mean (95% CI) | median |",
          "|---|---|---|---|---|---|"]
    for name in ["prior", *CASES]:
        scores = metrics[name]
        columns = [f"{metric}{suffix}" for metric in METRICS for suffix in ("", "_median")]
        best = {c: min(MODELS, key=lambda m, c=c: scores[m][c]) for c in columns}
        for i, model in enumerate(MODELS):
            cells = [f"\\textbf{{{fmt(scores[model][c])}}}" if best[c] == model else fmt(scores[model][c])
                     for c in columns]
            first = f"{TITLES[name]} ($n={scores[model]['n_datasets']}$)" if i == 0 else ""
            tex.append(f"{first} & {LABELS[model]} & " + " & ".join(cells) + r" \\")
            md_cells = []
            for metric in METRICS:
                mean = fmt(scores[model][metric])
                ci = f"{fmt(scores[model][metric + '_ci95_low'])}–{fmt(scores[model][metric + '_ci95_high'])}"
                median = fmt(scores[model][metric + "_median"])
                bold = "**" if best[metric] == model else ""
                md_cells += [f"{bold}{mean}{bold} ({ci})", median]
            md.append(f"| {TITLES[name] if i == 0 else ''} | {LABELS[model]} | " + " | ".join(md_cells) + " |")
        tex.append(r"\midrule" if name != CASES[-1] else r"\bottomrule")
    tex.append(r"\end{tabular}")
    (out / "tab_results.tex").write_text("\n".join(tex) + "\n")

    md += ["", "| Dataset | Our model best on outcome | Our model best on CATE |", "|---|---|---|"]
    wins = [r"\begin{tabular}{lrr}", r"\toprule",
            r"Dataset & Outcome & CATE \\", r"\midrule"]
    for name in ["prior", *CASES]:
        outcome, cate = (100 * rates[name][m] for m in METRICS)
        md.append(f"| {TITLES[name]} | {outcome:.0f} % | {cate:.0f} % |")
        wins.append(f"{TITLES[name]} & {outcome:.0f}\\,\\% & {cate:.0f}\\,\\% \\\\")
    wins += [r"\bottomrule", r"\end{tabular}"]
    (out / "tab_win_rates.tex").write_text("\n".join(wins) + "\n")
    (out / "tab_results.md").write_text("\n".join(md) + "\n")


def write_setup_table(checkpoint: dict[str, Any], n_parameters: int, training: dict[str, float], out: Path) -> None:
    """Model and training setup as a LaTeX table."""
    m, t = checkpoint["model_config"], checkpoint["train_config"]
    rows = (("Architecture", "attention between features, then between rows (nanoTabPFN)"),
            ("Layers / embedding size / heads / MLP size",
             f"{m['n_layers']} / {m['d_model']} / {m['n_heads']} / {m['d_ff']}"),
            ("Output", f"bar distribution with {m['n_bins']} bins (quantile borders)"),
            ("Parameters", f"{n_parameters / 1e6:.2f}\\,M"),
            ("Loss", "negative log-likelihood of the interventional outcome"),
            ("Optimizer", f"AdamW, learning rate {t['lr']:g}, weight decay {t['weight_decay']:g}"),
            ("Schedule", f"{t['warmup_steps']} warm-up steps, then cosine decay; gradient clipping {t['grad_clip']:g}"),
            ("Steps $\\times$ batch size", f"{t['steps']} $\\times$ {t['batch_size']} datasets "
                                           f"({t['steps'] * t['batch_size'] / 1000:.0f}k datasets)"),
            ("Context rows / query rows", f"{t['n_train_min']}--{t['n_train_max']} / {t['n_test']}"),
            ("Validation", f"{t['n_val_datasets']} held-out prior datasets every {t['val_every']} steps"),
            ("Training time", f"{training['minutes']:.0f} min on one GPU"))
    lines = [r"\begin{tabular}{ll}", r"\toprule"]
    lines += [f"{name} & {value} \\\\" for name, value in rows]
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "tab_setup.tex").write_text("\n".join(lines) + "\n")


def write_index(out: Path, training: dict[str, float], stats: dict[str, Any], metrics: dict[str, Any],
                rates: dict[str, dict[str, float]], n_parameters: int, n_examples: int,
                rows: list[dict[str, Any]]) -> None:
    """Index of all report material with suggested captions and the key numbers."""
    shown = {(r["dataset"], r["model"]): r["outcome_nmse"] for r in rows if int(r["index"]) == 0}
    example = lambda case: (f"{TITLES[case]}: our model {fmt(shown[case, 'dopfn'])}, regression "  # noqa: E731
                            f"{fmt(shown[case, 'observational_regression'])}, group means "
                            f"{fmt(shown[case, 'group_mean_difference'])}")
    prior = metrics["prior"]
    pct = lambda key: f"{100 * stats[key]:.0f} %"  # noqa: E731
    others = [c for c in CASES if c != "observed_mediator"]
    span = lambda metric: (f"{100 * min(rates[c][metric] for c in others):.0f}–"  # noqa: E731
                           f"{100 * max(rates[c][metric] for c in others):.0f} %")
    lines = f"""# Material for the ms1 report

Created by `python scripts/make_report_material.py` from the cluster results in `../results/`
(run `run2`). Every figure exists as `.png` and `.pdf`; tables are LaTeX (`booktabs` package) and
can be included with `\\input{{...}}`. `tab_results.md` is the results table for quick reading.

Note on the numbers: `tab_results.tex` holds the cluster evaluation (`../results/metrics.json`).
The box plot and the win rates use scores per dataset that were recomputed on a laptop with the same
checkpoint (`per_dataset_scores.csv`). The case-study files are identical on both machines; the 256
held-out prior datasets are not (the same seed gives slightly different datasets on another machine
and PyTorch version), so prior numbers differ in the third decimal.

## Figures

| File | Section | Suggested caption |
|---|---|---|
| `fig_prior_examples` | Exploration | {n_examples} datasets sampled from our prior. Left: the sampled causal graph with treatment `t` and outcome `y`. Middle: observational data, colored by treatment. Right: interventional outcomes of the query rows under do(t). Horizontal axis: first covariate. |
| `fig_case_studies_a`, `fig_case_studies_b` | Exploration | One dataset of each case study (first file of the Do-PFN case studies): observational data colored by treatment, and the interventional outcomes of the query rows. The causal graphs are not stored with the data; see Figure 3 of the Do-PFN paper. With a single column (treatment only) the horizontal axis is the row index. |
| `fig_case_studies_a_predictions`, `fig_case_studies_b_predictions` | Results, "where do they fail" | The same datasets with the predictions of the trained model (crosses). These are single datasets, not typical results: Front-Door and Confounder + Mediator are predicted well; in the Observed Confounder and Back-Door examples the treatment groups do not overlap in the covariate, and the predictions for the unobserved combination are far from the truth. |
| `fig_training` | Results, overfitting | Left: training and validation NLL during pre-training ({training['steps']} steps, {training['minutes']:.0f} min). Right: CATE error on the validation datasets, compared with the difference of group means. |
| `fig_nmse_boxplots` | Results | Normalized MSE per dataset (log scale) on held-out prior data and the six case studies, for the interventional outcome (top) and the CATE (bottom). Boxes: quartiles, line: median. |
| `fig_predictions` | Results | Predicted interventional outcomes of the trained model (crosses) on top of the true interventional data, for the datasets of the first figure. |
| `../results/case_study_metrics.png` | Results (alternative) | Mean normalized MSE with 95 % bootstrap intervals. The Observed Mediator panel is stretched by one dataset; prefer the box plot. |

## Tables

| File | Section | Content |
|---|---|---|
| `tab_prior_stats.tex` | Exploration, data quality | What the prior generates ({stats['n_datasets']} sampled datasets). |
| `tab_setup.tex` | Method | Model and training setup. |
| `tab_results.tex` | Results | Mean and median normalized MSE per dataset and model; best value in bold. |
| `tab_win_rates.tex` | Results | Share of datasets on which our model beats both baselines. |

## Key numbers for the text

Training
- {training['steps']} steps of {training['batch_size']} datasets in {training['minutes']:.0f} minutes on one GPU; the model has {n_parameters / 1e6:.2f} M parameters.
- Validation NLL falls from {training['val_nll_first']:.2f} to {training['val_nll_last']:.3f}; over the last 3000 steps it changes by {abs(training['val_nll_3000_before'] - training['val_nll_last']):.4f}.
- Training NLL at the end is {training['train_nll_last']:.2f}, validation {training['val_nll_last']:.2f}: no gap, so no overfitting. Every training step uses freshly sampled datasets, so the model never sees a dataset twice.
- Validation CATE MSE (normalized outcome): {training['val_cate_last']:.3f}, difference of group means: {training['val_cate_ref']:.3f}.

Held-out prior data ({prior['dopfn']['n_datasets']} datasets, cluster evaluation)
- Outcome NMSE: our model {fmt(prior['dopfn']['outcome_nmse'])}, regression {fmt(prior['observational_regression']['outcome_nmse'])}, group means {fmt(prior['group_mean_difference']['outcome_nmse'])}.
- CATE NMSE: our model {fmt(prior['dopfn']['cate_nmse'])}, regression {fmt(prior['observational_regression']['cate_nmse'])}, group means {fmt(prior['group_mean_difference']['cate_nmse'])}.
- Our model is better than both baselines on {100 * rates['prior']['outcome_nmse']:.0f} % of the datasets for the outcome and {100 * rates['prior']['cate_nmse']:.0f} % for the CATE.

Case studies (100 datasets each)
- Outcome: lowest mean error in five of six case studies (all except Observed Mediator).
- CATE: lowest mean error in four of six. Not on Unobserved Confounder ({fmt(metrics['unobserved_confounder']['dopfn']['cate_nmse'])} vs {fmt(metrics['unobserved_confounder']['group_mean_difference']['cate_nmse'])} for group means; the effect is not identifiable there) and not on Observed Mediator.
- Per dataset (`tab_win_rates.tex`): our model beats both baselines at once on {span('outcome_nmse')} of the datasets of a case study for the outcome and on {span('cate_nmse')} for the CATE (Observed Mediator: {100 * rates['observed_mediator']['outcome_nmse']:.0f} % and {100 * rates['observed_mediator']['cate_nmse']:.0f} %). The lower mean error therefore comes from fewer large errors, not from winning on every dataset.
- The datasets shown in `fig_case_studies_*` are the first file of each case study. Their outcome NMSE: {'; '.join(example(c) for c in CASES)}.
- Visible failure in those figures: in the Observed Confounder and Back-Door examples the treatment in the context is (almost) a threshold of the covariate, so the two groups do not overlap, and the query rows ask for the treatment that was never observed at that covariate value. There the model predicts roughly the outcome level of the queried treatment group in the context instead of following the covariate. Without overlap the data alone cannot separate the effect of the treatment from the effect of the covariate.
- Observed Mediator: the difference of group means is the right estimator (no confounding) and is best. Our mean error ({fmt(metrics['observed_mediator']['dopfn']['outcome_nmse'])}) comes from one dataset whose interventional outcomes are almost constant (range 0.002), so dividing by the squared range inflates the error; the median is {fmt(metrics['observed_mediator']['dopfn']['outcome_nmse_median'])} (group means {fmt(metrics['observed_mediator']['group_mean_difference']['outcome_nmse_median'])}).

What the prior generates ({stats['n_datasets']} datasets, `tab_prior_stats.tex`)
- A common cause of treatment and outcome exists in {pct('has_confounder')} of the datasets, and in {pct('hidden_confounder')} at least one is hidden.
- A mediator exists in {pct('has_mediator')}; in {pct('observed_mediator')} one is among the covariates.
- In {pct('no_effect')} there is no causal path from treatment to outcome (true effect 0).
- Overlap: in the median dataset {100 * stats['no_overlap'][2]:.0f} % of the context rows have an estimated propensity outside [0.05, 0.95] (logistic regression of the treatment on the covariates); in {pct('mostly_no_overlap')} of the datasets this holds for more than half of the rows. A likely reason (not tested): most datasets contain a covariate that is caused by the treatment (a mediator or another descendant), which separates the two groups.
- Treatment balance: the treated fraction is between {stats['treated_fraction'][0]:.2f} and {stats['treated_fraction'][4]:.2f} (median {stats['treated_fraction'][2]:.2f}).
- Data quality: {pct('one_group')} of the contexts contain only one treatment group, {pct('few_treated')} have a group below 5 %, and {pct('only_treatment')} have no covariates at all.

Preprocessing and representation (facts about the code)
- No `transformation` is passed to `load_data`; all preprocessing happens inside the model, so it is the same in training and at prediction time.
- Every feature column is standardized with the mean and standard deviation of the context rows and clipped to ±100. The outcome is standardized with the context mean and standard deviation, and predictions are mapped back to the original scale.
- The treatment is column 0 and gets an additional learned embedding, so the model can tell it apart from the covariates.
- Datasets of different size and shape: the number of rows is free; within a batch, missing feature columns are padded with NaN and masked in the attention between features. The prediction for a query row does not depend on the other query rows or on the order of the context rows.
- Train/test separation: training, validation and evaluation datasets come from different seed streams (training seed 0, validation stream 1, evaluation seed 1000); the case studies are never used for training.

Limits to state
- Results come from one training run with one seed.
- A fixed seed reproduces the data on one machine, not across machines or PyTorch versions.
- The mean normalized MSE is sensitive to datasets with a near-constant outcome; medians are in the table.
- CATE baselines of the paper (S/T/X-learner, DML, ...) and Dont-PFN are not part of ms1.
"""
    (out / "README.md").write_text(lines)


def main() -> None:
    """Create all report material."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=MS1 / "results")
    parser.add_argument("--out", type=Path, default=MS1 / "report")
    parser.add_argument("--n-prior-stats", type=int, default=1000)
    parser.add_argument("--limit", type=int, help="Quick test: only this many datasets per set.")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)

    checkpoint_path = args.results / "checkpoint_best.pt"
    model = Regressor.from_checkpoint(checkpoint_path)
    n_parameters = sum(p.numel() for p in model.model.parameters())
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    metrics = json.loads((args.results / "metrics.json").read_text())

    training = plot_training(args.results, args.out)
    training["batch_size"] = checkpoint["train_config"]["batch_size"]
    write_setup_table(checkpoint, n_parameters, training, args.out)
    stats = prior_statistics(args.limit * 10 if args.limit else args.n_prior_stats)
    write_prior_table(stats, args.out)
    n_examples = plot_examples(model, args.out)
    plot_case_studies(model, args.out)
    rows = score_all(model, args.results, args.out, args.limit)
    plot_boxplots(rows, args.out)
    rates = win_rates(rows)
    write_results_tables(metrics, rates, args.out)
    write_index(args.out, training, stats, metrics, rates, n_parameters, n_examples, rows)
    print(f"Report material written to {args.out}")


if __name__ == "__main__":
    main()
