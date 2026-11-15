"""Render SCMs, causal data, and CID/CATE/ATE metric comparisons."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from dopfn.evaluation._sample import unpack_sample
from dopfn.evaluation.metrics import _vector


_MODEL_LABELS = {
    "dopfn": "Do-PFN",
    "observational_regression": "Observational regression",
    "group_mean_difference": "Group mean difference",
}
_MODEL_COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")


def show_case_graphs(
    graphs: dict[str, nx.DiGraph],
    outfile: str | Path | None = None,
    *,
    columns: int = 6,
) -> Figure:
    """把多个 case study 的真实 DAG 合成一张图，按输入列区分可见和隐藏变量。"""
    titles = {
        "observed_confounder": "Observed\nConfounder",
        "observed_mediator": "Observed\nMediator",
        "confounder_mediator": "Confounder +\nMediator",
        "unobserved_confounder": "Unobserved\nConfounder",
        "frontdoor_criterion": "Front-Door\nCriterion",
        "backdoor_criterion": "Back-Door\nCriterion",
    }
    palette = {"t": "#ED752A", "y": "#2143B4", "observed": "#BF1818", "hidden": "#D6D6D6"}
    rows = (len(graphs) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(3.3 * columns, 3.2 * rows + 0.7),
                             squeeze=False)
    fig.subplots_adjust(left=0.025, right=0.975, top=0.89, bottom=0.16,
                        wspace=0.12, hspace=0.62)
    for ax, (name, graph) in zip(axes.flat, graphs.items()):
        observed = set(graph.graph["input_variables"])
        covariates = sorted(set(graph) - {"t", "y"}, key=lambda node: (node not in observed, node))
        positions = {"t": (-1, -0.55), "y": (1, -0.55)}
        positions.update({node: ((i - (len(covariates) - 1) / 2) * 1.35, 0.8)
                          for i, node in enumerate(covariates)})
        mediators = [node for node in covariates
                     if nx.has_path(graph, "t", node) and nx.has_path(graph, node, "y")]
        if len(covariates) == 2 and len(mediators) == 1:
            # 中介在上、混杂变量在下；这里只调整位置，所有箭头直接来自保存的 DAG。
            positions = {"t": (-1, 0), "y": (1, 0), mediators[0]: (0, 0.9)}
            positions.update({node: (0, -0.9) for node in covariates if node not in mediators})
        colors = [palette[node] if node in ("t", "y") else
                  palette["observed" if node in observed else "hidden"] for node in graph]
        borders = ["#A5A5A5" if color == palette["hidden"] else "#111111" for color in colors]
        nx.draw_networkx_nodes(graph, positions, ax=ax, node_size=780,
                               node_color=colors, edgecolors=borders, linewidths=1.15)
        nx.draw_networkx_edges(graph, positions, ax=ax, node_size=780,
                               arrows=True, arrowstyle="-|>", arrowsize=18,
                               width=1.25, edge_color="#111111")
        labels = {node: (f"${node[0]}_{{{node[1:]}}}$" if node[1:].isdigit() else f"${node}$")
                  for node in graph}
        texts = nx.draw_networkx_labels(graph, positions, labels=labels, ax=ax,
                                        font_size=17, font_color="white", font_weight="bold")
        for text in texts.values():
            text.set_math_fontfamily("stix")
        ax.set_title(titles.get(name, name.replace("_", " ").title()),
                     fontfamily="STIXGeneral", fontstyle="italic", fontweight="bold",
                     fontsize=18, pad=15)
        ax.set(xlim=(-1.5, 1.5), ylim=(-1.35, 1.35), aspect="equal")
        ax.set_axis_off()
    for ax in list(axes.flat)[len(graphs):]:
        ax.set_visible(False)
    legend = [Line2D([], [], linestyle="", marker="o", markersize=10,
                     markerfacecolor=palette[key], markeredgecolor="#555555", label=label)
              for key, label in (("t", "Treatment"), ("y", "Outcome"),
                                 ("observed", "Observed covariate"), ("hidden", "Hidden variable"))]
    fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.5, 0.045),
               ncol=4, frameon=False, fontsize=11)
    fig.text(0.5, 0.025, "Stored SCM node names and input visibility · published sample 1 per case",
             ha="center", fontsize=9, color="#666666")
    if outfile is not None:
        fig.savefig(outfile, dpi=300, bbox_inches="tight", facecolor="white")
    return fig


def _draw_scm(graph: nx.DiGraph, ax: Any) -> None:
    """按因果方向分层排布，保留 A 生成的所有节点和边。"""
    # 从终点反向分层，让直接指向 Y 的节点并排，避免长边穿过 x0、x4。
    layers = list(reversed(list(nx.topological_generations(graph.reverse(copy=False)))))
    positions = {
        node: (depth, (0 if depth == 0 else (len(layer) - 1) / 2) - row)
        for depth, layer in enumerate(layers)
        for row, node in enumerate(layer)
    }
    colors = ["#D55E00" if node == "t" else "#009E73" if node == "y"
              else "#dcecf7" for node in graph.nodes]
    nx.draw_networkx_nodes(graph, positions, ax=ax, node_color=colors, node_size=1000)
    # 弧线避免 U→Y 等跨层边与 U→T→Y 的连线重叠。
    # 优先使用直线；跨列长边被中间节点挡住时才向外绕。
    for source, target in graph.edges:
        source_x, source_y = positions[source]
        target_x, target_y = positions[target]
        curvature = 0.0
        blocked = any(
            source_x < x < target_x
            and abs(y - (source_y + (target_y - source_y)
                         * (x - source_x) / (target_x - source_x))) < 0.45
            for node, (x, y) in positions.items() if node not in (source, target)
        )
        if blocked:
            curvature = -0.65 if source_y >= target_y else 0.65
        nx.draw_networkx_edges(
            graph, positions, ax=ax, edgelist=[(source, target)],
            arrows=True, arrowstyle="-|>", arrowsize=16,
            edge_color="#455b70", node_size=1000,
            connectionstyle=f"arc3,rad={curvature}",
        )
    nx.draw_networkx_labels(graph, positions, ax=ax, font_size=9)
    ax.margins(0.18)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_axis_off()


def show(
    x: list[Any],
    outfile: str | Path | None = None,
    *,
    graphs: Sequence[nx.DiGraph | None] | None = None,
    predictions: Sequence[np.ndarray | None] | None = None,
    feature: int = 1,
    title: str | None = None,
) -> Figure:
    """Return one row of plots per loader sample and optionally save the figure.

    Graphs and outcome prediction vectors follow the order of the samples.
    `feature` selects the plotted column of X. With treatment-only data,
    the horizontal axis is the row index.
    """
    if not x:
        raise ValueError("show requires at least one dataset.")
    if graphs is not None and len(graphs) != len(x):
        raise ValueError("Pass one graph or None per dataset.")
    if predictions is not None and len(predictions) != len(x):
        raise ValueError("Pass one prediction vector or None per dataset.")
    if feature < 1:
        raise ValueError("feature must select a covariate column (index >= 1).")

    samples = [unpack_sample(sample) for sample in x]
    for X_ob, _, _, _, _ in samples:
        if X_ob.shape[1] > 1 and feature >= X_ob.shape[1]:
            raise ValueError(f"Feature column {feature} is missing.")
    overlays = [None] * len(x)
    if predictions is not None:
        for i, values in enumerate(predictions):
            if values is not None:
                overlays[i] = _vector(values, "predictions")
                if len(overlays[i]) != len(samples[i][3]):
                    raise ValueError("Each prediction vector must align with y_in.")

    fig, axes = plt.subplots(
        len(x), 3, figsize=(11.4, 3.6 * len(x)), squeeze=False, layout="constrained"
    )
    if title:
        fig.suptitle(title, fontweight="bold")
    for i, (X_ob, y_ob, X_in, y_in, _) in enumerate(samples):
        graph_ax, obs_ax, in_ax = axes[i]
        graph = None if graphs is None else graphs[i]
        graph_ax.set_title(f"Dataset {i + 1} · causal graph", loc="left")
        if graph is None:
            graph_ax.text(0.5, 0.5, "Graph unavailable", ha="center", va="center",
                          transform=graph_ax.transAxes, color="#596774")
        else:
            _draw_scm(graph, graph_ax)
        graph_ax.set_axis_off()

        has_covariates = X_ob.shape[1] > 1
        x_ob = X_ob[:, feature] if has_covariates else np.arange(len(X_ob))
        x_in = X_in[:, feature] if has_covariates else np.arange(len(X_in))
        obs_ax.sharex(in_ax)
        obs_ax.sharey(in_ax)
        for ax, coords, X, y, heading in (
            (obs_ax, x_ob, X_ob, y_ob, "Observational"),
            (in_ax, x_in, X_in, y_in, "Interventional"),
        ):
            for t, color in enumerate(("#0072B2", "#B44A00")):
                rows = X[:, 0] == t
                if rows.any():
                    # 观察数据是按 T 分组；干预数据对应 do(T=t)。
                    group = f"do(T={t})" if ax is in_ax else f"T={t}"
                    ax.scatter(coords[rows], y[rows], s=19, alpha=0.65,
                               color=color, label=f"{group} (n={rows.sum()})")
            ax.set_title(heading, loc="left")
            ax.set_xlabel(f"Covariate (column {feature})" if has_covariates else "Row index")
            ax.set_ylabel("Outcome Y")
            ax.grid(alpha=0.18)
            ax.set_axisbelow(True)
        if overlays[i] is not None:
            in_ax.scatter(x_in, overlays[i], s=25, color="#222222", marker="x",
                          linewidths=0.8, label="Prediction")
        for ax in (obs_ax, in_ax):
            ax.legend(frameon=False, fontsize=8)

    if outfile is not None:
        fig.savefig(outfile, dpi=180, bbox_inches="tight")
    return fig


def show_metrics(
    results: dict[str, dict[str, dict[str, float | None]]],
    outfile: str | Path | None = None,
    *,
    title: str | None = None,
) -> Figure:
    """三行分别比较 CID 的归一化 MSE、CATE 的归一化 MSE 和 ATE 相对误差。"""
    metrics = (("outcome_nmse", "MSE (CID)", "n_nmse_datasets"),
               ("cate_nmse", "MSE (CATE)", "n_nmse_datasets"),
               ("ate_relative_error", "Rel. Err (ATE)", "n_ate_datasets"))
    fig, axes = plt.subplots(3, len(results), figsize=(3.1 * len(results) + 2.5, 8.3),
                             squeeze=False, layout="constrained")
    if title:
        fig.suptitle(title, fontweight="bold")
    for column, (dataset, scores) in enumerate(results.items()):
        names = sorted(scores, key=lambda name: name != "dopfn")
        labels = [_MODEL_LABELS.get(name, name) for name in names]
        colors = [_MODEL_COLORS[i % len(_MODEL_COLORS)] for i in range(len(names))]
        for row, (metric, heading, count_key) in enumerate(metrics):
            ax = axes[row, column]
            values = np.array([np.nan if scores[name][metric] is None else scores[name][metric]
                               for name in names], dtype=float)
            positions = np.arange(len(names))
            valid = np.isfinite(values)
            ax.barh(positions[valid], values[valid], height=0.58,
                    color=np.array(colors)[valid])
            for i, name in enumerate(names):
                low, high = scores[name].get(f"{metric}_ci95_low"), scores[name].get(f"{metric}_ci95_high")
                if low is not None and high is not None and np.isfinite([low, high]).all():
                    ax.hlines(i, low, high, color="#333333", linewidth=1.4)
                    ax.vlines([low, high], i - 0.09, i + 0.09, color="#333333", linewidth=1.4)
                if not valid[i]:
                    ax.text(0.04, i, "N/A", transform=ax.get_yaxis_transform(), va="center", fontsize=9)
            ax.set_yticks(positions, labels if column == 0 else [""] * len(names))
            ax.set_ylim(len(names) - 0.5, -0.5)
            ax.set_xlim(left=0)
            ax.margins(x=0.25)
            ax.set_xlabel(heading)
            count = scores[names[0]][count_key]
            ax.text(0.98, 0.97, f"SCMs: {int(count)}", transform=ax.transAxes,
                    ha="right", va="top", fontsize=8, color="#666666")
            if row == 0:
                ax.set_title(dataset.replace("_", " ").title(), fontsize=11)
            ax.grid(axis="x", alpha=0.18)
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)
    fig.supxlabel("MSE normalized by [max(y_in) − min(y_in)]² · Mean across SCMs · 95% bootstrap CI",
                  fontsize=9)
    if outfile is not None:
        fig.savefig(outfile, dpi=300, bbox_inches="tight")
    return fig
