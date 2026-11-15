"""Signature, loader-sample compatibility, overlay, and export checks."""

import inspect
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import networkx as nx
import numpy as np
import pytest

from dopfn.evaluation._synthetic import synthetic_graph, synthetic_sample
from dopfn.viz import show, show_case_graphs, show_metrics


def test_show_signature() -> None:
    params = inspect.signature(show).parameters
    assert list(params)[:2] == ["x", "outfile"]
    assert params["outfile"].default is None


def test_three_metric_comparison_uses_normalized_scores_and_marks_undefined_ate(tmp_path: Path) -> None:
    scores = {
        "observational_regression": {"outcome_mse": 100., "cate_mse": 200.,
            "outcome_nmse": 0.25, "cate_nmse": 0.5, "ate_relative_error": 0.2,
            "n_nmse_datasets": 4, "n_ate_datasets": 3},
        "group_mean_difference": {"outcome_nmse": 0.75, "cate_nmse": 1.,
            "ate_relative_error": None, "n_nmse_datasets": 4, "n_ate_datasets": 3},
    }
    outfile = tmp_path / "metrics.png"
    figure = show_metrics({"observed_confounder": scores}, outfile)
    try:
        assert [ax.get_xlabel() for ax in figure.axes] == ["MSE (CID)", "MSE (CATE)", "Rel. Err (ATE)"]
        assert [bar.get_width() for bar in figure.axes[0].patches] == [0.25, 0.75]
        assert [bar.get_width() for bar in figure.axes[1].patches] == [0.5, 1.]
        assert [bar.get_width() for bar in figure.axes[2].patches] == [0.2]
        assert "N/A" in [text.get_text() for text in figure.axes[2].texts]
        assert outfile.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    finally:
        plt.close(figure)


def test_case_graphs_keep_edges_and_use_input_visibility(tmp_path: Path) -> None:
    graph = nx.DiGraph([("t", "M"), ("M", "y"), ("U", "t"), ("U", "y")])
    graph.graph["input_variables"] = ["t", "U"]
    outfile = tmp_path / "graphs.png"
    figure = show_case_graphs({"frontdoor_criterion": graph}, outfile, columns=1)
    try:
        ax = figure.axes[0]
        assert len(ax.patches) == graph.number_of_edges()
        colors = dict(zip(graph.nodes, ax.collections[0].get_facecolors()))
        # Visibility follows the actual inputs: U is observed, M is not part of the input.
        np.testing.assert_allclose(colors["U"], to_rgba("#BF1818"))
        np.testing.assert_allclose(colors["M"], to_rgba("#D6D6D6"))
        assert len(ax.collections) == 1
        assert outfile.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    finally:
        plt.close(figure)


def test_show_saves_multiple_samples_and_overlay(tmp_path: Path) -> None:
    sample = synthetic_sample()
    outfile = tmp_path / "datasets.png"
    figure = show([sample, sample], outfile, graphs=[synthetic_graph(), None],
                  predictions=[sample[3], None])
    try:
        assert len(figure.axes) == 6
        assert "Prediction" in figure.axes[2].get_legend_handles_labels()[1]
        assert "Graph unavailable" in [text.get_text() for text in figure.axes[3].texts]
        assert outfile.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    finally:
        plt.close(figure)


def test_show_accepts_items_from_load_data(tmp_path: Path) -> None:  # req V2, V4
    from dopfn.data import load_data

    samples = list(load_data("prior", n_datasets=2, seed=3, n_train=40, n_test=20))
    outfile = tmp_path / "prior.png"
    figure = show(samples, outfile)
    try:
        assert outfile.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    finally:
        plt.close(figure)


def test_show_rejects_misaligned_overlay() -> None:
    sample = synthetic_sample()
    with pytest.raises(ValueError, match="align"):
        show([sample], predictions=[sample[3][1:]])


def test_combined_plot_keeps_all_scm_nodes_and_directed_edges(tmp_path: Path) -> None:
    graph = synthetic_graph()
    outfile = tmp_path / "scm.png"
    figure = show([synthetic_sample()], outfile, graphs=[graph])
    try:
        assert len(figure.axes) == 3
        ax = figure.axes[0]
        assert {text.get_text() for text in ax.texts} == set(graph.nodes)
        assert len(ax.patches) == graph.number_of_edges()
        assert outfile.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    finally:
        plt.close(figure)
