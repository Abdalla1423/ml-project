"""Extend a Do-PFN-prior causal graph with a binary treatment node ``t``.

Do-PFN-prior samples DAGs whose target node is ``y`` but has no designated treatment. We add a new
node ``t`` with at least one directed path to ``y`` and (if possible) one parent that confounds
``t`` and ``y``.
"""

from collections.abc import Sequence

import networkx as nx
import torch

TREATMENT = "t"
OUTCOME = "y"


def _randint(low: int, high: int, generator: torch.Generator) -> int:
    """Uniform integer in ``[low, high]`` (both inclusive)."""
    return int(torch.randint(low, high + 1, (1,), generator=generator).item())


def _sample_without_replacement(
    items: Sequence[str], k: int, generator: torch.Generator
) -> list[str]:
    """Sample ``k`` distinct items, in a deterministic order for a given generator state."""
    idx = torch.randperm(len(items), generator=generator)[:k].tolist()
    return [items[i] for i in idx]


def add_treatment_node(
    graph: nx.DiGraph,
    generator: torch.Generator,
    max_children: int = 3,
) -> nx.DiGraph:
    """Return a copy of ``graph`` with a treatment node ``t`` added.

    ``t`` gets between 1 and ``max_children`` children, chosen among the ancestors of ``y`` and
    ``y`` itself, so there is always a causal path ``t -> ... -> y``. If the chosen children share
    ancestors, one of them is made a parent of ``t`` (a confounder). The new edges cannot create a
    cycle: the parent is an ancestor of every child.

    Args:
        graph: Causal DAG from Do-PFN-prior with node attributes ``dimension``, ``n_hidden``,
            ``n_visible`` and an outcome node ``y``.
        generator: Torch generator, the only source of randomness.
        max_children: Maximum number of children of ``t``.

    Returns:
        The extended DAG. The input graph is not modified.

    Raises:
        ValueError: If ``t`` already exists or ``y`` is missing.
    """
    if TREATMENT in graph.nodes:
        raise ValueError(f"Treatment node {TREATMENT!r} already exists.")
    if OUTCOME not in graph.nodes:
        raise ValueError(f"Outcome node {OUTCOME!r} does not exist.")

    extended = graph.copy()

    # Sorted, because set iteration order of strings changes between Python processes
    # (hash randomization) and would break reproducibility for a fixed seed.
    candidates = sorted(nx.ancestors(extended, OUTCOME)) + [OUTCOME]
    num_children = _randint(1, min(max_children, len(candidates)), generator)
    children = _sample_without_replacement(candidates, num_children, generator)

    common_ancestors = set(nx.ancestors(extended, children[0]))
    for child in children[1:]:
        common_ancestors &= nx.ancestors(extended, child)
    parents = sorted(common_ancestors)

    extended.add_node(TREATMENT, dimension=1, n_hidden=0, n_visible=1)
    if parents:
        extended.add_edge(parents[_randint(0, len(parents) - 1, generator)], TREATMENT)
    for child in children:
        extended.add_edge(TREATMENT, child)

    if not nx.is_directed_acyclic_graph(extended):
        raise RuntimeError("Extended graph contains a cycle.")
    if not nx.has_path(extended, TREATMENT, OUTCOME):
        raise RuntimeError("Treatment has no causal path to the outcome.")
    return extended


def get_pre_treatment_variables(graph: nx.DiGraph) -> list[str]:
    """Names of variables that are not descendants of ``t`` (excluding ``t`` and ``y``).

    Args:
        graph: Extended graph containing ``t`` and ``y``.

    Returns:
        Sorted variable names.
    """
    post = nx.descendants(graph, TREATMENT)
    return sorted(set(graph.nodes) - post - {TREATMENT, OUTCOME})


def get_input_variables(graph: nx.DiGraph) -> list[str]:
    """Variables used as covariates: visible ancestors of ``y``, in topological order.

    ``t`` is included if it is an ancestor of ``y`` (always true for an extended graph); the
    caller puts it in column 0. Variables with no visible features are skipped.

    TODO(team): D1 - Do-PFN-prior uses all visible nodes as features (also descendants of ``y``);
    we only use ancestors of ``y``. Confirm with the team.

    Args:
        graph: Extended graph containing ``t`` and ``y``.

    Returns:
        Variable names in topological order.
    """
    ancestors = nx.ancestors(graph, OUTCOME)
    return [
        v
        for v in nx.topological_sort(graph)
        if v in ancestors and graph.nodes[v]["n_visible"] > 0
    ]
