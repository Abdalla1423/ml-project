"""Extend an existing SCM graph with a treatment node."""

import random
import networkx as nx

def get_pre_treatment_variables(graph):
    """
    Extract variables that are not descendants of treatment.

    Args:
        graph (networkx.DiGraph):
            Extended causal graph containing 't' and 'y'.

    Returns:
        list:
            Names of pre-treatment variables.
    """

    treatment_descendants = nx.descendants(graph, "t")
    pre_treatment = (set(graph.nodes) - treatment_descendants - {"t", "y"})

    return sorted(pre_treatment)

def get_input_variables(graph):
    topo = list(nx.topological_sort(graph))
    y_idx = topo.index("y")
    input_vars = [var for var in topo[:y_idx] 
                  if graph.nodes[var]["n_visible"] > 0 and nx.has_path(graph, var, "y")]
    return input_vars

def add_treatment_node(graph, rng=None):
    """
    Add a treatment node 't' to an existing causal DAG.

    The treatment node is randomly connected to existing variables.
    At least one causal path from treatment 't' to outcome 'y'
    is guaranteed.

    Args:
        graph (networkx.DiGraph):
            Original causal DAG.

    Returns:
        networkx.DiGraph:
            Extended causal DAG containing treatment node 't'.
    """
    if rng is None:
        rng = random
    # Copy the original graph
    extended_graph = graph.copy()

    # Validate input graph
    if "t" in extended_graph.nodes:
        raise ValueError("Treatment node 't' already exists.")

    if "y" not in extended_graph.nodes:
        raise ValueError("Outcome node 'y' does not exist.")

    # Find candidate children of treatment
    # Select all ancestors of the outcome node 'y' + 'y' itself
    # ensuring that connecting 't' to any candidate creates
    # a causal path from treatment to outcome.
    candidates = list(nx.ancestors(extended_graph, "y"))
    candidates.append("y")

    # Randomly select treatment children
    num_children = rng.randint(1, min(3, len(candidates)))
    children = rng.sample(candidates, num_children)

    # Find common ancestors of selected children
    common_ancestors = set(nx.ancestors(extended_graph, children[0]))

    for child in children[1:]:
        common_ancestors &= nx.ancestors(extended_graph, child)

    parents = list(common_ancestors)

    # Add treatment node
    extended_graph.add_node("t", dimension=1, n_hidden=0,n_visible=1)

    # Randomly connect a parent, if available
    if parents:
        parent = rng.choice(parents)
        extended_graph.add_edge(parent, "t")

    # Connect treatment to selected children
    for child in children:
        extended_graph.add_edge("t", child)

    # Validate the extended graph
    if not nx.is_directed_acyclic_graph(extended_graph):
        raise RuntimeError("Generated graph contains a cycle.")

    if not nx.has_path(extended_graph, "t", "y"):
        raise RuntimeError(
            "Treatment has no causal path to outcome."
        )

    return extended_graph
