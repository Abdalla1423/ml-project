"""SCM prior with a designated binary treatment and do(t) interventions.

Extends Do-PFN-prior (https://github.com/oossen/Do-PFN-prior). 
"""

import random
import networkx as nx
import torch
from typing import Optional

from dopfnprior.configs.default_config import prior_config
from dopfnprior.dataloaders.observational_dataloader import ObservationalDataLoader
from .graph_extension import add_treatment_node, get_pre_treatment_variables, get_input_variables
from .intervention import build_extended_scm, propagate_with_treatment


def create_prior_scm(seed=None, config=None):
    """
    Create an extended SCM from the Do-PFN prior.

    The function first samples an original SCM, adds a treatment
    variable to its causal graph, and builds a new SCM.

    It also extracts the pre-treatment covariates.

    Args:
        seed:
            Random seed used for sampling the original SCM.
        config:
            Prior configuration. Uses the default configuration
            if None.

    Returns:
        tuple:
            extended_scm:
                The SCM containing the treatment variable.
            extended_graph:
                The extended causal graph.
            pre_treatment_vars:
                Names of pre-treatment covariates.
    """

    # Use the default prior configuration if none is provided
    if config is None:
        config = prior_config

    if seed is None:
        seed = 42

    # Generate an original SCM
    loader = ObservationalDataLoader(num_steps=1, batch_size=1, prior_config=config, seed=seed)
    batch = loader.batch_function()
    original_scm = batch["scm"]
    original_graph = original_scm.dag

    graph_rng = random.Random(seed)
    # Extend the original graph with a treatment variable
    extended_graph = add_treatment_node(original_graph, rng=graph_rng)

    # Build an extended SCM
    extended_scm = build_extended_scm(extended_graph=extended_graph, scm_samplers=loader.scm_samplers, generator=loader.generator)

    # Identify pre-treatment variables
    pre_treatments = get_pre_treatment_variables(extended_graph)
    input_vars = get_input_variables(extended_graph)

    return extended_scm, extended_graph, pre_treatments, input_vars

def extract_visible_data(data, graph):
    visible_data = {}
    for var, values in data.items():
        n_visible = graph.nodes[var]["n_visible"]
        visible_data[var] = values[..., :n_visible]
    return visible_data

def build_input_target(data, graph, input_vars):
    visible_data = extract_visible_data(data, graph)

    # Treatment is the first input feature
    features = [visible_data["t"]]
    for var in input_vars:
        if var != "t":
            features.append(visible_data[var])

    y_visible = visible_data["y"]
    target = y_visible[..., :1]
    y_remaining = y_visible[..., 1:]
    if y_remaining.shape[-1] > 0:
        features.append(y_remaining)

    X = torch.cat(features, dim=-1)
    return X, target


def generate_observational_data(scm, pre_treatments, input_vars, n_samples):
    """
    Generate observational data from an extended SCM.

    Returns:
        Xob: [n_samples, 1 + visible features]
        yob: [n_samples, 1]
    """

    data = propagate_with_treatment(scm=scm, treatment_value=None, sample_shape=(n_samples,))
    Xob, yob = build_input_target(data, scm.dag, input_vars)
    return Xob, yob



def generate_interventional_data(scm, pre_treatments, input_vars, n_samples):
    """
    Generate interventional query data from an extended SCM.

    Treatment is randomly assigned to 0 or 1 for each sample.

    Returns:
        Xin: [n_samples, 1 + visible features]
        yin: [n_samples, 1]
    """

    scm.sample_noise((n_samples,))
    noise = scm._sampled_noise

    treatment = torch.randint(0, 2, (n_samples, 1), device=noise["t"].device).to(noise["t"].dtype)

    control_data = propagate_with_treatment(scm=scm, treatment_value=0, noise=noise)
    treated_data = propagate_with_treatment(scm=scm, treatment_value=1, noise=noise)
    observational_data = propagate_with_treatment(scm=scm, treatment_value=None, noise=noise)

    mask = treatment[:, 0] == 1
    selected_data = {}

    for var in scm.dag.nodes:
        selected_data[var] = torch.where(mask.unsqueeze(-1), treated_data[var], control_data[var])
    selected_data["t"] = treatment

    Xin, yin = build_input_target(selected_data, scm.dag, input_vars)
    y_in_ob = extract_visible_data(observational_data, scm.dag)["y"][..., :1]

    return Xin, yin, y_in_ob


def compute_cate(scm, pre_treatments, input_vars, Xin, n_mc=100):
    """
    Estimate ground-truth CATE using Monte Carlo sampling.

    Args:
        scm:
            Extended structural causal model.
        pre_treatments:
            Names of pre-treatment variables.
        Xin:
            Query features. Treatment must be the first column.
        n_mc:
            Number of Monte Carlo repetitions.

    Returns:
        cate:
            Estimated CATE for each query sample.
            Shape: [n_samples, 1]
    """

    n_samples = Xin.shape[0]
    fixed_values = {}
    start = 1  # Skip the treatment
    variable_pos = {}

    for var in input_vars:
        if var == "t":
            continue
        dim = scm.dag.nodes[var]["n_visible"]
        end = start + dim
        variable_pos[var] = (start, end)

    for var in pre_treatments:
        if var not in variable_pos:
            continue

        start, end = variable_pos[var]
        covariates = Xin[:, start:end]
        # [n_samples, d] -> [n_samples, n_mc, d]
        fixed_values[var] = (covariates.unsqueeze(1).expand(-1, n_mc, -1))

    scm.sample_noise((n_samples, n_mc))
    noise = scm._sampled_noise

    control_data = propagate_with_treatment(scm=scm, treatment_value=0, noise=noise, fixed_values=fixed_values)
    treated_data = propagate_with_treatment(scm=scm, treatment_value=1, noise=noise, fixed_values=fixed_values)

    y_control = control_data["y"][..., :1]
    y_treated = treated_data["y"][..., :1]

    individual_effects = y_treated - y_control

    cate = individual_effects.mean(dim=1)

    return cate

def generate_prior_samples(n_train=None, n_test=None, seed=None, config=None, n_mc=100):
    base_seed = 42 if seed is None else seed
    iteration = 0

    while True:
        current_seed = base_seed + iteration
        scm, graph, pre_treatments, input_vars = create_prior_scm(seed, config)
        Xob, yob = generate_observational_data(scm, pre_treatments, input_vars, n_samples=n_train if n_train is not None else 1000)
        Xin, yin, y_in_ob = generate_interventional_data(scm, pre_treatments, input_vars, n_samples=n_test if n_test is not None else 1000)
        cate = compute_cate(scm, pre_treatments, input_vars, Xin, n_mc=n_mc)
        yield Xob, yob, Xin, yin, y_in_ob, cate
        iteration += 1

