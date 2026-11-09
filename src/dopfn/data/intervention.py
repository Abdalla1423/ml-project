
"""
Build and propagate an SCM with an explicit treatment variable.

This module extends the original SCM propagation logic from
dopfnprior/scm/scm.py, specifically the SCM.propagate() method.

The original propagation logic is preserved for all non-treatment
variables, while the treatment variable ('t') can be either
observationally generated as a binary variable or fixed by an intervention.
"""

import torch
from dopfnprior.scm.scm_builder import SCMBuilder


def build_extended_scm(
    extended_graph,
    scm_samplers,
    generator=None,
):
    """
    Build an SCM from an extended causal graph.

    This function uses the original SCMBuilder from dopfnprior
    to construct an SCM containing the treatment variable.

    Args:
        extended_graph:
            Causal graph containing the treatment node 't'.
        scm_samplers:
            Sampling configurations used by the original SCMBuilder.
        generator:
            Optional PyTorch random number generator.

    Returns:
        SCM:
            Constructed structural causal model.
    """

    builder = SCMBuilder(
        graph=extended_graph,
        **scm_samplers,
    )

    scm = builder.sample(generator)

    return scm

@torch.no_grad()
def propagate_with_treatment(scm, treatment_value=None, sample_shape=(1,), noise=None, fixed_values=None):
    """
    Args:
        scm:
            A structural causal model containing node 't'.
        treatment_value:
            If None, treatment is generated observationally.
            Otherwise, treatment is fixed to the given value (0 or 1).
        sample_shape:
            Shape of the generated samples.
        noise:
            Previously generated exogenous noise.
            If None, new noise is generated.
        fixed_values:
            Dictionary of variables whose values should be fixed.

    Returns:
        dict:
            Generated values for all variables.
    """

    if fixed_values is None:
        fixed_values = {}

    # Generate noise if necessary
    if noise is None:
        scm.sample_noise(sample_shape)
        noise = scm._sampled_noise

    # Store generated variable values
    xs = {}

    # Propagate variables in topological order
    for v in scm._topo:

        # Handle treatment separately
        if v == "t":
            # Intervention: fix treatment to the specified value
            if treatment_value is not None:
                x = torch.full_like(noise[v], treatment_value)

            # Observation: generate binary treatment
            else:
                x = noise[v]
                parents_feat = {p: xs[p] for p in scm._parents[v]}
                if len(parents_feat) > 0:
                    mech = scm.mechanisms[v]
                    x = mech(parents_feat) + x
                treatment_prob = torch.sigmoid(x)
                x = torch.bernoulli(treatment_prob)

        elif v in fixed_values:
            x = fixed_values[v].to(device=noise[v].device, dtype=noise[v].dtype)
            if x.shape != noise[v].shape:
                raise ValueError(
                    f"Shape mismatch for '{v}': "
                    f"expected {noise[v].shape}, got {x.shape}"
                )
        
        else:
            # Use the original SCM propagation logic
            x = noise[v]
            parents_feat = {p: xs[p] for p in scm._parents[v]}
            if len(parents_feat) > 0:
                mech = scm.mechanisms[v]
                x = mech(parents_feat) + x

            if v in scm.post_activations:
                x = scm.post_activations[v](x)

        xs[v] = x

    return xs
