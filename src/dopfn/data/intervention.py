"""Propagate a Do-PFN-prior SCM with a binary treatment node, observationally or under do(t).

Adapted from ``SCM.propagate`` in Do-PFN-prior (``Do-PFN-prior/dopfnprior/scm/scm.py``).
All non-treatment nodes follow the upstream equations ``x_v = g_v(f_v(pa(v)) + eps_v)``. The
treatment node is either

* observational: ``t = 1[u_t < sigmoid(f_t(pa(t)) + eps_t)]``, a Bernoulli draw that is
  deterministic given the exogenous noise ``(eps_t, u_t)``, or
* intervened: ``do(t = value)``, which cuts the incoming edges of ``t``.

Because the treatment draw is part of the exogenous noise, the same noise can be propagated
observationally and under ``do(0)`` / ``do(1)`` to get factual and interventional values of the
same unit.
"""

from dataclasses import dataclass

import torch
from dopfnprior.scm.scm import SCM

from dopfn.data.graph_extension import TREATMENT


@dataclass
class Exogenous:
    """Exogenous noise for ``n`` units.

    Attributes:
        noise: Per-node additive noise, ``noise[v]`` has shape ``(n, dimension_v)``.
        u_t: Uniform draws for the Bernoulli treatment assignment, shape ``(n, 1)``.
    """

    noise: dict[str, torch.Tensor]
    u_t: torch.Tensor


def sample_exogenous(scm: SCM, n: int, generator: torch.Generator) -> Exogenous:
    """Sample exogenous noise for ``n`` units.

    Args:
        scm: SCM containing the node ``t``.
        n: Number of units (rows).
        generator: Torch generator, the only source of randomness.

    Returns:
        The exogenous noise.
    """
    scm.sample_noise((n,), generator=generator)
    noise = dict(scm._sampled_noise)
    u_t = torch.rand((n, 1), generator=generator, dtype=noise[TREATMENT].dtype)
    return Exogenous(noise=noise, u_t=u_t)


@torch.no_grad()
def propagate(
    scm: SCM,
    exogenous: Exogenous,
    do_t: float | torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Compute all node values from the exogenous noise.

    Args:
        scm: SCM containing the node ``t``.
        exogenous: Noise from ``sample_exogenous``.
        do_t: ``None`` for observational propagation. Otherwise the intervened treatment value,
            either a scalar (same value for all units) or a tensor of shape ``(n, 1)``.

    Returns:
        Values of every node, ``values[v]`` has shape ``(n, dimension_v)``.
    """
    noise = exogenous.noise
    values: dict[str, torch.Tensor] = {}
    for v in scm._topo:
        parent_values = {p: values[p] for p in scm._parents[v]}
        if v == TREATMENT:
            if do_t is not None:
                t = torch.as_tensor(do_t, dtype=noise[v].dtype, device=noise[v].device)
                x = t.expand_as(noise[v]).clone()
            else:
                logit = noise[v]
                if parent_values:
                    logit = scm.mechanisms[v](parent_values) + logit
                x = (exogenous.u_t < torch.sigmoid(logit)).to(noise[v].dtype)
        else:
            x = noise[v]
            if parent_values:
                x = scm.mechanisms[v](parent_values) + x
            if v in scm.post_activations:
                x = scm.post_activations[v](x)
        values[v] = x
    return values
