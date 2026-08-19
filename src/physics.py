"""
Physics-informed loss term for the solid-mechanics GNN-PINN.

Physical law enforced (energy-force work conjugacy):
    For a quasi-static elastic body loaded through a single generalized
    displacement `delta` (here: the prescribed top-edge displacement step),
    the total strain energy Psi(delta) is a potential, and the work-conjugate
    reaction force is its derivative:

        F(delta) = dPsi/ddelta

    This is the structural-mechanics analogue of the original repo's
    Gibbs-free-energy equilibrium penalty (Gv == Gl) -- a different physical
    law, from a different field, expressed as an automatic-differentiation
    residual instead of an algebraic one, because here the two predicted
    quantities (Psi, F) are related through a derivative rather than a
    direct equality.

We compute dPsi_pred/ddelta via `torch.autograd.grad` with `create_graph=True`
so the residual itself is differentiable and can be backpropagated through
during PINN finetuning.
"""
from __future__ import annotations

import torch


def energy_force_consistency(model, x, edge_index, edge_attr, batch, delta):
    """Returns (residual_loss, dpsi_ddelta, psi_pred, force_pred).

    `delta` must NOT already require grad from an outer graph in a way that
    conflicts; we clone+detach it and turn on requires_grad locally so the
    derivative is taken cleanly w.r.t. this call's load-step input.
    """
    delta = delta.clone().detach().requires_grad_(True)
    psi_pred, force_pred = model(x, edge_index, edge_attr, batch, delta)

    dpsi_ddelta = torch.autograd.grad(
        outputs=psi_pred, inputs=delta,
        grad_outputs=torch.ones_like(psi_pred),
        create_graph=True, retain_graph=True,
    )[0]

    residual = force_pred - dpsi_ddelta
    residual_loss = torch.mean(residual ** 2)
    return residual_loss, dpsi_ddelta, psi_pred, force_pred


def monotonic_energy_penalty(dpsi_ddelta: torch.Tensor) -> torch.Tensor:
    """Soft physical prior: under monotonically increasing prescribed
    displacement away from the undeformed state, a stable elastic material's
    strain energy should be non-decreasing, i.e. dPsi/ddelta >= 0. Penalizes
    violations; weight this small relative to the data-fit terms."""
    return torch.mean(torch.relu(-dpsi_ddelta) ** 2)
