"""Uniform random transport plan for AB-04 (D-AB04).

Replaces Sinkhorn OT solver with uniform coupling.
P = torch.ones((n_spots, n_cells)) / n_cells

No optimal transport optimization applied.
Each spot has uniform probability over all cells.
"""

from __future__ import annotations

import torch
from torch import Tensor

from tglue.deconv.ot_solver import TransportPlan


class UniformTransport:
    """Uniform random transport plan (AB-04 baseline).

    Replaces OTSolver with uniform coupling.
    Interface matches OTSolver signature.

    No cost matrix computation, no Sinkhorn iterations.
    Returns uniform plan: P[i,j] = 1/n_cells for all spots.
    """

    def __call__(
        self,
        fused_st: Tensor,
        scRNA_profiles: Tensor,
        **kwargs,
    ) -> TransportPlan:
        """Compute uniform transport plan (no OT solving).

        Parameters
        ----------
        fused_st : Tensor
            (n_spots, latent_dim) — ignored, used only for shape
        scRNA_profiles : Tensor
            (n_cells, latent_dim) — used only for shape
        **kwargs
            Ignored (no OT parameters needed)

        Returns
        -------
        TransportPlan
            Uniform coupling plan with:
            - plan: (n_spots, n_cells) uniform = 1/n_cells
            - cost: zeros
            - marginal_error: 0.0 (uniform is exact)
            - convergence_passed: True
            - n_iter: 0
        """
        n_spots = fused_st.shape[0]
        n_cells = scRNA_profiles.shape[0]

        # Uniform distribution: P = ones / n_cells
        P = torch.ones(n_spots, n_cells, dtype=torch.float32) / n_cells

        # Zero cost (no optimization)
        cost = torch.zeros(n_spots, n_cells, dtype=torch.float32)

        return TransportPlan(
            plan=P,
            cost=cost,
            marginal_error=0.0,  # Uniform has exact marginals
            convergence_passed=True,
            n_iter=0,  # No iterations
        )