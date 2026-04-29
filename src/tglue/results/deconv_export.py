"""OT deconvolution export with cell type proportions (D-06).

Extracted from scripts/generate_all_results.py steps 3/6.
"""

from __future__ import annotations

import logging

import numpy as np
import torch

from ..deconv import compute_cell_type_proportions
from ..deconv.label_mapping import get_canonical_cell_types
from ..deconv.ot_solver import OTSolver

logger = logging.getLogger(__name__)


def run_deconvolution(
    u_st: np.ndarray,
    u_sc: np.ndarray,
    sc_adata,
    device: str = "cuda",
    epsilon: float = 0.5,
    k_neighbors=50,
    chunk_size: int = 5000,
    two_pass: bool = True,
) -> tuple[np.ndarray, object, object]:
    """Run OT deconvolution and compute cell type proportions.

    Parameters
    ----------
    u_st : np.ndarray
        ST latent embeddings, shape (n_spots, latent_dim).
    u_sc : np.ndarray
        scRNA latent embeddings, shape (n_cells, latent_dim).
    sc_adata : sc.AnnData
        scRNA AnnData with ``obs["cell_type"]`` column.
    device : str
        Torch device.
    epsilon : float
        OT entropy regularization (>= 0.1 enforced by OTSolver).
    k_neighbors : int or "auto"
        k-NN pre-filtering neighbors. Default 50 for backward compat.
    chunk_size : int
        Chunk size for solve_chunked().
    two_pass : bool
        Use two-pass solving for improved marginal accuracy.

    Returns
    -------
    tuple[np.ndarray, CanonicalCellTypes, TransportPlan]
        (proportions_np, cell_types, transport_result)
        proportions_np: shape (n_spots, n_cell_types), numpy float32.
    """
    cell_types = get_canonical_cell_types()
    ct_labels = sc_adata.obs["cell_type"].values
    cell_type_onehot = torch.tensor(
        cell_types.to_onehot(ct_labels), dtype=torch.float32,
    ).to(device)
    n_cell_types = cell_type_onehot.shape[1]

    u_st_t = torch.tensor(u_st, dtype=torch.float32).to(device)
    u_sc_t = torch.tensor(u_sc, dtype=torch.float32).to(device)

    solver = OTSolver(epsilon=epsilon, k_neighbors=k_neighbors, n_iters=100)
    result = solver.solve_chunked(
        u_st_t, u_sc_t, chunk_size=chunk_size,
        n_cell_types=n_cell_types, two_pass=two_pass,
    )
    logger.info(
        f"  Transport plan: {result.plan.shape}, "
        f"converged={result.convergence_passed}"
    )

    proportions = compute_cell_type_proportions(result.plan, cell_type_onehot)
    proportions_np = proportions.cpu().numpy()

    logger.info(f"  Proportions shape: {proportions_np.shape}")
    return proportions_np, cell_types, result
