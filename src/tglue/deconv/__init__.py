"""Cell type deconvolution utilities."""

import torch

from .cell_type_proportions import (
    transport_to_proportions,
    simplex_projection,
    validate_proportions,
    validate_transport_plan,
)
from .ot_solver import OTSolver
from .evaluator import (
    compute_pearson_correlation,
    compute_bulk_alignment_kl,
    DeconvolutionMetrics,
    evaluate_deconvolution,
    log_deconvolution_metrics,
)
from .bulk_prior import (
    BulkPriorConfig,
    BulkPriorOutput,
    compute_bulk_prior_lambda,
    preprocess_bulk_ssgsea,
    compute_cluster_level_kl,
    compute_bulk_prior_loss,
)


def compute_cell_type_proportions(transport_plan: torch.Tensor, cell_type_onehot: torch.Tensor) -> torch.Tensor:
    """
    Compute cell type proportions from transport plan using one-hot encoded cell types.

    Parameters
    ----------
    transport_plan : torch.Tensor
        Transport plan of shape (n_spots, n_cells).
    cell_type_onehot : torch.Tensor
        Binary matrix of shape (n_cells, n_cell_types).

    Returns
    -------
    torch.Tensor
        Proportions of shape (n_spots, n_cell_types).
    """
    # Compute proportions via matrix multiplication
    proportions = transport_plan @ cell_type_onehot  # (n_spots, n_cell_types)

    # Project to simplex (non-negative, sum to 1 per row)
    # Uses the standard projection onto the probability simplex
    n_spots, n_types = proportions.shape
    projected = torch.zeros_like(proportions)

    for i in range(n_spots):
        x = proportions[i]

        # Sort values in descending order
        sorted_x, _ = torch.sort(x, descending=True)

        # Compute cumulative sums
        cumsum = torch.cumsum(sorted_x, dim=0)

        # Find largest t such that sorted_x[t-1] > (cumsum[t-1] - 1) / t
        t = n_types
        for j in range(n_types - 1, -1, -1):
            tau = (cumsum[j] - 1) / (j + 1)
            if sorted_x[j] > tau:
                t = j + 1
                break

        # Compute lambda
        if t > 0:
            lam = (cumsum[t - 1] - 1) / t
        else:
            lam = 0.0

        # Project: max(x - lam, 0)
        proj_i = torch.clamp(x - lam, min=0)

        # Normalize to sum to 1
        row_sum = proj_i.sum()
        if row_sum > 0:
            proj_i = proj_i / row_sum

        projected[i] = proj_i

    return projected


__all__ = [
    "OTSolver",
    "compute_cell_type_proportions",
    "transport_to_proportions",
    "simplex_projection",
    "validate_proportions",
    "validate_transport_plan",
    "compute_pearson_correlation",
    "compute_bulk_alignment_kl",
    "DeconvolutionMetrics",
    "evaluate_deconvolution",
    "log_deconvolution_metrics",
    "BulkPriorConfig",
    "BulkPriorOutput",
    "compute_bulk_prior_lambda",
    "preprocess_bulk_ssgsea",
    "compute_cluster_level_kl",
    "compute_bulk_prior_loss",
]