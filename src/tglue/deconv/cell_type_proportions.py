"""
Cell Type Deconvolution Pipeline

Converts transport plan to cell type proportions per spot.
Ensures proportions are valid probability distributions (non-negative, sum to 1 per spot).
"""

import numpy as np


def transport_to_proportions(P: np.ndarray, cell_type_matrix: np.ndarray) -> np.ndarray:
    """
    Convert transport plan to cell type proportions per spot.

    Parameters
    ----------
    P : np.ndarray
        Transport plan of shape (n_spots, n_cells).
        P[i, j] represents the mass transported from spot i to cell j.
    cell_type_matrix : np.ndarray
        Binary matrix of shape (n_cells, n_cell_types).
        cell_type_matrix[j, k] = 1 if cell j belongs to cell type k, 0 otherwise.

    Returns
    -------
    np.ndarray
        Proportions of shape (n_spots, n_cell_types).
        proportions[i, k] is the proportion of cell type k in spot i.
    """
    if P.ndim != 2:
        raise ValueError(f"P must be 2D array, got shape {P.shape}")
    if cell_type_matrix.ndim != 2:
        raise ValueError(f"cell_type_matrix must be 2D array, got shape {cell_type_matrix.shape}")
    if P.shape[1] != cell_type_matrix.shape[0]:
        raise ValueError(
            f"P has {P.shape[1]} cells but cell_type_matrix has {cell_type_matrix.shape[0]} cells"
        )

    # Compute cell type proportions: sum transport mass weighted by cell type membership
    # P @ cell_type_matrix gives (n_spots, n_cell_types) where each entry is total mass
    # for that spot's cells that belong to each cell type
    proportions = P @ cell_type_matrix

    # Project to simplex (non-negative, sum to 1)
    proportions = simplex_projection(proportions)

    return proportions


def simplex_projection(proportions: np.ndarray) -> np.ndarray:
    """
    Project proportions to the simplex Δ^{K-1}: non-negative and sum to 1 per row.

    Uses the standard projection onto the probability simplex:
    For each row x, find τ such that the projected values are max(x - τ, 0)
    and sum to 1. The optimal τ is found via binary search.

    Parameters
    ----------
    proportions : np.ndarray
        Array of shape (n_spots, n_cell_types). Can have negative values.

    Returns
    -------
    np.ndarray
        Projected array of shape (n_spots, n_cell_types) where each row
        is a valid probability distribution (non-negative, sums to 1).
    """
    if proportions.ndim != 2:
        raise ValueError(f"proportions must be 2D array, got shape {proportions.shape}")

    n_spots, n_types = proportions.shape
    projected = np.zeros_like(proportions)

    for i in range(n_spots):
        x = proportions[i]

        # Find the projection onto the simplex using the standard algorithm
        # Sort values in descending order
        sorted_x = np.sort(x)[::-1]

        # Find the largest τ such that x_j - τ > 0 for the relevant indices
        # Cumulative sum of sorted values: CSS(t) = sum_{j=1}^t (x_j - τ)
        # We need CSS(t) = 1 for the optimal t

        # Compute cumulative sums of sorted values
        cumsum = np.cumsum(sorted_x)

        # Find the largest j where sorted_x[j] > (cumsum[j] - 1) / (j + 1)
        # This is the condition for τ = (cumsum[j] - 1) / (j + 1)
        t = len(sorted_x)
        for j in range(n_types - 1, -1, -1):
            tau = (cumsum[j] - 1) / (j + 1)
            if sorted_x[j] > tau:
                t = j + 1
                break

        # The threshold is (cumsum[t-1] - 1) / t if t > 0
        if t > 0:
            tau = (cumsum[t - 1] - 1) / t
        else:
            tau = 0

        # Project: max(x - tau, 0)
        projected[i] = np.maximum(x - tau, 0)

        # Ensure sum is exactly 1 (numerical stability)
        row_sum = projected[i].sum()
        if row_sum > 0:
            projected[i] /= row_sum

    return projected


def validate_proportions(proportions: np.ndarray) -> None:
    """
    Validate that proportions are valid probability distributions.

    Parameters
    ----------
    proportions : np.ndarray
        Array of shape (n_spots, n_cell_types) to validate.

    Raises
    ------
    AssertionError
        If proportions are not valid probability distributions.
    """
    assert np.all(proportions >= 0), "Proportions must be non-negative"
    assert np.allclose(proportions.sum(axis=1), 1), "Proportions must sum to 1 per spot"


def validate_transport_plan(P: np.ndarray) -> None:
    """
    Validate that transport plan is finite.

    Parameters
    ----------
    P : np.ndarray
        Transport plan of shape (n_spots, n_cells).

    Raises
    ------
    AssertionError
        If transport plan contains NaN or Inf values.
    """
    assert np.all(np.isfinite(P)), "Transport plan must be finite (no NaN or Inf)"
