"""SC<->ST mapping matrix export and bulk comparison (D-06).

Extracted from scripts/generate_all_results.py steps 4/6 and 5/6.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch
from scipy import sparse as sp

logger = logging.getLogger(__name__)

# Dense threshold: if plan nnz < this fraction of total, use sparse path
_SPARSE_THRESHOLD = 0.01


def _torch_sparse_to_scipy(plan: torch.Tensor) -> sp.csr_matrix:
    """Convert torch sparse COO tensor to scipy CSR on CPU."""
    plan = plan.coalesce().cpu()
    indices = plan.indices().numpy()
    values = plan.values().numpy()
    shape = plan.shape
    coo = sp.coo_matrix((values, (indices[0], indices[1])), shape=shape)
    return coo.tocsr()


def _compute_mapping_sparse(
    plan_csr: sp.csr_matrix,
    topk: int,
    n_workers: int,
) -> tuple[np.ndarray, dict, np.ndarray]:
    """Sparse-path mapping: top-k per row without dense materialization.

    Operates entirely in scipy sparse on CPU. Each row typically has
    ~50 non-zero entries (after k-NN prefiltering), so finding top-k
    is O(nnz_per_row) not O(n_cells).
    """
    n_spots, n_cells = plan_csr.shape
    actual_topk = min(topk, n_cells)

    # Pre-allocate output arrays
    topk_vals = np.zeros((n_spots, actual_topk), dtype=np.float32)
    topk_idx = np.zeros((n_spots, actual_topk), dtype=np.int64)
    mapping_counts = np.zeros(n_cells, dtype=np.int64)

    def _process_row_chunk(start: int, end: int) -> None:
        for i in range(start, end):
            row = plan_csr.getrow(i)
            row_data = row.data
            row_indices = row.indices
            k = min(actual_topk, len(row_data))
            if k > 0:
                # argpartition is O(n) vs O(n log n) for argsort
                part_idx = np.argpartition(row_data, -k)[-k:]
                sorted_part = part_idx[np.argsort(row_data[part_idx])[::-1]]
                topk_vals[i, :k] = row_data[sorted_part]
                topk_idx[i, :k] = row_indices[sorted_part]
                np.add.at(mapping_counts, topk_idx[i, :k], 1)

    # Parallel row processing
    chunk_size = max(1, n_spots // n_workers)
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = []
        for start in range(0, n_spots, chunk_size):
            end = min(start + chunk_size, n_spots)
            futures.append(pool.submit(_process_row_chunk, start, end))
        for f in futures:
            f.result()

    # Build sparse trans_matrix (only top-k entries per row)
    trans_rows = np.repeat(np.arange(n_spots), actual_topk)
    trans_cols = topk_idx.ravel()
    trans_data = topk_vals.ravel().copy()

    # WR-01 FIX: Preserve density signal BEFORE normalization
    density_signal = topk_vals.sum(axis=1)
    density_max = max(density_signal.max(), 1e-8)

    # Row-normalize
    row_sums = topk_vals.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-8)
    topk_vals /= row_sums
    trans_data_norm = topk_vals.ravel()

    trans_sparse = sp.coo_matrix(
        (trans_data_norm, (trans_rows, trans_cols)), shape=(n_spots, n_cells)
    )

    stats = {
        "n_spots": n_spots,
        "n_cells": n_cells,
        "topk": actual_topk,
        "mean_spots_per_cell": float(mapping_counts.mean()),
        "max_spots_per_cell": int(mapping_counts.max()),
        "cells_with_zero_mapping": int((mapping_counts == 0).sum()),
        "density_mean": float(density_signal.mean()),
        "density_max": float(density_signal.max()),
        "density_min": float(density_signal.min()),
        "sparse_path": True,
    }
    return trans_sparse.toarray(), stats, mapping_counts


def compute_mapping(
    transport_plan: torch.Tensor,
    topk: int = 10,
    force_cpu: bool = False,
    n_workers: int = 4,
) -> tuple[np.ndarray, dict, np.ndarray]:
    """Compute SC<->ST top-k mapping matrix and statistics.

    Supports two execution paths:
    - **Sparse CPU path** (default for large plans): converts to scipy CSR,
      extracts top-k per row without dense materialization. Handles 100K×76K
      plans in ~40 MB sparse storage.
    - **Dense GPU path** (small plans only): uses torch.topk on GPU.

    Parameters
    ----------
    transport_plan : torch.Tensor
        Sparse or dense transport plan, shape (n_spots, n_cells).
    topk : int
        Number of top mappings per spot.
    force_cpu : bool
        Force CPU sparse path regardless of plan size.
    n_workers : int
        Thread count for parallel row processing (CPU path only).

    Returns
    -------
    tuple[np.ndarray, dict, np.ndarray]
        (trans_matrix_numpy, stats_dict, mapping_counts)
        trans_matrix_numpy: row-normalized (n_spots, n_cells) numpy array.
        stats_dict: mapping statistics.
        mapping_counts: per-cell mapping count array (n_cells,).
    """
    n_spots, n_cells = transport_plan.shape
    total = n_spots * n_cells
    nnz = transport_plan._nnz() if transport_plan.is_sparse else total
    sparsity = nnz / total if total > 0 else 1.0

    # Use sparse CPU path for large or sparse plans
    use_sparse = force_cpu or sparsity < _SPARSE_THRESHOLD or total > 500_000_000
    if use_sparse and transport_plan.is_sparse:
        logger.info(
            f"  Mapping (sparse CPU): {n_spots:,}×{n_cells:,}, "
            f"nnz={nnz:,}, sparsity={sparsity:.6f}, workers={n_workers}"
        )
        plan_csr = _torch_sparse_to_scipy(transport_plan)
        return _compute_mapping_sparse(plan_csr, topk, n_workers)

    # Fallback: dense GPU path (original logic, for small plans)
    logger.info(f"  Mapping (dense): {n_spots:,}×{n_cells:,}")
    plan = transport_plan.coalesce()
    plan_dense = plan.to_dense()
    actual_topk = min(topk, plan_dense.shape[1])
    topk_vals, topk_idx = plan_dense.topk(k=actual_topk, dim=1)

    mapping_counts = torch.zeros(plan_dense.shape[1], dtype=torch.long)
    for k in range(topk_idx.shape[1]):
        mapping_counts.scatter_add_(
            0, topk_idx[:, k], torch.ones(topk_idx.shape[0], dtype=torch.long),
        )

    trans_matrix_st = torch.zeros_like(plan_dense)
    row_idx = torch.arange(plan_dense.shape[0]).unsqueeze(1).expand_as(topk_idx)
    trans_matrix_st[row_idx.flatten(), topk_idx.flatten()] = topk_vals.flatten()

    # WR-01 FIX: Preserve density signal BEFORE normalization
    density_signal = trans_matrix_st.sum(dim=1)
    density_max = density_signal.max().clamp(min=1e-8)

    row_sums = trans_matrix_st.sum(dim=1, keepdim=True).clamp(min=1e-8)
    trans_matrix_st = trans_matrix_st / row_sums

    stats = {
        "n_spots": int(plan_dense.shape[0]),
        "n_cells": int(plan_dense.shape[1]),
        "topk": actual_topk,
        "mean_spots_per_cell": float(mapping_counts.float().mean()),
        "max_spots_per_cell": int(mapping_counts.max()),
        "cells_with_zero_mapping": int((mapping_counts == 0).sum()),
        "density_mean": float(density_signal.mean()),
        "density_max": float(density_signal.max()),
        "density_min": float(density_signal.min()),
        "sparse_path": False,
    }
    logger.info(f"  Mapping stats: {stats}")
    return trans_matrix_st.cpu().numpy(), stats, mapping_counts.cpu().numpy()


def compute_bulk_comparison(
    st_adata,
    bulk_adata,
    proportions_np: np.ndarray,
    conditions,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Compare bulk ssGSEA proportions vs OT-predicted proportions by condition.

    Parameters
    ----------
    st_adata : sc.AnnData
        ST AnnData with ``obs["condition"]``.
    bulk_adata : sc.AnnData
        Bulk AnnData with ``obs["condition"]``.
    proportions_np : np.ndarray
        OT deconvolution proportions, shape (n_spots, n_cell_types).
    conditions : CanonicalConditions
        Condition normalization helper.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, list[str]]
        (bulk_matrix, predicted_matrix, bulk_cell_type_names)
        Each matrix is shape (n_conditions, n_cell_types).
    """
    st_conditions = conditions.normalize_array(
        st_adata.obs.get(
            "condition", pd.Series(["Unknown"] * st_adata.n_obs),
        ).values,
    )
    bulk_conditions = conditions.normalize_array(
        bulk_adata.obs.get(
            "condition", pd.Series(["Unknown"] * bulk_adata.n_obs),
        ).values,
    )

    # Predicted proportions: aggregate by condition
    from ..deconv.label_mapping import get_canonical_cell_types
    cell_types = get_canonical_cell_types()
    predicted_by_condition = {}
    for cond in conditions.names:
        mask = st_conditions == cond
        if mask.sum() > 0:
            cond_props = proportions_np[mask].mean(axis=0)
        else:
            cond_props = np.zeros(len(cell_types.names))
        predicted_by_condition[cond] = cond_props

    pred_matrix = np.array([predicted_by_condition[c] for c in conditions.names])

    # Bulk ssGSEA proportions
    bulk_ct_names = cell_types.names  # fallback
    try:
        from tglue.preprocessing.ssgsea_bulk import preprocess_bulk_ssgsea

        ssgsea_output = preprocess_bulk_ssgsea(bulk_adata)
        ssgsea_result = ssgsea_output.proportions.numpy()
        bulk_ct_names = ssgsea_output.cell_type_names
        n_ssgsea_types = len(bulk_ct_names)

        bulk_props = {}
        for cond in conditions.names:
            mask = bulk_conditions == cond
            if mask.sum() > 0:
                cond_mean = ssgsea_result[mask].mean(axis=0)
                cond_mean = np.maximum(cond_mean, 0)
                s = cond_mean.sum()
                if s > 0:
                    cond_mean = cond_mean / s
                bulk_props[cond] = cond_mean
            else:
                bulk_props[cond] = np.zeros(n_ssgsea_types)
    except (ValueError, RuntimeError) as e:
        logger.warning(f"  ssGSEA failed ({e}), using uniform bulk proportions")
        n_fallback = len(cell_types.names)
        bulk_props = {}
        for cond in conditions.names:
            bulk_props[cond] = np.ones(n_fallback) / n_fallback

    bulk_matrix = np.array([bulk_props[c] for c in conditions.names])
    return bulk_matrix, pred_matrix, bulk_ct_names
