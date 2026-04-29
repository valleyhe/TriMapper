"""Preprocessing pipeline for triple-modal data.

QC and normalization functions following D-02 (QC defaults), D-03 (normalization),
and D-04 (logging requirements).

QC parameters:
- min_genes=200 (cells with fewer genes dropped)
- min_cells=3 (genes expressed in fewer cells dropped)
- mt_pct_threshold=20 (cells with high mitochondrial content dropped)

Normalization:
- normalize_total(target_sum=1e4) + log1p
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import scanpy as sc
import zarr

logger = logging.getLogger(__name__)


def preprocess_scrna(
    adata: sc.AnnData,
    min_genes: int = 200,
    min_cells: int = 3,
    mt_pct_threshold: float = 20.0,
    target_sum: float = 1e4,
) -> sc.AnnData:
    """Preprocess scRNA-seq AnnData with QC filtering and normalization.

    Following D-02 (QC defaults), D-03 (normalization), D-04 (logging).

    Parameters
    ----------
    adata : sc.AnnData
        scRNA-seq AnnData object with raw counts.
    min_genes : int, default 200
        Minimum genes per cell threshold (D-02).
    min_cells : int, default 3
        Minimum cells per gene threshold (D-02).
    mt_pct_threshold : float, default 20.0
        Maximum mitochondrial percentage per cell (D-02).
    target_sum : float, default 1e4
        Target sum for normalization (D-03).

    Returns
    -------
    sc.AnnData
        Preprocessed AnnData with QC filtering and normalization applied.
    """
    # Log before QC
    n_cells_before = adata.n_obs
    n_genes_before = adata.n_vars
    logger.info("[scRNA QC] Before: %d cells, %d genes", n_cells_before, n_genes_before)

    # Identify mitochondrial genes (human: MT- prefix)
    adata.var["mt"] = adata.var.index.str.startswith("MT-")

    # Calculate QC metrics
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)

    # Filter cells by min_genes
    sc.pp.filter_cells(adata, min_genes=min_genes)

    # Filter by mt percentage
    adata = adata[adata.obs.pct_counts_mt < mt_pct_threshold].copy()

    # Filter genes by min_cells
    sc.pp.filter_genes(adata, min_cells=min_cells)

    # Log after QC
    n_cells_after = adata.n_obs
    n_genes_after = adata.n_vars
    logger.info("[scRNA QC] After: %d cells, %d genes", n_cells_after, n_genes_after)
    logger.info(
        "[scRNA QC] Filtered: %d cells, %d genes",
        n_cells_before - n_cells_after,
        n_genes_before - n_genes_after,
    )

    # Normalization: D-03
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)
    logger.info("[scRNA Normalization] Completed normalize_total + log1p")

    return adata


def preprocess_st(
    adata: sc.AnnData,
    min_genes: int = 200,
    min_cells: int = 3,
    mt_pct_threshold: float = 20.0,
    target_sum: float = 1e4,
) -> sc.AnnData:
    """Preprocess spatial transcriptomics AnnData with QC filtering and normalization.

    Following same QC parameters as scRNA (D-02, D-03, D-04).

    Parameters
    ----------
    adata : sc.AnnData
        ST AnnData object with raw counts and spatial coordinates.
    min_genes : int, default 200
        Minimum genes per spot threshold (D-02).
    min_cells : int, default 3
        Minimum spots per gene threshold (D-02).
    mt_pct_threshold : float, default 20.0
        Maximum mitochondrial percentage per spot (D-02).
    target_sum : float, default 1e4
        Target sum for normalization (D-03).

    Returns
    -------
    sc.AnnData
        Preprocessed AnnData with QC filtering and normalization applied.
    """
    # Log before QC
    n_spots_before = adata.n_obs
    n_genes_before = adata.n_vars
    logger.info("[ST QC] Before: %d spots, %d genes", n_spots_before, n_genes_before)

    # Identify mitochondrial genes (human: MT- prefix)
    adata.var["mt"] = adata.var.index.str.startswith("MT-")

    # Calculate QC metrics
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)

    # Filter spots by min_genes
    sc.pp.filter_cells(adata, min_genes=min_genes)

    # Filter by mt percentage
    adata = adata[adata.obs.pct_counts_mt < mt_pct_threshold].copy()

    # Filter genes by min_cells
    sc.pp.filter_genes(adata, min_cells=min_cells)

    # Log after QC
    n_spots_after = adata.n_obs
    n_genes_after = adata.n_vars
    logger.info("[ST QC] After: %d spots, %d genes", n_spots_after, n_genes_after)
    logger.info(
        "[ST QC] Filtered: %d spots, %d genes",
        n_spots_before - n_spots_after,
        n_genes_before - n_genes_after,
    )

    # Normalization: D-03
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)
    logger.info("[ST Normalization] Completed normalize_total + log1p")

    return adata


def preprocess_bulk(
    adata: sc.AnnData,
    target_sum: float = 1e4,
) -> sc.AnnData:
    """Preprocess bulk RNA-seq AnnData with normalization.

    Bulk typically has fewer samples and simpler QC requirements.

    Parameters
    ----------
    adata : sc.AnnData
        Bulk AnnData object with raw counts.
    target_sum : float, default 1e4
        Target sum for normalization (D-03).

    Returns
    -------
    sc.AnnData
        Preprocessed AnnData with normalization applied.
    """
    # Log before normalization
    n_samples_before = adata.n_obs
    n_genes_before = adata.n_vars
    logger.info("[Bulk QC] Before: %d samples, %d genes", n_samples_before, n_genes_before)

    # Normalization: D-03
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)

    logger.info("[Bulk QC] After normalization: %d samples, %d genes", adata.n_obs, adata.n_vars)

    return adata


def convert_h5ad_to_zarr(
    h5ad_path: str,
    chunk_shape: tuple[int, int] = (256, 20_000),
    force: bool = False,
) -> str:
    """Convert H5AD to Zarr with optimized chunks (D-10, D-11).

    Cache location: same directory as .h5ad, with .zarr suffix.
    One-time conversion: full memory load unavoidable, warn user.

    Parameters
    ----------
    h5ad_path : str
        Path to .h5ad file.
    chunk_shape : tuple[int, int], default (256, 20000)
        Chunk layout: (cells_per_chunk, genes_per_chunk).
    force : bool, default False
        If True, overwrite existing cache.

    Returns
    -------
    str
        Path to .zarr directory.

    Notes
    -----
    D-11: One-time full load is unavoidable for H5AD → Zarr conversion.
    Future runs will use cached Zarr for lazy loading, avoiding this cost.

    Examples
    --------
    >>> zarr_path = convert_h5ad_to_zarr("data/scrna.h5ad")
    >>> # First run: converts with warning
    >>> zarr_path = convert_h5ad_to_zarr("data/scrna.h5ad")
    >>> # Second run: uses cached Zarr, no conversion
    """
    h5ad_path_obj = Path(h5ad_path)
    zarr_path = h5ad_path_obj.with_suffix('.zarr')

    # Check if cache already exists (check for X array inside group)
    zarr_x_path = str(Path(zarr_path) / 'X')
    if Path(zarr_x_path).exists() and not force:
        logger.info(f"Using cached Zarr at {zarr_x_path}")
        return zarr_x_path

    # D-11: One-time full load (unavoidable)
    logger.warning(
        f"Converting {h5ad_path} to Zarr (one-time operation, "
        f"requires full memory load)..."
    )
    adata = sc.read_h5ad(h5ad_path)

    # Write with chunked layout per D-10 recommendation
    # chunks=(256, n_genes) for row-major batch iteration
    actual_chunk_shape = (
        chunk_shape[0],
        min(chunk_shape[1], adata.n_vars),
    )
    # AnnData.write_zarr creates a Zarr group with 'X' array
    adata.write_zarr(str(zarr_path), chunks=actual_chunk_shape)
    # Update zarr_path to point to the 'X' array inside the group
    zarr_x_path = str(Path(zarr_path) / 'X')

    # D-11: Warn user about cache location
    logger.warning(
        f"Zarr cache created at {zarr_x_path}. "
        f"Future runs will use cached Zarr for lazy loading."
    )

    return zarr_x_path