"""ssGSEA Bulk preprocessing for AB-07 (D-AB05).

Replaces placeholder Bulk preprocessing with real ssGSEA via gseapy.
Produces simplex proportions: enrichment scores normalized to sum=1.

Key: Gene name matching handles different naming conventions (Pitfall 3).
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Union, List

import numpy as np
import pandas as pd
import scipy.sparse
import torch
from torch import Tensor

try:
    import gseapy as gp
except ImportError:
    gp = None  # type: ignore


@dataclass
class SsgseaOutput:
    """Output from ssGSEA preprocessing.

    Attributes:
        proportions: (n_samples, n_cell_types) cell type proportion estimates
        cell_type_names: List of cell type names from GMT/dict (column order)
    """
    proportions: Tensor
    cell_type_names: List[str]


def get_default_markers_path() -> Path:
    """Get path to bundled brain marker GMT file.

    Returns
    -------
    Path
        Path to brain_markers.gmt bundled with the package

    Raises
    ------
    FileNotFoundError
        If bundled marker file not found (should not happen in correct install)
    """
    try:
        # Python 3.9+ syntax
        return Path(resources.files('tglue.data.markers') / 'brain_markers.gmt')
    except AttributeError:
        # Python 3.7-3.8 fallback
        with resources.path('tglue.data.markers', 'brain_markers.gmt') as p:
            return p


def get_skin_markers_path() -> Path:
    """Get path to bundled skin marker GMT file.

    Returns
    -------
    Path
        Path to skin_markers.gmt bundled with the package

    Raises
    ------
    FileNotFoundError
        If bundled marker file not found (should not happen in correct install)
    """
    try:
        # Python 3.9+ syntax
        return Path(resources.files('tglue.data.markers') / 'skin_markers.gmt')
    except AttributeError:
        # Python 3.7-3.8 fallback
        with resources.path('tglue.data.markers', 'skin_markers.gmt') as p:
            return p


def preprocess_bulk_ssgsea(
    bulk_adata,
    gene_sets: Union[str, Path, dict, None] = None,
    normalize: bool = True,
    return_names: bool = True,
) -> Union[Tensor, SsgseaOutput]:
    """Preprocess Bulk RNA-seq via ssGSEA for cell type proportion estimation.

    D-AB05: ssGSEA computes enrichment scores per sample per cell type.
    Scores are normalized to simplex (sum=1, all >=0).

    Parameters
    ----------
    bulk_adata : ann.AnnData
        Bulk RNA-seq AnnData with .X (samples x genes) and .var_names (gene names)
    gene_sets : str | Path | dict | None, default None
        Gene set database:
        - None: Load bundled brain markers (default)
        - str/Path: GMT file path (e.g., data/markers/panglao_db.gmt)
        - dict: {cell_type_name: [gene1, gene2, ...]}
    normalize : bool, default True
        Normalize enrichment scores to simplex (rows sum to 1)
    return_names : bool, default True
        Return SsgseaOutput with cell type names. If False, return Tensor only
        (backward compatibility for existing code).

    Returns
    -------
    Tensor | SsgseaOutput
        If return_names=True:
            SsgseaOutput with proportions and cell_type_names
        If return_names=False:
            Tensor of shape (n_samples, n_cell_types) proportions

    Raises
    ------
    ImportError
        If gseapy not installed
    ValueError
        If bulk_adata.X has no genes or samples
    """
    if gp is None:
        raise ImportError(
            "gseapy not installed. Run: pip install gseapy==1.1.9"
        )

    # Load bundled brain markers if gene_sets not provided
    if gene_sets is None:
        gene_sets = str(get_default_markers_path())

    # Extract expression matrix
    X = bulk_adata.X
    if X is None:
        raise ValueError("bulk_adata.X is None")

    # Convert sparse to dense
    if scipy.sparse.issparse(X):
        X = X.toarray()

    # Build DataFrame with genes as index (gseapy expects this format)
    # gseapy expects: rows=genes, columns=samples
    # bulk_adata.X is (n_samples, n_genes), .var_names has gene names
    # Need to transpose to get genes as rows
    X_T = X.T  # (n_genes, n_samples)

    expr_df = pd.DataFrame(
        X_T,
        index=bulk_adata.var_names,  # genes as index (rows)
        columns=bulk_adata.obs_names if bulk_adata.obs_names is not None else range(X.shape[0]),  # samples as columns
    )

    # Run ssGSEA
    # gseapy.ssgsea returns enrichment scores per sample per gene set
    # Note: gene_sets parameter accepts dict or GMT file path
    ss = gp.ssgsea(
        data=expr_df,
        gene_sets=gene_sets,
        outdir=None,  # No output directory
        no_plot=True,  # Disable plotting
        sample_norm_method="rank",  # Rank normalization (default)
        min_size=1,  # Allow small gene sets for testing
        max_size=1000,  # Allow large gene sets
    )

    # Extract enrichment scores DataFrame
    # ss.res2d has columns: ['Name', 'Term', 'ES', 'NES']
    # Name = sample, Term = cell type, ES = enrichment score
    # Need to pivot to get (samples, cell_types) matrix
    scores_df = ss.res2d.pivot(index='Name', columns='Term', values='ES')

    # Extract cell type names from columns (column order = GMT/dict order)
    cell_type_names: List[str] = scores_df.columns.tolist()

    # Convert to numpy array
    scores = scores_df.values.astype(np.float32)

    # Clamp to non-negative (some scores may be negative)
    scores = np.maximum(scores, 0.0)

    # Normalize to simplex if requested
    if normalize:
        row_sums = scores.sum(axis=1, keepdims=True)
        # Avoid division by zero - if row sum is zero, use uniform distribution
        zero_rows = (row_sums == 0).flatten()
        if zero_rows.any():
            # Replace zero rows with uniform distribution (1/n_cell_types)
            n_cell_types = scores.shape[1]
            scores[zero_rows] = 1.0 / n_cell_types
            row_sums = scores.sum(axis=1, keepdims=True)
        scores = scores / row_sums

    # Convert to tensor
    proportions = torch.from_numpy(scores)

    if return_names:
        return SsgseaOutput(proportions=proportions, cell_type_names=cell_type_names)
    else:
        return proportions