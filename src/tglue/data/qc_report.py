"""QC statistics report generation for triple-modal data.

Generates per-modality statistics (cell counts, gene counts, QC metrics)
following RL-04 requirement.

Usage:
    from tglue.data.qc_report import generate_qc_report
    from tglue.data.rosacea_loader import load_rosacea_dataset

    dataset = load_rosacea_dataset()
    report = generate_qc_report(dataset)
    print(report)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import scanpy as sc

logger = logging.getLogger(__name__)


def generate_qc_report(
    dataset_or_adatas: Any,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate QC statistics report for triple-modal data.

    Parameters
    ----------
    dataset_or_adatas : Any
        Either TripleModalDataset instance or dict with keys:
        'scRNA', 'ST', 'Bulk' containing AnnData objects.
    output_path : Optional[str], default None
        If provided, save report as JSON to this path.

    Returns
    -------
    Dict[str, Any]
        Report with sections: 'scRNA', 'ST', 'Bulk', 'shared_genes'.

    Notes
    -----
    RL-04: QC statistics output for each modality.
    Handles precomputed QC metrics (Rosacea data has percent.mt, nCount columns).
    """
    # Extract AnnData objects
    if hasattr(dataset_or_adatas, 'adata_sc'):
        # TripleModalDataset instance
        adata_sc = dataset_or_adatas.adata_sc
        adata_st = dataset_or_adatas.adata_st
        adata_bulk = dataset_or_adatas.adata_bulk
        n_genes = dataset_or_adatas.n_genes
    elif isinstance(dataset_or_adatas, dict):
        adata_sc = dataset_or_adatas.get('scRNA')
        adata_st = dataset_or_adatas.get('ST')
        adata_bulk = dataset_or_adatas.get('Bulk')
        n_genes = adata_sc.n_vars if adata_sc is not None else None
    else:
        raise TypeError("Expected TripleModalDataset or dict of AnnData objects")

    report = {}

    # scRNA section
    if adata_sc is not None:
        report['scRNA'] = _get_scrna_stats(adata_sc)
        logger.info(f"[QC Report] scRNA: {report['scRNA']['n_cells']} cells, "
                    f"{report['scRNA']['n_genes']} genes")

    # ST section
    if adata_st is not None:
        report['ST'] = _get_st_stats(adata_st)
        logger.info(f"[QC Report] ST: {report['ST']['n_spots']} spots, "
                    f"{report['ST']['n_genes']} genes")

    # Bulk section
    if adata_bulk is not None:
        report['Bulk'] = _get_bulk_stats(adata_bulk)
        logger.info(f"[QC Report] Bulk: {report['Bulk']['n_samples']} samples, "
                    f"{report['Bulk']['n_genes']} genes")

    # Shared genes section
    report['shared_genes'] = {
        'n_genes': n_genes,
        'min_threshold': 2000,  # D-07
        'threshold_passed': n_genes >= 2000 if n_genes else False,
    }

    # Save to file if output_path provided
    if output_path:
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path_obj, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"[QC Report] Saved to {output_path}")

    return report


def _get_scrna_stats(adata: sc.AnnData) -> Dict[str, Any]:
    """Get scRNA QC statistics."""
    stats = {
        'n_cells': adata.n_obs,
        'n_genes': adata.n_vars,
    }

    # QC metrics (handle missing columns)
    if 'percent.mt' in adata.obs.columns:
        stats['mt_pct_mean'] = float(adata.obs['percent.mt'].mean())
        stats['mt_pct_median'] = float(adata.obs['percent.mt'].median())
    else:
        stats['mt_pct_mean'] = None
        stats['mt_pct_median'] = None
        logger.warning("[QC Report] percent.mt column missing in scRNA")

    if 'nCount_RNA' in adata.obs.columns:
        stats['nCount_mean'] = float(adata.obs['nCount_RNA'].mean())
        stats['nCount_median'] = float(adata.obs['nCount_RNA'].median())
    elif 'total_counts' in adata.obs.columns:
        stats['nCount_mean'] = float(adata.obs['total_counts'].mean())
        stats['nCount_median'] = float(adata.obs['total_counts'].median())
    else:
        stats['nCount_mean'] = None
        logger.warning("[QC Report] nCount column missing in scRNA")

    return stats


def _get_st_stats(adata: sc.AnnData) -> Dict[str, Any]:
    """Get ST QC statistics."""
    stats = {
        'n_spots': adata.n_obs,
        'n_genes': adata.n_vars,
    }

    # QC metrics
    if 'percent.mt' in adata.obs.columns:
        stats['mt_pct_mean'] = float(adata.obs['percent.mt'].mean())
        stats['mt_pct_median'] = float(adata.obs['percent.mt'].median())
    else:
        stats['mt_pct_mean'] = None
        logger.warning("[QC Report] percent.mt column missing in ST")

    if 'nCount_Spatial' in adata.obs.columns:
        stats['nCount_mean'] = float(adata.obs['nCount_Spatial'].mean())
    elif 'total_counts' in adata.obs.columns:
        stats['nCount_mean'] = float(adata.obs['total_counts'].mean())
    else:
        stats['nCount_mean'] = None
        logger.warning("[QC Report] nCount column missing in ST")

    # Spatial coordinates
    if 'spatial' in adata.obsm:
        stats['spatial_coords_shape'] = list(adata.obsm['spatial'].shape)

    return stats


def _get_bulk_stats(adata: sc.AnnData) -> Dict[str, Any]:
    """Get Bulk QC statistics."""
    stats = {
        'n_samples': adata.n_obs,
        'n_genes': adata.n_vars,
    }

    # Condition distribution
    if 'condition' in adata.obs.columns:
        stats['condition_distribution'] = adata.obs['condition'].value_counts().to_dict()
    elif 'group' in adata.obs.columns:
        stats['condition_distribution'] = adata.obs['group'].value_counts().to_dict()
    else:
        logger.warning("[QC Report] condition/group column missing in Bulk")

    return stats