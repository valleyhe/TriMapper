"""Spatial and comparison visualization functions (D-06, D-02).

All spatial plots use matplotlib scatter (not scanpy spatial), per D-02.
All functions call set_publication_style() at start.
All functions return the output Path.

Also provides generate_he_overlays() for batch HE overlay generation across
all samples -- overlays spatial domains, deconvolution proportions, and
mapping density onto per-sample HE histopathology images.

Extracted from scripts/generate_all_results.py.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..visualization.publication_quality import set_publication_style

logger = logging.getLogger(__name__)


def plot_spatial_domains(
    st_adata,
    output_dir: Path,
    spot_size: float = 1.0,
) -> Path:
    """Plot spatial domain segmentation using matplotlib scatter.

    Parameters
    ----------
    st_adata : sc.AnnData
        AnnData with obsm["spatial"] and obs["domain"].
    output_dir : Path
        Directory to save PDF.
    spot_size : float
        Marker size.

    Returns
    -------
    Path
        Output file path.
    """
    set_publication_style()
    coords = st_adata.obsm["spatial"]
    domains = st_adata.obs["domain"].values
    n_domains = len(np.unique(domains))
    domain_codes = pd.Categorical(domains.astype(str)).codes

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    scatter = ax.scatter(
        coords[:, 0], coords[:, 1],
        c=domain_codes,
        cmap="tab20", s=spot_size, rasterized=True,
    )
    ax.set_title(f"Spatial Domains (n={n_domains})", fontsize=12)
    ax.set_aspect("equal")
    ax.axis("off")
    plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04, label="Domain")
    out_path = output_dir / "spatial_domains.pdf"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {out_path}")
    return out_path


def plot_deconvolution_spatial(
    st_adata,
    proportions: np.ndarray,
    cell_type_names: list[str],
    output_dir: Path,
    spot_size: float = 1.0,
    ncols: int = 4,
) -> Path:
    """Plot cell type proportion spatial heatmaps (stSCI-style).

    Parameters
    ----------
    st_adata : sc.AnnData
        AnnData with obsm["spatial"].
    proportions : np.ndarray
        Shape (n_spots, n_cell_types).
    cell_type_names : list[str]
        Cell type labels for subplot titles.
    output_dir : Path
        Directory to save PDF.
    spot_size : float
        Marker size.
    ncols : int
        Number of columns in subplot grid.

    Returns
    -------
    Path
        Output file path.
    """
    set_publication_style()
    n_types = len(cell_type_names)
    nrows = int(np.ceil(n_types / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 4))
    axes = axes.flatten()

    coords = st_adata.obsm["spatial"]
    for i, ct in enumerate(cell_type_names):
        ax = axes[i]
        sc_plot = ax.scatter(
            coords[:, 0], coords[:, 1],
            c=proportions[:, i], cmap="Blues", s=spot_size, rasterized=True,
        )
        ax.set_title(ct, fontsize=9)
        ax.set_aspect("equal")
        ax.axis("off")
        plt.colorbar(sc_plot, ax=ax, fraction=0.046, pad=0.04)

    for j in range(n_types, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    out_path = output_dir / "deconvolution_spatial.pdf"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {out_path}")
    return out_path


def plot_mapping_density(
    mapping_counts: np.ndarray,
    output_dir: Path,
) -> Path:
    """Plot histogram of mapping counts per cell.

    Parameters
    ----------
    mapping_counts : np.ndarray
        Per-cell mapping count, shape (n_cells,).
    output_dir : Path
        Directory to save PDF.

    Returns
    -------
    Path
        Output file path.
    """
    set_publication_style()
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.bar(range(len(mapping_counts)), mapping_counts)
    ax.set_xlabel("Cell index (scRNA)")
    ax.set_ylabel("# mapped spots")
    ax.set_title("SC<->ST Mapping Density")
    plt.tight_layout()
    out_path = output_dir / "mapping_density.pdf"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {out_path}")
    return out_path


def plot_bulk_vs_predicted(
    bulk_proportions: np.ndarray,
    predicted_proportions: np.ndarray,
    cell_type_names: list[str],
    condition_names: list[str],
    output_dir: Path,
) -> Path:
    """Compare bulk ssGSEA proportions vs OT-predicted proportions.

    If dimensions match, shows side-by-side bars. Otherwise plots
    ssGSEA alone (dimension-aware fallback from 16-01).

    Parameters
    ----------
    bulk_proportions : np.ndarray
        Shape (n_conditions, n_bulk_types).
    predicted_proportions : np.ndarray
        Shape (n_conditions, n_pred_types).
    cell_type_names : list[str]
        Cell type labels (from ssGSEA).
    condition_names : list[str]
        Condition labels.
    output_dir : Path
        Directory to save PDF.

    Returns
    -------
    Path
        Output file path.
    """
    set_publication_style()
    n_types = len(cell_type_names)
    n_conds = len(condition_names)

    if n_conds == 0:
        logger.warning("No conditions to plot for bulk comparison")
        return output_dir / "bulk_vs_predicted.pdf"

    if bulk_proportions.shape[1] == predicted_proportions.shape[1]:
        # Dimensions match: full side-by-side comparison
        fig, axes = plt.subplots(1, n_conds, figsize=(5 * n_conds, 5))
        if n_conds == 1:
            axes = [axes]
        for idx, cond in enumerate(condition_names):
            ax = axes[idx]
            x = np.arange(n_types)
            width = 0.35
            ax.bar(
                x - width / 2, bulk_proportions[idx],
                width, label="Bulk (ssGSEA)", alpha=0.8,
            )
            ax.bar(
                x + width / 2, predicted_proportions[idx],
                width, label="Predicted (OT)", alpha=0.8,
            )
            ax.set_xticks(x)
            ax.set_xticklabels(cell_type_names, rotation=90, fontsize=7)
            ax.set_ylabel("Proportion")
            ax.set_title(f"Condition: {cond}")
            ax.legend(fontsize=8)
    else:
        # Dimension mismatch: plot ssGSEA bulk composition only
        logger.info(
            f"  ssGSEA ({bulk_proportions.shape[1]} types) vs OT "
            f"({predicted_proportions.shape[1]} types) dimension mismatch, "
            f"plotting ssGSEA only",
        )
        fig, axes = plt.subplots(1, n_conds, figsize=(5 * n_conds, 5))
        if n_conds == 1:
            axes = [axes]
        for idx, cond in enumerate(condition_names):
            ax = axes[idx]
            x = np.arange(n_types)
            ax.bar(
                x, bulk_proportions[idx],
                alpha=0.8, label="Bulk (ssGSEA)",
            )
            ax.set_xticks(x)
            ax.set_xticklabels(cell_type_names, rotation=90, fontsize=7)
            ax.set_ylabel("Proportion")
            ax.set_title(f"Bulk Composition: {cond}")
            ax.legend(fontsize=8)

    plt.tight_layout()
    out_path = output_dir / "bulk_vs_predicted.pdf"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {out_path}")
    return out_path


def generate_he_overlays(
    st_adata,
    data_dir: Path,
    output_dir: Path,
    mapping_density: np.ndarray | None = None,
    he_max_dim: int = 4000,
) -> list[Path]:
    """Generate HE overlay figures for all samples.

    For each sample in st_adata.obs["sample"], produces:
    1. Domain segmentation overlay (he_domain_{sample_id}.pdf)
    2. Cell type deconvolution overlay (he_deconv_{sample_id}.pdf)
    3. Mapping density overlay if mapping_density provided (he_mapping_{sample_id}.pdf)

    Parameters
    ----------
    st_adata : sc.AnnData
        AnnData with obsm["spatial"], obs["domain"], obs["sample"],
        and obsm["cell_type_proportions"].
    data_dir : Path
        Directory containing HE TIF files.
    output_dir : Path
        Directory to save PDF figures.
    mapping_density : np.ndarray or None
        Per-spot mapping count array, shape (n_spots,).
        If provided, mapping overlay is generated for each sample.
        Must be in the same order as st_adata observations.
    he_max_dim : int
        Maximum HE image dimension after downsampling.

    Returns
    -------
    list[Path]
        Paths to all generated figure files.
    """
    from .he_overlay import (
        plot_he_deconvolution_overlay,
        plot_he_domain_overlay,
        plot_he_mapping_overlay,
    )

    cell_type_names = list(st_adata.obsm.get("cell_type_proportions_colnames", []))
    # Fallback: if column names not stored, use generic labels
    if not cell_type_names:
        n_types = st_adata.obsm["cell_type_proportions"].shape[1]
        cell_type_names = [f"CT{i}" for i in range(n_types)]

    samples = st_adata.obs["sample"].unique()
    output_paths: list[Path] = []

    for sample_id in samples:
        logger.info(f"Generating HE overlays for sample {sample_id}")

        # Domain overlay
        try:
            path = plot_he_domain_overlay(
                st_adata, sample_id, data_dir, output_dir, he_max_dim,
            )
            output_paths.append(path)
        except FileNotFoundError as e:
            logger.warning(f"Skipping {sample_id}: {e}")
            continue

        # Deconvolution overlay
        try:
            path = plot_he_deconvolution_overlay(
                st_adata, sample_id, cell_type_names,
                data_dir, output_dir, he_max_dim,
            )
            output_paths.append(path)
        except FileNotFoundError as e:
            logger.warning(f"Skipping deconvolution for {sample_id}: {e}")

        # Mapping density overlay
        if mapping_density is not None:
            mask = st_adata.obs["sample"] == sample_id
            sample_density = mapping_density[mask.values]
            try:
                path = plot_he_mapping_overlay(
                    st_adata, sample_id, sample_density,
                    data_dir, output_dir, he_max_dim,
                )
                output_paths.append(path)
            except FileNotFoundError as e:
                logger.warning(f"Skipping mapping for {sample_id}: {e}")

    logger.info(f"Generated {len(output_paths)} HE overlay figures")
    return output_paths
