"""Rosacea-specific data loading convenience wrapper.

Provides load_rosacea_dataset() function that configures TripleModalDataset
with correct paths and settings for Rosacea triple-modal data.

Usage:
    from tglue.data.rosacea_loader import load_rosacea_dataset
    dataset = load_rosacea_dataset()
"""

from __future__ import annotations

import logging
from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING

from .dataset import TripleModalDataset
from .preprocessing import convert_h5ad_to_zarr

if TYPE_CHECKING:
    from ..graph.guidance_graph import GuidanceGraph

logger = logging.getLogger(__name__)

# Rosacea data paths (canonical locations)
ROSACEA_DATA_DIR = Path("data/rosacea")
SCRNA_PATH = ROSACEA_DATA_DIR / "sc_reference.h5ad"
ST_PATH = ROSACEA_DATA_DIR / "spatial_100k.h5ad"  # D-11-01: sampled to 100K
BULK_PATH = ROSACEA_DATA_DIR / "array_test.h5ad"
GTF_PATH = ROSACEA_DATA_DIR / "mock_gtf.txt"  # Wave 0 blocker resolved
MARKERS_PATH = files("tglue.data.markers").joinpath("skin_markers.gmt")  # TDX-01: use importlib.resources


def load_rosacea_dataset(
    batch_size_sc: int = 128,
    batch_size_st: int = 128,
    is_validation: bool = False,
    preprocessed: bool = True,  # D-11: QC already computed, skip preprocessing
    device: str = "cpu",
    use_lazy_loading: bool = True,  # D-11-05: Zarr format for lazy loading
    validation_quadrant: int = 2,
    convert_to_zarr: bool = True,  # D-11-05: Convert H5AD to Zarr if needed
    guidance_graph: "GuidanceGraph | None" = None,  # Pre-built graph to reuse
) -> TripleModalDataset:
    """Load Rosacea triple-modal dataset with recommended settings.

    Parameters
    ----------
    batch_size_sc : int, default 128
        Batch size for scRNA samples.
    batch_size_st : int, default 128
        Batch size for ST spots.
    is_validation : bool, default False
        Whether to load validation split.
    preprocessed : bool, default True
        Skip QC preprocessing (Rosacea data already QC'd).
    device : str, default "cpu"
        Device for tensor placement.
    use_lazy_loading : bool, default True
        Enable Zarr lazy loading (D-11-05).
    validation_quadrant : int, default 2
        Quadrant for spatial validation split (D-05: SW quadrant).
    convert_to_zarr : bool, default True
        Convert H5AD to Zarr if cached Zarr doesn't exist.
    guidance_graph : GuidanceGraph, optional
        Pre-built guidance graph to reuse (avoids recomputation for val split).

    Returns
    -------
    TripleModalDataset
        Dataset instance configured for Rosacea data.

    Notes
    -----
    - scRNA: 76K cells (D-11-02: full data)
    - ST: 100K spots sampled from 345K (D-11-01)
    - Bulk: 58 samples (D-11-03: full data)
    - Shared genes: 17,825 (verified via shared_genes.txt)
    - Zarr lazy loading enabled (D-11-05)
    """
    # Verify blocker files exist (Wave 0 should have created these)
    if not GTF_PATH.exists():
        raise FileNotFoundError(
            f"GTF file missing: {GTF_PATH}. Run Wave 0 plan first."
        )
    if not MARKERS_PATH.exists():
        raise FileNotFoundError(
            f"Skin markers file missing: {MARKERS_PATH}. Run Wave 0 plan first."
        )

    # Verify sampled ST data exists
    if not ST_PATH.exists():
        raise FileNotFoundError(
            f"Sampled ST data missing: {ST_PATH}. Run Task 1 first."
        )

    # Convert to Zarr if needed (D-11-05)
    if convert_to_zarr and use_lazy_loading:
        # TripleModalDataset.__init__ handles conversion internally via D-10
        pass  # Auto-conversion handled by dataset

    logger.info(
        f"Loading Rosacea dataset: scRNA={SCRNA_PATH}, ST={ST_PATH}, "
        f"Bulk={BULK_PATH}, GTF={GTF_PATH}"
    )

    dataset = TripleModalDataset(
        scRNA_path=str(SCRNA_PATH),
        st_path=str(ST_PATH),
        bulk_path=str(BULK_PATH),
        gtf_path=str(GTF_PATH),
        batch_size_sc=batch_size_sc,
        batch_size_st=batch_size_st,
        is_validation=is_validation,
        preprocessed=preprocessed,
        device=device,
        validation_quadrant=validation_quadrant,
        use_lazy_loading=use_lazy_loading,
        guidance_graph=guidance_graph,
    )

    logger.info(
        f"Rosacea dataset loaded: {dataset.adata_sc.n_obs} cells, "
        f"{dataset.adata_st.n_obs} spots, {dataset.adata_bulk.n_obs} samples, "
        f"{dataset.n_genes} shared genes"
    )

    return dataset