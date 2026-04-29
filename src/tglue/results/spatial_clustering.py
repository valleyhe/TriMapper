"""Spatial domain segmentation via Leiden clustering (D-06).

Extracted from scripts/generate_all_results.py run_leiden_clustering().
"""

from __future__ import annotations

import logging

import scanpy as sc
import squidpy as sq

logger = logging.getLogger(__name__)


def run_leiden(
    adata: sc.AnnData,
    resolution: float = 1.0,
    use_rep: str = "X_embedding",
    use_spatial_neighbors: bool = True,
    n_neighbors: int = 15,
) -> sc.AnnData:
    """Run Leiden clustering on latent embeddings or spatial coordinates.

    Parameters
    ----------
    adata : sc.AnnData
        AnnData with latent embeddings in ``obsm[use_rep]`` and spatial coords in ``obsm["spatial"]``.
    resolution : float
        Leiden resolution parameter. Higher = more clusters.
    use_rep : str
        Key in ``obsm`` containing the embedding.
    use_spatial_neighbors : bool
        If True and "spatial" in obsm, use spatial coordinates for k-NN graph (faster for large datasets).
        If False, use latent embedding for neighbors.
    n_neighbors : int
        Number of neighbors for spatial graph (default 15, suitable for Visium).

    Returns
    -------
    sc.AnnData
        The input adata with ``obs["domain"]`` column added.
    """
    # Use spatial neighbors if available (much faster for large spatial datasets)
    if use_spatial_neighbors and "spatial" in adata.obsm:
        logger.info(
            f"Running Leiden clustering (resolution={resolution}, spatial neighbors)"
        )
        # Spatial k-NN graph based on coordinates (fast: O(n) instead of O(n²))
        sq.gr.spatial_neighbors(adata, n_neighs=n_neighbors, coord_type="generic")
        # Use spatial neighbors for Leiden (key='spatial_neighbors')
        sc.tl.leiden(
            adata, resolution=resolution, key_added="domain", neighbors_key="spatial_neighbors"
        )
    else:
        logger.info(
            f"Running Leiden clustering (resolution={resolution}, rep={use_rep})"
        )
        sc.pp.neighbors(adata, use_rep=use_rep, n_neighbors=n_neighbors)
        sc.tl.leiden(adata, resolution=resolution, key_added="domain")

    n_domains = adata.obs["domain"].nunique()
    logger.info(f"  Found {n_domains} spatial domains")
    return adata
