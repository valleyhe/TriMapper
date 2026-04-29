"""Leiden clustering evaluation for spatial transcriptomics.

Per D-14-03: Compute ARI/NMI against spatial domain annotations using Leiden clustering.

Key pattern from RESEARCH.md:
- Always call sc.pp.neighbors() before sc.tl.leiden() (Pitfall 1)
- Use condition labels (Normal/Rosacea) as spatial_labels proxy
"""

from __future__ import annotations

import logging

import numpy as np
import scanpy as sc
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

logger = logging.getLogger(__name__)


def evaluate_leiden_clustering(
    latent: np.ndarray,
    spatial_labels: np.ndarray,
    resolution: float = 1.0,
    n_neighbors: int = 15,
) -> dict:
    """Compute Leiden clustering ARI/NMI vs spatial domain annotations.

    Per D-14-03 and RESEARCH.md Pattern 3.

    Parameters
    ----------
    latent : np.ndarray
        (n_spots, latent_dim) ST latent embeddings
    spatial_labels : np.ndarray
        (n_spots,) True spatial domain labels (e.g., condition labels: Normal=0, Rosacea=1)
    resolution : float
        Leiden resolution parameter (default 1.0)
    n_neighbors : int
        k-NN neighbors for graph construction (default 15)

    Returns
    -------
    dict with 'ari', 'nmi', 'n_leiden_clusters', 'n_true_clusters'
    """
    # Create AnnData with latent embeddings
    adata = sc.AnnData(X=latent)

    # Build k-NN graph (REQUIRED before Leiden - Pitfall 1)
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep="X")

    # Run Leiden clustering
    sc.tl.leiden(adata, resolution=resolution, key_added="leiden")

    # Get predicted cluster labels
    leiden_labels = adata.obs["leiden"].values.astype(int)

    # Compute ARI and NMI against true spatial labels
    ari = adjusted_rand_score(spatial_labels, leiden_labels)
    nmi = normalized_mutual_info_score(spatial_labels, leiden_labels)

    logger.info(
        f"Leiden clustering: {len(np.unique(leiden_labels))} clusters, "
        f"ARI={ari:.4f}, NMI={nmi:.4f}"
    )

    return {
        "ari": ari,
        "nmi": nmi,
        "n_leiden_clusters": len(np.unique(leiden_labels)),
        "n_true_clusters": len(np.unique(spatial_labels)),
    }