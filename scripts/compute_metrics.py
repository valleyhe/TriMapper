#!/usr/bin/env python3
"""Compute ASW/NMI/ARI metrics for TripleModalVAE results."""

import json
import sys
from pathlib import Path

import numpy as np
import scanpy as sc
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tglue.evaluation.metrics import (
    compute_asw,
    compute_spatial_ari,
    compute_spatial_nmi,
)


def main():
    results_dir = Path("results/intergrated")
    output_file = results_dir / "metrics.json"

    print("Loading embeddings...")

    # Load ST with domains
    st_full = sc.read_h5ad(results_dir / "st_full_results.h5ad", backed="r")
    u_st = st_full.obsm["X_embedding"]
    domains = st_full.obs["domain"].values.astype(str)
    print(f"  ST: {u_st.shape}, domains: {len(np.unique(domains))}")

    # Load scRNA
    sc_adata = sc.read_h5ad(results_dir / "scrna_with_embeddings.h5ad", backed="r")
    u_sc = sc_adata.obsm["X_embedding"]
    print(f"  scRNA: {u_sc.shape}")

    # Load Bulk
    bulk_adata = sc.read_h5ad(results_dir / "bulk_with_embeddings.h5ad", backed="r")
    u_bulk = bulk_adata.obsm["X_embedding"]
    print(f"  Bulk: {u_bulk.shape}")

    # --- Spatial Domain Metrics (ST only) ---
    print("\nComputing spatial domain metrics...")
    # Convert domain strings to integers
    unique_domains = np.unique(domains)
    domain_map = {d: i for i, d in enumerate(unique_domains)}
    domain_labels = np.array([domain_map[d] for d in domains])

    ari = compute_spatial_ari(torch.from_numpy(u_st), domain_labels)
    nmi = compute_spatial_nmi(torch.from_numpy(u_st), domain_labels)
    print(f"  ARI (ST vs Leiden): {ari:.4f}")
    print(f"  NMI (ST vs Leiden): {nmi:.4f}")

    # --- Modality Alignment Metrics (all modalities) ---
    print("\nComputing modality alignment metrics...")
    # Combine all embeddings
    u_all = np.vstack([u_st, u_sc, u_bulk])
    modality_labels = np.concatenate([
        np.zeros(len(u_st), dtype=int),      # ST = 0
        np.ones(len(u_sc), dtype=int),       # scRNA = 1
        np.full(len(u_bulk), 2, dtype=int),  # Bulk = 2
    ])
    print(f"  Combined: {u_all.shape}, labels: {np.bincount(modality_labels)}")

    # Sample for ASW (sklearn silhouette is O(n^2), too slow for 530K)
    sample_size = min(5000, len(u_all))
    rng = np.random.default_rng(42)
    idx = rng.choice(len(u_all), size=sample_size, replace=False)
    u_sample = u_all[idx]
    labels_sample = modality_labels[idx]

    # Need at least 2 samples per label for silhouette
    from sklearn.metrics import silhouette_score
    try:
        asw = silhouette_score(u_sample, labels_sample)
        print(f"  ASW (modality separation, n={sample_size}): {asw:.4f}")
    except Exception as e:
        print(f"  ASW failed: {e}")
        asw = None

    # --- NMI for modality alignment (k-means on combined vs modality labels) ---
    from sklearn.cluster import KMeans
    n_clusters = 3
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    # Sample for k-means too (530K is too large)
    kmeans_sample_size = min(50000, len(u_all))
    idx_km = rng.choice(len(u_all), size=kmeans_sample_size, replace=False)
    u_km = u_all[idx_km]
    labels_km = modality_labels[idx_km]

    pred_labels = kmeans.fit_predict(u_km)
    from sklearn.metrics import normalized_mutual_info_score
    nmi_modality = normalized_mutual_info_score(labels_km, pred_labels)
    print(f"  NMI (modality alignment, n={kmeans_sample_size}): {nmi_modality:.4f}")

    # Save metrics
    metrics = {
        "spatial": {
            "ari": float(ari),
            "nmi": float(nmi),
            "n_domains": int(len(unique_domains)),
        },
        "modality_alignment": {
            "asw": float(asw) if asw is not None else None,
            "nmi": float(nmi_modality),
            "sample_size_asw": sample_size,
            "sample_size_nmi": kmeans_sample_size,
        },
        "data": {
            "n_spots": int(len(u_st)),
            "n_cells": int(len(u_sc)),
            "n_bulk": int(len(u_bulk)),
        },
    }

    with open(output_file, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nMetrics saved to: {output_file}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
