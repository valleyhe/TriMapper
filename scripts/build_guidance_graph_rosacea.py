#!/usr/bin/env python3
"""Build guidance graph from Rosacea triple-modal data.

Task 1 of 12-01-PLAN.md: Construct guidance graph from scRNA co-expression
and mock_gtf genomic annotations for triple-modal VAE training.

Optimized version: Uses vectorized numpy correlation and cell sampling
for faster computation with 17,825 genes.
"""

import sys
sys.path.insert(0, "/home/scu/stSCI-GLUE-workspace/src")

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import scanpy as sc
import numpy as np
from scipy.stats import spearmanr

from tglue.graph.guidance_graph import build_genomic_edges, GuidanceGraph
from tglue.graph.genes import load_gtf_annotations

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def build_coexpr_edges_fast(
    gene_list: list,
    X: np.ndarray,
    threshold: float = 0.3,
    n_cells_sample: int = 10000,
) -> list:
    """Build co-expression edges using vectorized Spearman correlation.

    Uses subsampling and vectorized computation for efficiency:
    - Samples n_cells_sample cells to reduce computation
    - Uses Spearman correlation (rank-based, faster with sparse data)
    - Returns edges where rho > threshold

    Parameters
    ----------
    gene_list : list
        List of gene names matching X columns.
    X : np.ndarray
        Expression matrix of shape (n_cells, n_genes).
    threshold : float
        Correlation threshold (Spearman rho > threshold).
    n_cells_sample : int
        Number of cells to sample for correlation computation.

    Returns
    -------
    list
        List of (gene_i, gene_j, rho) tuples.
    """
    n_genes = X.shape[1]
    logger.info(f"Computing co-expression on {n_cells_sample} sampled cells...")

    # Sample cells
    if X.shape[0] > n_cells_sample:
        idx = np.random.choice(X.shape[0], n_cells_sample, replace=False)
        X = X[idx]
        logger.info(f"Sampled {n_cells_sample} cells for correlation computation")

    # Normalize: rank-transform each column (Spearman is rank-based)
    from scipy.stats import rankdata
    X_rank = np.column_stack([rankdata(X[:, i]) for i in range(n_genes)])

    # Compute correlation matrix efficiently using matrix multiplication
    # Spearman rho = Pearson correlation of ranks
    n = X_rank.shape[0]
    X_centered = X_rank - X_rank.mean(axis=0)
    X_std = X_rank.std(axis=0)
    X_std[X_std == 0] = 1  # Avoid division by zero

    # Correlation matrix: C = (X_centered^T @ X_centered) / (n-1)
    C = (X_centered.T @ X_centered) / (n - 1)
    C = C / (X_std[:, None] * X_std[None, :])

    # Extract upper triangle (i < j)
    edges = []
    for i in range(n_genes):
        for j in range(i + 1, n_genes):
            rho = C[i, j]
            if rho > threshold:
                edges.append((gene_list[i], gene_list[j], float(rho)))

    logger.info(f"Found {len(edges)} co-expression edges (rho > {threshold})")
    return edges


def main():
    """Build and save guidance graph statistics from Rosacea data."""

    # Data paths
    data_dir = Path("/home/scu/stSCI-GLUE-workspace/data/rosacea")
    scRNA_path = data_dir / "sc_reference.h5ad"
    st_path = data_dir / "spatial_100k.h5ad"
    bulk_path = data_dir / "array_test.h5ad"
    gtf_path = data_dir / "mock_gtf.txt"
    stats_path = data_dir / "graph_stats.json"

    # Verify files exist
    for p in [scRNA_path, st_path, bulk_path, gtf_path]:
        if not p.exists():
            raise FileNotFoundError(f"Required file missing: {p}")

    logger.info("Loading Rosacea triple-modal data...")

    # Load AnnData objects
    scRNA_adata = sc.read_h5ad(scRNA_path)
    st_adata = sc.read_h5ad(st_path)
    bulk_adata = sc.read_h5ad(bulk_path)

    logger.info(f"Loaded dimensions:")
    logger.info(f"  scRNA: {scRNA_adata.n_obs} cells x {scRNA_adata.n_vars} genes")
    logger.info(f"  ST: {st_adata.n_obs} spots x {st_adata.n_vars} genes")
    logger.info(f"  Bulk: {bulk_adata.n_obs} samples x {bulk_adata.n_vars} genes")

    # Gene harmonization per D-04
    from tglue.graph.genes import harmonize_genes

    scRNA_genes = list(scRNA_adata.var.index)
    st_genes = list(st_adata.var.index)
    bulk_genes = list(bulk_adata.var.index)

    common_sc_st, _ = harmonize_genes(scRNA_genes, st_genes, min_shared=2000)
    common_sc_bulk, _ = harmonize_genes(scRNA_genes, bulk_genes, min_shared=2000)

    st_set = set(common_sc_st)
    bulk_set = set(common_sc_bulk)
    canonical_genes = [g for g in scRNA_genes if g in st_set and g in bulk_set]

    logger.info(f"Canonical gene list: {len(canonical_genes)} genes")

    # Build genomic edges per D-01 (window_bp = 150,000)
    logger.info("Building genomic proximity edges (window=150kb)...")
    annotation = load_gtf_annotations(str(gtf_path))
    genomic_edges = build_genomic_edges(annotation, window_bp=150_000)
    logger.info(f"Genomic edges: {len(genomic_edges)}")

    # Build co-expression edges per D-01 (threshold = 0.3)
    logger.info("Building co-expression edges (rho > 0.3)...")

    # Subset scRNA X to canonical genes
    sc_gene_to_col = {g: i for i, g in enumerate(scRNA_genes)}
    col_indices = [sc_gene_to_col[g] for g in canonical_genes if g in sc_gene_to_col]
    sc_X = scRNA_adata.X[:, col_indices]
    if hasattr(sc_X, "toarray"):
        sc_X = sc_X.toarray()
    sc_X = np.asarray(sc_X, dtype=np.float32)

    # Sample 10,000 cells for faster correlation computation
    coexpr_edges = build_coexpr_edges_fast(
        canonical_genes,
        sc_X,
        threshold=0.3,
        n_cells_sample=10000,
    )

    logger.info(f"Co-expression edges: {len(coexpr_edges)}")

    # Build GuidanceGraph per D-01 (both edge types required)
    logger.info("Building GuidanceGraph...")

    if not genomic_edges:
        raise ValueError("No genomic edges produced")
    if not coexpr_edges:
        raise ValueError("No co-expression edges produced")

    graph = GuidanceGraph.from_edges(genomic_edges, coexpr_edges, canonical_genes)

    # Verify both edge types present
    if not graph.has_edge_type("genomic"):
        raise ValueError("GuidanceGraph missing 'genomic' edge type")
    if not graph.has_edge_type("coexpr"):
        raise ValueError("GuidanceGraph missing 'coexpr' edge type")

    logger.info(f"Guidance graph built: {graph}")

    # Get edge counts by type
    n_genomic = sum(1 for t in graph.edge_type if t == "genomic")
    n_coexpr = sum(1 for t in graph.edge_type if t == "coexpr")
    n_total = len(graph.edge_type)

    logger.info(f"Edge statistics:")
    logger.info(f"  Genomic edges: {n_genomic}")
    logger.info(f"  Co-expression edges: {n_coexpr}")
    logger.info(f"  Total edges: {n_total}")

    # Convert to PyG Data for VAE consumption
    guidance_data = graph.to_data()
    edge_index_shape = tuple(guidance_data.edge_index.shape)

    logger.info(f"PyG Data object:")
    logger.info(f"  edge_index shape: {edge_index_shape}")
    logger.info(f"  num_nodes: {guidance_data.num_nodes}")

    # Build statistics JSON
    stats = {
        "n_genes": len(graph.gene_list),
        "n_genomic_edges": n_genomic,
        "n_coexpr_edges": n_coexpr,
        "n_total_edges": n_total,
        "edge_index_shape": list(edge_index_shape),
        "build_timestamp": datetime.now(timezone.utc).isoformat(),
        "parameters": {
            "genomic_window_bp": 150_000,
            "coexpr_threshold": 0.3,
            "coexpr_n_cells_sampled": 10000
        },
        "validation": {
            "has_genomic_edges": n_genomic > 0,
            "has_coexpr_edges": n_coexpr > 0,
            "min_genomic_threshold": 10000,
            "min_coexpr_threshold": 50000,
            "genomic_threshold_passed": n_genomic >= 10000,
            "coexpr_threshold_passed": n_coexpr >= 50000
        }
    }

    # Save statistics
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    logger.info(f"Graph statistics saved to: {stats_path}")

    # Validation summary
    logger.info("Validation summary:")
    logger.info(f"  n_genes = {stats['n_genes']}")
    logger.info(f"  genomic_threshold_passed = {stats['validation']['genomic_threshold_passed']}")
    logger.info(f"  coexpr_threshold_passed = {stats['validation']['coexpr_threshold_passed']}")

    # Final sanity check
    assert stats["n_genes"] == 17825, f"Gene count mismatch: expected 17825, got {stats['n_genes']}"
    assert stats["validation"]["has_genomic_edges"], "Missing genomic edges"
    assert stats["validation"]["has_coexpr_edges"], "Missing co-expression edges"

    logger.info("Task 1 complete: Guidance graph built successfully")

    return graph, stats


if __name__ == "__main__":
    main()