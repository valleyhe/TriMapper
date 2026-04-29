"""Erdos-Renyi random graph baseline for AB-01 (D-AB01).

Replaces guidance graph topology with random edges while maintaining:
- Same gene count (n_genes)
- Same edge density (p = n_edges / n_possible_edges)
- No self-loops (diagonal = 0)
- Undirected (symmetric adjacency)

Pattern: scipy.sparse.random -> setdiag(0) -> make symmetric
"""

from __future__ import annotations

from typing import List

import numpy as np
import scipy.sparse
import torch
from torch import Tensor

from tglue.graph.guidance_graph import GuidanceGraph


def build_erdos_renyi_baseline(
    guidance_graph: GuidanceGraph,
    seed: int = 42,
) -> GuidanceGraph:
    """Build Erdos-Renyi random graph matching target statistics.

    Replaces genomic/co-expression edges with random edges at the same density.
    Used for AB-01: Remove guidance graph structure.

    Parameters
    ----------
    guidance_graph : GuidanceGraph
        Original guidance graph to match statistics
    seed : int, default 42
        Random seed for reproducibility

    Returns
    -------
    GuidanceGraph
        Random graph with same gene_list, same edge density, no structure
    """
    n_genes = len(guidance_graph.gene_list)

    # Calculate edge density from original graph
    # Original graph is undirected, so count unique edges
    n_edges_directed = guidance_graph.edge_index.shape[1]
    n_edges_undirected = n_edges_directed // 2  # Each edge stored twice

    # Edge density for undirected graph: p = n_edges / (n * (n-1) / 2)
    n_possible_edges = n_genes * (n_genes - 1) // 2
    density = n_edges_undirected / n_possible_edges if n_possible_edges > 0 else 0.0

    # Generate random adjacency matrix with scipy.sparse.random
    rng = np.random.default_rng(seed)

    # Create sparse random matrix (upper triangular to avoid duplicates)
    adj_upper = scipy.sparse.random(
        n_genes,
        n_genes,
        density=density,
        format="csr",
        random_state=rng,
        dtype=np.float32,
    )

    # Set diagonal to zero (no self-loops, Pitfall 4)
    adj_upper.setdiag(0)

    # Make symmetric (undirected)
    adj = adj_upper + adj_upper.T

    # Convert to edge_index format (2, n_edges)
    adj_coo = adj.tocoo()
    edge_index = torch.tensor(
        np.stack([adj_coo.row, adj_coo.col]),
        dtype=torch.long,
    )

    # Edge weights: uniform 1.0
    edge_weight = torch.ones(adj_coo.nnz, dtype=torch.float32)

    # Edge types: all 'random'
    edge_type = ["random"] * adj_coo.nnz

    return GuidanceGraph(
        edge_index=edge_index,
        edge_weight=edge_weight,
        edge_type=edge_type,
        gene_list=guidance_graph.gene_list,
    )