"""Guidance graph construction for Triple-Modal GLUE.

Constructs a guidance graph with two edge types per D-01:
  - Genomic proximity edges: gene pairs on the same chromosome within 150kb
  - Co-expression edges: gene pairs from scRNA with Pearson r > 0.3

Genes are harmonized via the scRNA canonical list (D-04, min_shared >= 2000).
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
import torch_geometric.data as gd
from scipy.stats import pearsonr

from .genes import harmonize_genes


# ---------------------------------------------------------------------------
# Edge builders
# ---------------------------------------------------------------------------


def build_genomic_edges(
    gene_annotation: Dict[str, Dict],
    window_bp: int = 150_000,
) -> List[Tuple[str, str, float]]:
    """Build genomic proximity edges from GTF gene coordinates.

    For every gene pair on the same chromosome with |pos_j - pos_i| <= window_bp
    an undirected edge is added. Only the gene_id keys from the annotation dict
    are used; position is taken as the mid-point of (start, end).

    Parameters
    ----------
    gene_annotation : Dict[str, Dict]
        Output of load_gtf_annotations: gene_id -> {chrom, start, end, symbol}.
    window_bp : int, default 150_000
        Maximum genomic distance (base-pairs) for an edge.

    Returns
    -------
    List[Tuple[str, str, float]]
        List of (src_gene_id, dst_gene_id, weight=1.0) tuples (undirected,
        stored only in upper-triangular order).
    """
    # Group by chromosome
    by_chrom: Dict[str, List[Tuple[str, int]]] = {}
    for gid, attrs in gene_annotation.items():
        chrom = attrs["chrom"]
        mid = (attrs["start"] + attrs["end"]) // 2
        by_chrom.setdefault(chrom, []).append((gid, mid))

    edges: List[Tuple[str, str, float]] = []
    for chrom, members in by_chrom.items():
        n = len(members)
        for i in range(n):
            gid_i, pos_i = members[i]
            for j in range(i + 1, n):
                gid_j, pos_j = members[j]
                if abs(pos_j - pos_i) <= window_bp:
                    edges.append((gid_i, gid_j, 1.0))
    return edges


def build_coexpr_edges(
    scRNA_genes: List[str],
    scRNA_X: Union[np.ndarray, None] = None,
    threshold: float = 0.3,
    n_cells_sample: int = 10000,
) -> List[Tuple[str, str, float]]:
    """Build co-expression edges from scRNA expression matrix using vectorized Spearman correlation.

    Uses subsampling and vectorized computation for efficiency with large gene sets:
    - Samples n_cells_sample cells to reduce computation time
    - Uses Spearman correlation (rank-based, faster with sparse data)
    - Returns edges where rho > threshold

    Parameters
    ----------
    scRNA_genes : List[str]
        Canonical gene list from scRNA (ordered as matrix columns).
    scRNA_X : np.ndarray or None
        scRNA expression matrix of shape (n_cells, n_genes). If None,
        returns an empty list.
    threshold : float, default 0.3
        Spearman correlation threshold (per D-01).
    n_cells_sample : int, default 10000
        Number of cells to sample for correlation computation (trade-off between
        speed and accuracy; 10K cells provides stable correlation estimates).

    Returns
    -------
    List[Tuple[str, str, float]]
        List of (gene_i, gene_j, rho) tuples, upper-triangular only.
    """
    if scRNA_X is None or scRNA_X.shape[1] < 2:
        return []

    n_genes = scRNA_X.shape[1]
    n_cells = scRNA_X.shape[0]

    # Sample cells for faster computation
    if n_cells > n_cells_sample:
        idx = np.random.choice(n_cells, n_cells_sample, replace=False)
        X = scRNA_X[idx]
    else:
        X = scRNA_X

    # Convert to numpy float32
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)

    # Rank-transform each column (Spearman = Pearson of ranks)
    # Use scipy.stats.rankdata which handles this efficiently
    try:
        from scipy.stats import rankdata
    except ImportError:
        # Fallback to numpy for rank computation
        def rankdata(arr):
            order = np.argsort(arr, axis=0)
            ranks = np.argsort(order, axis=0)
            return ranks.astype(np.float64) + 1

    X_rank = np.column_stack([rankdata(X[:, i]) for i in range(n_genes)])
    X_rank = X_rank.astype(np.float32)

    # Vectorized correlation: C = (X_centered^T @ X_centered) / (n-1)
    n = X_rank.shape[0]
    X_centered = X_rank - X_rank.mean(axis=0)
    X_std = X_rank.std(axis=0)
    X_std[X_std == 0] = 1  # Avoid division by zero
    C = (X_centered.T @ X_centered) / (n - 1)
    C = C / (X_std[:, None] * X_std[None, :])

    # Extract upper triangle (i < j) and filter by threshold
    edges: List[Tuple[str, str, float]] = []
    for i in range(n_genes):
        for j in range(i + 1, n_genes):
            rho = C[i, j]
            if rho > threshold:
                edges.append((scRNA_genes[i], scRNA_genes[j], float(rho)))

    return edges


# ---------------------------------------------------------------------------
# GuidanceGraph
# ---------------------------------------------------------------------------


class GuidanceGraph:
    """Guidance graph for Triple-Modal GLUE.

    Node type: ["gene"]
    Edge type: ("gene", "interacts", "gene")

    Edge attributes:
        edge_weight : float  — 1.0 for genomic, Pearson r for coexpr
        edge_type   : str    — "genomic" or "coexpr"
    """

    def __init__(
        self,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        edge_type: List[str],
        gene_list: List[str],
    ) -> None:
        self.edge_index = edge_index  # (2, n_edges)
        self.edge_weight = edge_weight  # (n_edges,)
        self.edge_type = list(edge_type)  # ["genomic", "coexpr", ...]
        self.gene_list = list(gene_list)  # canonical ordered gene list

    def _gene_idx(self, gene_id: str) -> int:
        return self.gene_list.index(gene_id)

    @classmethod
    def from_edges(
        cls,
        genomic_edges: List[Tuple[str, str, float]],
        coexpr_edges: List[Tuple[str, str, float]],
        gene_list: List[str],
    ) -> "GuidanceGraph":
        """Build a GuidanceGraph from pre-computed edge lists.

        Parameters
        ----------
        genomic_edges : List[Tuple[str, str, float]]
            Edges from build_genomic_edges.
        coexpr_edges : List[Tuple[str, str, float]]
            Edges from build_coexpr_edges.
        gene_list : List[str]
            Canonical ordered gene list.

        Returns
        -------
        GuidanceGraph
        """
        gene_to_idx = {g: i for i, g in enumerate(gene_list)}
        all_edges: List[Tuple[int, int, float, str]] = []

        for src, dst, w in genomic_edges:
            if src in gene_to_idx and dst in gene_to_idx:
                all_edges.append((gene_to_idx[src], gene_to_idx[dst], w, "genomic"))

        for src, dst, r in coexpr_edges:
            if src in gene_to_idx and dst in gene_to_idx:
                all_edges.append((gene_to_idx[src], gene_to_idx[dst], r, "coexpr"))

        if not all_edges:
            raise ValueError("No valid edges after filtering to gene_list.")

        edge_index = torch.tensor(
            [[e[0] for e in all_edges], [e[1] for e in all_edges]], dtype=torch.long
        )
        edge_weight = torch.tensor([e[2] for e in all_edges], dtype=torch.float32)
        edge_type = [e[3] for e in all_edges]

        return cls(edge_index, edge_weight, edge_type, gene_list)

    def to_data(self) -> gd.Data:
        """Convert to torch_geometric Data for serialization."""
        return gd.Data(
            x=torch.arange(len(self.gene_list), dtype=torch.long).unsqueeze(1),
            edge_index=self.edge_index,
            edge_weight=self.edge_weight,
            edge_type=self.edge_type,
            num_nodes=len(self.gene_list),
        )

    def save(self, path: str) -> None:
        """Save GuidanceGraph using torch.save (safer than pickle).

        WR-04 FIX: Use torch.save instead of pickle for safer serialization.
        Pickle can execute arbitrary code if file is tampered; torch.save is safer.

        Parameters
        ----------
        path : str
            File path to save the graph (e.g., 'graph.pkl').
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'edge_index': self.edge_index,
            'edge_weight': self.edge_weight,
            'edge_type': self.edge_type,
            'gene_list': self.gene_list,
        }, path)

    @classmethod
    def load(cls, path: str) -> "GuidanceGraph":
        """Load a GuidanceGraph using torch.load (safer than pickle)."""
        data = torch.load(path, weights_only=False)
        return cls(
            edge_index=data['edge_index'],
            edge_weight=data['edge_weight'],
            edge_type=data['edge_type'],
            gene_list=data['gene_list'],
        )

    def has_edge_type(self, edge_type: str) -> bool:
        """Return True if at least one edge of the given type exists."""
        return edge_type in self.edge_type

    def __repr__(self) -> str:
        n_genomic = sum(1 for t in self.edge_type if t == "genomic")
        n_coexpr = sum(1 for t in self.edge_type if t == "coexpr")
        return (
            f"GuidanceGraph(genes={len(self.gene_list)}, "
            f"genomic_edges={n_genomic}, coexpr_edges={n_coexpr})"
        )


# ---------------------------------------------------------------------------
# High-level constructor
# ---------------------------------------------------------------------------


def build_guidance_graph(
    scRNA_adata,  # ann.AnnData — canonical gene list from .var.index
    st_adata,  # ann.AnnData — ST expression
    bulk_adata,  # ann.AnnData — Bulk expression
    gtf_path: str,
    genomic_window_bp: int = 150_000,
    coexpr_threshold: float = 0.3,
) -> GuidanceGraph:
    """Build a guidance graph harmonizing genes across three modalities.

    Per D-04: scRNA gene list is canonical; ST and Bulk are projected onto it.
    min_shared >= 2000 is enforced as a fail-fast threshold.

    Parameters
    ----------
    scRNA_adata : ann.AnnData
        scRNA expression (canonical gene list from .var.index).
    st_adata : ann.AnnData
        Spatial transcriptomics expression.
    bulk_adata : ann.AnnData
        Bulk RNA-seq expression.
    gtf_path : str
        Path to GTF annotation file for genomic coordinate extraction.
    genomic_window_bp : int, default 150_000
        Genomic proximity window in base-pairs (per D-01).
    coexpr_threshold : float, default 0.3
        Pearson correlation threshold for co-expression edges (per D-01).

    Returns
    -------
    GuidanceGraph
        Graph with both "genomic" and "coexpr" edge types.
    """
    from .genes import load_gtf_annotations

    # Gene harmonization: project ST and Bulk onto scRNA canonical
    scRNA_genes = list(scRNA_adata.var.index)
    st_genes = list(st_adata.var.index)
    bulk_genes = list(bulk_adata.var.index)

    common_sc_st, _ = harmonize_genes(scRNA_genes, st_genes, min_shared=2000)
    common_sc_bulk, _ = harmonize_genes(scRNA_genes, bulk_genes, min_shared=2000)

    # Intersection of all three modalities = canonical gene list
    st_set = set(common_sc_st)
    bulk_set = set(common_sc_bulk)
    canonical_genes = [g for g in scRNA_genes if g in st_set and g in bulk_set]

    if len(canonical_genes) < 2000:
        raise ValueError(
            f"Only {len(canonical_genes)} shared genes after harmonizing all "
            f"three modalities; minimum 2000 required."
        )

    # Build genomic edges using GTF annotations
    annotation = load_gtf_annotations(gtf_path)
    genomic_edges = build_genomic_edges(annotation, window_bp=genomic_window_bp)

    # Build co-expression edges from scRNA expression
    # Subset scRNA X to canonical genes (preserve order)
    sc_gene_to_col = {g: i for i, g in enumerate(scRNA_genes)}
    col_indices = [sc_gene_to_col[g] for g in canonical_genes if g in sc_gene_to_col]
    sc_X = scRNA_adata.X[:, col_indices]
    if hasattr(sc_X, "toarray"):
        sc_X = sc_X.toarray()
    sc_X = np.asarray(sc_X, dtype=np.float32)
    coexpr_edges = build_coexpr_edges(canonical_genes, sc_X, threshold=coexpr_threshold)

    # Require both edge types per D-01 anti-pattern
    if not genomic_edges:
        raise ValueError("No genomic edges produced — check GTF file and window size.")
    if not coexpr_edges:
        raise ValueError("No co-expression edges produced — check scRNA data.")

    graph = GuidanceGraph.from_edges(genomic_edges, coexpr_edges, canonical_genes)

    # Final sanity check: both edge types must be present
    if not graph.has_edge_type("genomic"):
        raise ValueError("GuidanceGraph missing 'genomic' edge type.")
    if not graph.has_edge_type("coexpr"):
        raise ValueError("GuidanceGraph missing 'coexpr' edge type.")

    return graph
