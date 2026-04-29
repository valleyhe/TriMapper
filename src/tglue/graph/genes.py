"""Gene harmonization utilities for Triple-Modal GLUE.

Gene harmonization follows D-04: scRNA gene list is canonical;
ST and Bulk are projected onto it. Genes not in the canonical list
are dropped (not created).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple


def load_gtf_annotations(gtf_path: str) -> Dict[str, Dict]:
    """Parse a GTF file and return a dict mapping gene_id to gene attributes.

    Parameters
    ----------
    gtf_path : str
        Path to GTF annotation file.

    Returns
    -------
    Dict[str, Dict]
        Mapping from gene_id -> {chrom, start, end, symbol}.
        start/end are 0-based integer coordinates.
    """
    annotations: Dict[str, Dict] = {}
    pattern = re.compile(r'gene_id "([^"]+)".*?gene_name "([^"]+)";')

    with open(gtf_path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9:
                continue
            chrom, kind, start_str, end_str = fields[0], fields[2], fields[3], fields[4]
            if kind != "gene":
                continue
            attrs_str = fields[8]
            match = pattern.search(attrs_str)
            if not match:
                continue
            gene_id = match.group(1)
            symbol = match.group(2)
            annotations[gene_id] = {
                "chrom": chrom,
                "start": int(start_str) - 1,  # convert to 0-based
                "end": int(end_str) - 1,
                "symbol": symbol,
            }
    return annotations


def harmonize_genes(
    scRNA_genes: List[str],
    other_genes: List[str],
    min_shared: int = 2000,
) -> Tuple[List[str], List[str]]:
    """Project another modality's gene list onto the canonical scRNA list.

    Parameters
    ----------
    scRNA_genes : List[str]
        Canonical gene list from scRNA (e.g. AnnData.var.index).
    other_genes : List[str]
        Gene list from another modality (ST or Bulk).
    min_shared : int, default 2000
        Minimum number of shared genes required. Raises ValueError if not met.

    Returns
    -------
    Tuple[List[str], List[str]]
        (common_genes, other_filtered) where both lists contain only genes
        present in the scRNA canonical list, in their original order.
    """
    canonical = set(scRNA_genes)
    filtered = [g for g in other_genes if g in canonical]
    common = [g for g in scRNA_genes if g in canonical and g in filtered]
    if len(common) < min_shared:
        raise ValueError(
            f"Only {len(common)} shared genes found, minimum {min_shared} required."
        )
    return common, filtered


def harmonize_genes_three_modalities(
    scrna_genes: List[str],
    st_genes: List[str],
    bulk_genes: List[str],
    min_shared: int = 2000,
) -> Tuple[List[str], List[int], List[int], List[int]]:
    """Harmonize genes across three modalities (scRNA, ST, Bulk).

    scRNA gene list is canonical; ST and Bulk are projected onto it.

    Parameters
    ----------
    scrna_genes : List[str]
        Canonical gene list from scRNA.
    st_genes : List[str]
        Gene list from spatial transcriptomics.
    bulk_genes : List[str]
        Gene list from bulk RNA-seq.
    min_shared : int, default 2000
        Minimum number of shared genes required.

    Returns
    -------
    Tuple[List[str], List[int], List[int], List[int]]
        (shared_genes, gene_idx_sc, gene_idx_st, gene_idx_bulk)
        Gene indices are positions in each modality's original gene list.
    """
    # First intersection: scRNA <-> ST
    common_sc_st, _ = harmonize_genes(scrna_genes, st_genes, min_shared=min_shared)

    # Second intersection: (scRNA-ST) <-> Bulk
    scrna_set = set(scrna_genes)
    st_set = set(st_genes)
    bulk_set = set(bulk_genes)

    # Find intersection of all three
    shared_genes = [g for g in scrna_genes if g in st_set and g in bulk_set]

    if len(shared_genes) < min_shared:
        raise ValueError(
            f"Only {len(shared_genes)} shared genes after harmonizing all "
            f"three modalities; minimum {min_shared} required."
        )

    # Build gene index lists for each modality
    gene_idx_sc = [scrna_genes.index(g) for g in shared_genes]
    gene_idx_st = [st_genes.index(g) for g in shared_genes]
    gene_idx_bulk = [bulk_genes.index(g) for g in shared_genes]

    return shared_genes, gene_idx_sc, gene_idx_st, gene_idx_bulk


def symbol_to_id_mapping(
    genes: List[str], annotation: Dict[str, Dict]
) -> Dict[str, str]:
    """Map gene symbols to gene IDs using a GTF annotation dict.

    Parameters
    ----------
    genes : List[str]
        List of gene symbols (or mixed symbols/gene IDs).
    annotation : Dict[str, Dict]
        Output of load_gtf_annotations.

    Returns
    -------
    Dict[str, str]
        Mapping from symbol -> gene_id for genes that have both in the annotation.
    """
    symbol_to_id: Dict[str, str] = {}
    # Build reverse map: symbol -> gene_id (prefer exact symbol)
    symbol_map: Dict[str, str] = {}
    for gene_id, attrs in annotation.items():
        sym = attrs.get("symbol", "")
        if sym:
            symbol_map[sym] = gene_id

    for gene in genes:
        if gene in symbol_map:
            symbol_to_id[gene] = symbol_map[gene]
    return symbol_to_id


def gene_statistics(genes: List[str]) -> Dict[str, int]:
    """Return summary statistics for a gene list.

    Parameters
    ----------
    genes : List[str]
        Gene list to summarize.

    Returns
    -------
    Dict[str, int]
        {total, symbols, ids, unmapped} counts.
        A gene is considered a symbol if it is NOT a gene_id (no dots).
    """
    total = len(genes)
    symbols = sum(1 for g in genes if "." not in g)
    ids = total - symbols
    unmapped = 0  # determined by downstream annotation
    return {"total": total, "symbols": symbols, "ids": ids, "unmapped": unmapped}
