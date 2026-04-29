"""Evaluation metrics module.

Includes:
- compute_asw: Average Silhouette Width for modality alignment
- compute_nmi: Normalized Mutual Information for clustering quality
- compute_gc: Graph Connectivity for guidance graph evaluation
- compute_spatial_ari: Adjusted Rand Index for spatial domain identification
- compute_spatial_nmi: NMI for spatial domain identification
- evaluate_alignment: Combined alignment evaluation (ASW + NMI + GC)
- evaluate_spatial_domains: Spatial domain clustering evaluation
- evaluate_leiden_clustering: Leiden-based spatial evaluation
- log_metrics: TensorBoard/logging helper
"""

from __future__ import annotations

from .metrics import (
    compute_asw,
    compute_nmi,
    compute_spatial_ari,
    compute_spatial_nmi,
    compute_gc,
    evaluate_alignment,
    evaluate_spatial_domains,
    log_metrics,
)
from .leiden_metrics import evaluate_leiden_clustering

__all__ = [
    "compute_asw",
    "compute_nmi",
    "compute_spatial_ari",
    "compute_spatial_nmi",
    "compute_gc",
    "evaluate_alignment",
    "evaluate_spatial_domains",
    "evaluate_leiden_clustering",
    "log_metrics",
]