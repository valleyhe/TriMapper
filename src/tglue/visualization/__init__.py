"""Visualization module for Phase 07 (VZ-01, VZ-02, VZ-03).

This module provides publication-quality visualization components for
triple-modal integration ablation experiments:

- LossCurvePlotter: Training loss curves with phase annotations (VZ-03)
- AblationComparisonPlotter: Grouped bar charts (VZ-02)
- LatentUMAPPlotter: UMAP embeddings from checkpoint (VZ-01)

All components consume Phase 06 outputs without creating new data.
"""

from __future__ import annotations

from .publication_quality import set_publication_style, get_colorblind_palette
from .loss_curves import LossCurvePlotter
from .ablation_comparison import AblationComparisonPlotter
from .latent_umap import LatentUMAPPlotter

# Apply publication style at module import time (VZ-01 requirement)
set_publication_style()

__all__ = [
    'set_publication_style',
    'get_colorblind_palette',
    'LossCurvePlotter',
    'AblationComparisonPlotter',
    'LatentUMAPPlotter',
]