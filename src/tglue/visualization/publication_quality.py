"""Publication-quality matplotlib/seaborn/scanpy visualization configuration.

This module provides centralized rcParams settings for publication-quality
figures across all visualization components.

VZ-01 requirement: font.size >= 12, dpi=300, colorblind-friendly palette.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns


def set_publication_style():
    """Configure matplotlib for publication-quality figures.

    Sets rcParams following VZ-01 requirements:
    - font.size >= 12 (readability)
    - figure.dpi = 300 (high resolution)
    - savefig.dpi = 300 (export quality)
    - colorblind-friendly palette

    Also configures scanpy defaults for consistent visualization.
    """
    plt.rcParams.update({
        # Font settings (VZ-01: font.size >= 12)
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Helvetica'],
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 10,

        # Figure settings (VZ-01: dpi=300)
        'figure.figsize': (6, 4),
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',

        # Axis and line settings
        'axes.linewidth': 1.5,
        'lines.linewidth': 2,
        'lines.markersize': 8,

        # Grid settings
        'axes.grid': True,
        'grid.alpha': 0.3,
    })

    # Configure scanpy defaults (D-02: latent_dim=128 ensures consistent embedding size)
    import scanpy as sc
    sc.set_figure_params(
        figsize=(6, 4),
        fontsize=12,
        dpi=300,
        dpi_save=300,
        frameon=True
    )


def get_colorblind_palette():
    """Return colorblind-friendly palette for seaborn.

    VZ-01 requirement: accessible color palette for publication figures.

    Returns:
        list: 10-color seaborn colorblind palette.

    Usage:
        palette = get_colorblind_palette()
        sns.barplot(..., palette=palette)
    """
    return sns.color_palette('colorblind')