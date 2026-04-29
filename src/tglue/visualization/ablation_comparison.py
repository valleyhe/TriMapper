"""Ablation comparison visualization (VZ-02).

This module implements grouped bar charts comparing 5 ablation experiments
across 6 metrics (ASW, NMI, GC, ARI, Pearson, KL) to quantify each
component's contribution to triple-modal integration.

Experiments (from Phase 06):
- ablation_full (full model)
- ablation_no_guidance_graph (AB-01)
- ablation_no_fusion_conv (AB-02)
- ablation_no_bulk_prior (AB-03)
- ablation_no_ot_deconv (AB-04)

Metrics structure (from Phase 06-07-SUMMARY.md):
{
    "config": {name, git_hash, seed, timestamp},
    "metrics": {
        "alignment": {"ASW": float, "NMI": float, "GC": float},
        "deconvolution": {"Pearson": float, "KL": float},
        "spatial": {"ARI": float}
    }
}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class AblationComparisonPlotter:
    """VZ-02: Generate ablation comparison bar charts.

    This class loads metrics from Phase 06 ablation experiments and creates
    grouped bar charts showing each component's contribution to the model.

    Parameters
    ----------
    results_dir : str, default "results"
        Directory containing ablation experiment results.

    Attributes
    ----------
    METRICS : list[str]
        List of 6 metrics to compare.
    EXPERIMENTS : list[str]
        List of 5 ablation experiment names.

    Examples
    --------
    >>> plotter = AblationComparisonPlotter("results")
    >>> all_metrics = plotter.load_all_metrics()
    >>> df_long = plotter.create_comparison_dataframe(all_metrics)
    >>> plotter.plot_grouped_bars(df_long, "figures/ablation_comparison")
    """

    METRICS = ['ASW', 'NMI', 'GC', 'ARI', 'Pearson', 'KL']
    EXPERIMENTS = [
        'ablation_full',
        'ablation_no_guidance_graph',
        'ablation_no_fusion_conv',
        'ablation_no_bulk_prior',
        'ablation_no_ot_deconv',
    ]

    def __init__(self, results_dir: str = "results"):
        """Initialize plotter with results directory."""
        self.results_dir = Path(results_dir)

    def load_all_metrics(self) -> Dict[str, Dict[str, float]]:
        """Load metrics.json from all 5 ablation experiments.

        Returns
        -------
        Dict[str, Dict[str, float]]
            Dictionary mapping experiment name to metrics dict.
            Each metrics dict contains: ASW, NMI, GC, ARI, Pearson, KL.
            Missing experiments return placeholder 0.0 values.

        Notes
        -----
        Metrics structure from Phase 06-07-SUMMARY.md:
        - metrics.json contains nested structure with alignment, deconvolution, spatial
        - This method extracts all 6 metrics into a flat dict per experiment
        - Missing files return 0.0 placeholders (Open Question 3 in plan)

        Examples
        --------
        >>> plotter = AblationComparisonPlotter("results")
        >>> metrics = plotter.load_all_metrics()
        >>> len(metrics)  # 5 experiments
        5
        >>> metrics['ablation_full'].keys()  # 6 metrics
        ['ASW', 'NMI', 'GC', 'ARI', 'Pearson', 'KL']
        """
        all_metrics = {}

        for exp_name in self.EXPERIMENTS:
            metrics_path = self.results_dir / exp_name / "metrics.json"

            if metrics_path.exists():
                with open(metrics_path) as f:
                    data = json.load(f)
                    # Extract nested metrics structure
                    metrics = data.get('metrics', {})
                    alignment = metrics.get('alignment', {})
                    deconvolution = metrics.get('deconvolution', {})
                    spatial = metrics.get('spatial', {})

                    all_metrics[exp_name] = {
                        'ASW': alignment.get('ASW', 0.0),
                        'NMI': alignment.get('NMI', 0.0),
                        'GC': alignment.get('GC', 0.0),
                        'ARI': spatial.get('ARI', 0.0),
                        'Pearson': deconvolution.get('Pearson', 0.0),
                        'KL': deconvolution.get('KL', 0.0),
                    }
            else:
                # Placeholder for missing experiments (Open Question 3)
                all_metrics[exp_name] = {m: 0.0 for m in self.METRICS}

        return all_metrics

    def create_comparison_dataframe(self, all_metrics: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """Reshape metrics to long format for seaborn grouped bars.

        Parameters
        ----------
        all_metrics : Dict[str, Dict[str, float]]
            Dictionary from load_all_metrics() containing experiment metrics.

        Returns
        -------
        pd.DataFrame
            Long-format DataFrame with columns: experiment, metric, value.
            Suitable for seaborn.barplot(x='experiment', y='value', hue='metric').

        Notes
        -----
        Uses pd.melt() pattern from RESEARCH.md Pattern 2:
        - Wide format: rows=experiments, cols=metrics
        - Long format: each row is (experiment, metric, value) tuple

        Examples
        --------
        >>> plotter = AblationComparisonPlotter("results")
        >>> metrics = plotter.load_all_metrics()
        >>> df = plotter.create_comparison_dataframe(metrics)
        >>> df.columns.tolist()
        ['experiment', 'metric', 'value']
        >>> len(df)  # 5 experiments * 6 metrics = 30 rows
        30
        """
        # Create wide-format DataFrame (rows=experiments, cols=metrics)
        df_wide = pd.DataFrame(all_metrics).T
        df_wide.index.name = 'experiment'
        df_wide.reset_index(inplace=True)

        # Melt to long format: (experiment, metric, value)
        df_long = df_wide.melt(
            id_vars='experiment',
            value_vars=self.METRICS,
            var_name='metric',
            value_name='value'
        )

        return df_long

    def compute_delta_from_full(self, all_metrics: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """Compute delta (contribution) from full model baseline.

        Parameters
        ----------
        all_metrics : Dict[str, Dict[str, float]]
            Dictionary from load_all_metrics() containing experiment metrics.

        Returns
        -------
        Dict[str, Dict[str, float]]
            Dictionary mapping ablation experiment name to delta dict.
            Delta = full_metric - ablation_metric (positive = component helped).
            Does NOT include 'ablation_full' key.

        Notes
        -----
        Delta interpretation:
        - Positive delta: removing component hurt performance (component helped)
        - Negative delta: removing component improved performance (component hurt)
        - Zero delta: component had no effect

        Examples
        --------
        >>> plotter = AblationComparisonPlotter("results")
        >>> metrics = plotter.load_all_metrics()
        >>> delta = plotter.compute_delta_from_full(metrics)
        >>> 'ablation_full' in delta  # Baseline excluded
        False
        >>> len(delta)  # 4 ablation experiments
        4
        """
        full_metrics = all_metrics['ablation_full']

        delta_df = {}
        for exp_name, metrics in all_metrics.items():
            if exp_name == 'ablation_full':
                continue  # Skip baseline

            delta_df[exp_name] = {
                m: full_metrics[m] - metrics[m]  # Positive = component helped
                for m in self.METRICS
            }

        return delta_df

    def plot_grouped_bars(
        self,
        df_long: pd.DataFrame,
        save_path: Optional[str] = None,
        figsize: tuple = (12, 6)
    ) -> plt.Figure:
        """Generate grouped bar chart with seaborn.

        Parameters
        ----------
        df_long : pd.DataFrame
            Long-format DataFrame from create_comparison_dataframe().
        save_path : str, optional
            Base path for saving figures. Will save both .pdf and .png.
        figsize : tuple, default (12, 6)
            Figure size in inches.

        Returns
        -------
        plt.Figure
            Matplotlib figure object.

        Notes
        -----
        Uses seaborn.barplot with hue='metric' and palette='colorblind'.
        Saves both PDF (vector) and PNG (raster) at 300 DPI.

        Examples
        --------
        >>> plotter = AblationComparisonPlotter("results")
        >>> metrics = plotter.load_all_metrics()
        >>> df = plotter.create_comparison_dataframe(metrics)
        >>> fig = plotter.plot_grouped_bars(df, "figures/ablation_comparison")
        """
        # Import here to avoid dependency if not used
        try:
            from .publication_quality import set_publication_style
            set_publication_style()
        except ImportError:
            # Fallback if publication_quality module not available
            plt.rcParams['font.size'] = 12
            plt.rcParams['figure.dpi'] = 300

        fig, ax = plt.subplots(figsize=figsize)

        # Grouped bar plot: x=experiment, y=value, hue=metric
        sns.barplot(
            data=df_long,
            x='experiment',
            y='value',
            hue='metric',
            palette='colorblind',
            ax=ax,
            dodge=True,
        )

        # Customize
        ax.set_xlabel('Ablation Experiment', fontsize=14)
        ax.set_ylabel('Metric Value', fontsize=14)
        ax.set_title('Component Contribution to Triple-Modal Integration', fontsize=14)
        ax.legend(title='Metric', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.grid(True, axis='y', alpha=0.3)

        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            plt.savefig(save_path.with_suffix('.pdf'), dpi=300, bbox_inches='tight')
            plt.savefig(save_path.with_suffix('.png'), dpi=300, bbox_inches='tight')

        return fig

    def plot_delta_bars(
        self,
        delta_df: Dict[str, Dict[str, float]],
        save_path: Optional[str] = None,
        figsize: tuple = (10, 6)
    ) -> plt.Figure:
        """Generate delta bar chart showing component contribution.

        Parameters
        ----------
        delta_df : Dict[str, Dict[str, float]]
            Dictionary from compute_delta_from_full().
        save_path : str, optional
            Base path for saving figures. Will save both .pdf and .png.
        figsize : tuple, default (10, 6)
            Figure size in inches.

        Returns
        -------
        plt.Figure
            Matplotlib figure object.

        Notes
        -----
        Shows delta from full model with axhline(y=0) baseline.
        Positive values indicate component contribution.
        Saves both PDF (vector) and PNG (raster) at 300 DPI.

        Examples
        --------
        >>> plotter = AblationComparisonPlotter("results")
        >>> metrics = plotter.load_all_metrics()
        >>> delta = plotter.compute_delta_from_full(metrics)
        >>> fig = plotter.plot_delta_bars(delta, "figures/ablation_delta")
        """
        # Import here to avoid dependency if not used
        try:
            from .publication_quality import set_publication_style
            set_publication_style()
        except ImportError:
            # Fallback if publication_quality module not available
            plt.rcParams['font.size'] = 12
            plt.rcParams['figure.dpi'] = 300

        df_wide = pd.DataFrame(delta_df).T
        df_long = df_wide.reset_index().melt(
            id_vars='index',
            var_name='metric',
            value_name='delta'
        )
        df_long = df_long.rename(columns={'index': 'experiment'})

        fig, ax = plt.subplots(figsize=figsize)

        sns.barplot(
            data=df_long,
            x='experiment',
            y='delta',
            hue='metric',
            palette='colorblind',
            ax=ax,
        )

        ax.set_xlabel('Component Removed', fontsize=14)
        ax.set_ylabel('Delta from Full Model', fontsize=14)
        ax.set_title('Component Contribution Analysis', fontsize=14)
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax.legend(title='Metric', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.grid(True, axis='y', alpha=0.3)

        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            plt.savefig(save_path.with_suffix('.pdf'), dpi=300, bbox_inches='tight')
            plt.savefig(save_path.with_suffix('.png'), dpi=300, bbox_inches='tight')

        return fig