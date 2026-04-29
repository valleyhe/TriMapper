"""Loss curve visualization for VZ-03.

Generates training loss curves with pre-warm/adversarial phase annotations
from Phase 06 checkpoint history.

Key patterns:
- torch.load(weights_only=False) for checkpoint loading
- checkpoint.get('history', {}) with default empty dict
- LOSS_KEYS validation for all 4 components
- ax.axvline(x=20) for phase boundary (D-03)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List

import torch
import numpy as np
import matplotlib.pyplot as plt


class LossCurvePlotter:
    """VZ-03: Generate training loss curves from history.

    Consumes Phase 06 checkpoint history dict without modification.
    Annotates pre-warm (0-20) vs adversarial (20+) phases per D-03.
    """

    LOSS_KEYS = ['vae_loss', 'recon_loss', 'kl_loss', 'disc_loss']

    def __init__(self, checkpoint_dir: str = "results/ablation_full/checkpoints"):
        """Initialize plotter with default checkpoint directory.

        Parameters
        ----------
        checkpoint_dir : str, default "results/ablation_full/checkpoints"
            Directory containing Phase 06 checkpoint files.
        """
        self.checkpoint_dir = Path(checkpoint_dir)

    def load_history_from_checkpoint(self, checkpoint_path: str) -> Dict[str, List[float]]:
        """Load training history from checkpoint.

        Parameters
        ----------
        checkpoint_path : str
            Path to .pt checkpoint file.

        Returns
        -------
        Dict[str, List[float]]
            History dict with keys: 'vae_loss', 'recon_loss', 'kl_loss', 'disc_loss'.
            Missing keys are filled with empty lists.

        Notes
        -----
        Uses torch.load(weights_only=False) per Pitfall 1.
        Validated history structure ensures all LOSS_KEYS present.
        """
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        history = checkpoint.get('history', {})

        # Validate history structure - fill missing keys with empty lists
        for key in self.LOSS_KEYS:
            if key not in history:
                history[key] = []

        return history

    def load_all_histories(self, results_dir: str = "results") -> Dict[str, Dict[str, List[float]]]:
        """Load history from all 5 ablation experiments.

        Parameters
        ----------
        results_dir : str, default "results"
            Root directory containing all ablation experiment subdirectories.

        Returns
        -------
        Dict[str, Dict[str, List[float]]]
            Dict mapping experiment names to history dicts.
            Missing experiments get placeholder empty histories.

        Notes
        -----
        Experiments: ablation_full, ablation_no_guidance_graph, ablation_no_fusion_conv,
                     ablation_no_bulk_prior, ablation_no_ot_deconv.
        """
        results_dir = Path(results_dir)
        all_histories = {}

        experiments = [
            'ablation_full',
            'ablation_no_guidance_graph',
            'ablation_no_fusion_conv',
            'ablation_no_bulk_prior',
            'ablation_no_ot_deconv',
        ]

        for exp_name in experiments:
            checkpoint_path = results_dir / exp_name / "checkpoints" / "final.pt"
            if checkpoint_path.exists():
                all_histories[exp_name] = self.load_history_from_checkpoint(str(checkpoint_path))
            else:
                # Placeholder empty history for missing experiments
                all_histories[exp_name] = {k: [] for k in self.LOSS_KEYS}

        return all_histories

    def plot_multi_line_loss(
        self,
        history: Dict[str, List[float]],
        save_path: str | None = None,
        figsize: tuple = (12, 8),
        pre_warm_epochs: int = 20
    ) -> plt.Figure:
        """Plot multi-line loss curves with phase annotations.

        Parameters
        ----------
        history : Dict[str, List[float]]
            History dict with loss components (vae_loss, recon_loss, kl_loss, disc_loss).
        save_path : str | None, optional
            Path to save figure (both .pdf and .png will be saved).
        figsize : tuple, default (12, 8)
            Figure size in inches.
        pre_warm_epochs : int, default 20
            Epoch boundary for pre-warm vs adversarial phase (D-03).

        Returns
        -------
        plt.Figure
            Matplotlib figure object with 2x2 subplot grid.

        Notes
        -----
        Creates 2x2 subplot grid for 4 loss components.
        Each subplot shows:
        - Loss curve over epochs
        - ax.axvline(x=20) for phase boundary (D-03)
        - fill_between for phase regions (blue pre-warm, orange adversarial)
        - Grid with alpha=0.3 for readability
        """
        from .publication_quality import set_publication_style
        set_publication_style()

        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # Loss components to plot
        loss_labels = [
            ('vae_loss', 'Total VAE Loss'),
            ('recon_loss', 'Reconstruction Loss'),
            ('kl_loss', 'KL Divergence'),
            ('disc_loss', 'Discriminator Loss'),
        ]

        for ax, (key, title) in zip(axes.flat, loss_labels):
            if key in history and len(history[key]) > 0:
                epochs = np.arange(len(history[key]))
                values = np.array(history[key])

                # Plot loss curve
                ax.plot(epochs, values, linewidth=2, color='#1f77b4')  # Standard blue

                # Annotate pre-warm vs adversarial phase (D-03)
                ax.axvline(
                    x=pre_warm_epochs,
                    color='red',
                    linestyle='--',
                    linewidth=2,
                    label='Pre-warm end'
                )

                # Fill regions with phase colors
                ylim = ax.get_ylim()
                ax.fill_betweenx(
                    ylim,
                    0,
                    pre_warm_epochs,
                    alpha=0.1,
                    color='blue',
                    label='Pre-warm'
                )

                max_epoch = max(epochs) if len(epochs) > 0 else pre_warm_epochs + 1
                ax.fill_betweenx(
                    ylim,
                    pre_warm_epochs,
                    max_epoch,
                    alpha=0.1,
                    color='orange',
                    label='Adversarial'
                )

                # Labels and formatting
                ax.set_xlabel('Epoch', fontsize=12)
                ax.set_ylabel(title, fontsize=12)
                ax.set_title(title, fontsize=14)
                ax.legend(loc='upper right', fontsize=10)
                ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save figure if path provided
        if save_path:
            save_path = Path(save_path)
            plt.savefig(save_path.with_suffix('.pdf'), dpi=300, bbox_inches='tight')
            plt.savefig(save_path.with_suffix('.png'), dpi=300, bbox_inches='tight')

        return fig

    def plot_all_experiments_comparison(
        self,
        all_histories: Dict[str, Dict[str, List[float]]],
        save_path: str | None = None,
        figsize: tuple = (12, 6),
        pre_warm_epochs: int = 20
    ) -> plt.Figure:
        """Compare VAE loss across all 5 experiments.

        Parameters
        ----------
        all_histories : Dict[str, Dict[str, List[float]]]
            Dict mapping experiment names to history dicts.
        save_path : str | None, optional
            Path to save figure (both .pdf and .png will be saved).
        figsize : tuple, default (12, 6)
            Figure size in inches.
        pre_warm_epochs : int, default 20
            Epoch boundary for pre-warm vs adversarial phase (D-03).

        Returns
        -------
        plt.Figure
            Matplotlib figure object with single axes.

        Notes
        -----
        Shows all 5 VAE loss curves on single axes.
        Uses colorblind palette for accessibility.
        Annotates pre-warm boundary with ax.axvline(x=20).
        """
        from .publication_quality import set_publication_style, get_colorblind_palette
        set_publication_style()

        palette = get_colorblind_palette()

        fig, ax = plt.subplots(figsize=figsize)

        # Experiment labels for legend
        experiments = [
            ('ablation_full', 'Full Model'),
            ('ablation_no_guidance_graph', 'No GuidanceGraph'),
            ('ablation_no_fusion_conv', 'No fusion_conv'),
            ('ablation_no_bulk_prior', 'No BulkPrior'),
            ('ablation_no_ot_deconv', 'No OT Deconv'),
        ]

        # Plot each experiment
        for i, (exp_name, label) in enumerate(experiments):
            if exp_name in all_histories and 'vae_loss' in all_histories[exp_name]:
                history = all_histories[exp_name]
                if len(history['vae_loss']) > 0:
                    epochs = np.arange(len(history['vae_loss']))
                    values = np.array(history['vae_loss'])

                    ax.plot(
                        epochs,
                        values,
                        linewidth=2,
                        color=palette[i % len(palette)],
                        label=label
                    )

        # Annotate pre-warm boundary (D-03)
        ax.axvline(
            x=pre_warm_epochs,
            color='gray',
            linestyle='--',
            linewidth=1,
            label='Pre-warm end'
        )

        # Labels and formatting
        ax.set_xlabel('Epoch', fontsize=14)
        ax.set_ylabel('VAE Loss', fontsize=14)
        ax.set_title('Training Loss Comparison Across Ablations', fontsize=14)
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save figure if path provided
        if save_path:
            save_path = Path(save_path)
            plt.savefig(save_path.with_suffix('.pdf'), dpi=300, bbox_inches='tight')
            plt.savefig(save_path.with_suffix('.png'), dpi=300, bbox_inches='tight')

        return fig