"""Latent UMAP visualization for VZ-01.

Generates UMAP embeddings from Phase 06 checkpoints, colored by modality
(scRNA/ST/Bulk) and optionally by cell type.

Key patterns:
- torch.load(weights_only=False) for checkpoint loading
- TripleModalVAE(n_genes, latent_dim=128) reconstruction per D-02
- vae.eval() + torch.no_grad() for inference
- Concatenate all modalities with modality labels (0=scRNA, 1=ST, 2=Bulk)
- umap.UMAP(random_state=42) for reproducibility
- sc.pl.umap(adata, color='modality') with seaborn colorblind palette
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import torch
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import umap
import seaborn as sns
from torch import Tensor

from tglue.models.vae import TripleModalVAE


class LatentUMAPPlotter:
    """VZ-01: Generate UMAP from Phase 06 checkpoints.

    Consumes Phase 06 checkpoint outputs without modification.
    Extracts latent embeddings (u_sc, u_st, u_bulk) for UMAP projection.
    """

    def __init__(self, checkpoint_dir: str = "results/ablation_full/checkpoints"):
        """Initialize plotter with default checkpoint directory.

        Parameters
        ----------
        checkpoint_dir : str, default "results/ablation_full/checkpoints"
            Directory containing Phase 06 checkpoint files.
        """
        self.checkpoint_dir = Path(checkpoint_dir)

    def load_checkpoint(self, checkpoint_path: str) -> Tuple[Dict[str, Tensor], Dict[str, Any]]:
        """Load model state from .pt checkpoint.

        Parameters
        ----------
        checkpoint_path : str
            Path to .pt checkpoint file.

        Returns
        -------
        Tuple[Dict[str, Tensor], Dict[str, Any]]
            (vae_state_dict, history) from checkpoint.

        Notes
        -----
        Uses torch.load(weights_only=False) per Phase 06 checkpoint structure.
        """
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        state_dict = checkpoint.get('vae_state_dict', {})
        history = checkpoint.get('history', {})
        return state_dict, history

    def reconstruct_vae(self, state_dict: Dict[str, Tensor], n_genes: int, latent_dim: int = 128) -> TripleModalVAE:
        """Reconstruct VAE model from state dict.

        Parameters
        ----------
        state_dict : Dict[str, Tensor]
            VAE state dict from checkpoint.
        n_genes : int
            Number of genes for model initialization.
        latent_dim : int, default 128
            Latent dimension (D-02: latent_dim=128).

        Returns
        -------
        TripleModalVAE
            Reconstructed VAE model with loaded weights.

        Notes
        -----
        TripleModalVAE initialized per D-02 (latent_dim=128).
        """
        vae = TripleModalVAE(n_genes=n_genes, latent_dim=latent_dim)
        vae.load_state_dict(state_dict)
        return vae

    def extract_latent_embeddings(
        self,
        vae: TripleModalVAE,
        batches: List[Dict[str, Any]],
        guidance_data: Optional[Any] = None,
        device: str = 'cpu'
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Run inference to extract u_sc, u_st, u_bulk from VAE forward pass.

        Parameters
        ----------
        vae : TripleModalVAE
            VAE model to run inference.
        batches : List[Dict[str, Any]]
            List of batch dicts with x_sc, x_st, x_bulk tensors.
        guidance_data : Optional[Any], default None
            PyG Data object for guidance graph (optional for inference).
        device : str, default 'cpu'
            Device for inference ('cpu' or 'cuda').

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]
            (latent_matrix, modality_labels, cell_type_labels)
            - latent_matrix: (n_total, latent_dim) concatenated embeddings
            - modality_labels: (n_total,) with values 0=scRNA, 1=ST, 2=Bulk
            - cell_type_labels: (n_total,) or None if not available

        Notes
        -----
        Uses vae.eval() + torch.no_grad() for proper inference mode.
        Concatenates embeddings from all three modalities per batch.
        Cell type labels optional: color by modality if missing.
        """
        vae.eval()
        vae.to(device)

        all_embeddings = []
        all_labels = []  # Modality labels: 0=scRNA, 1=ST, 2=Bulk
        all_cell_types = []

        with torch.no_grad():
            for batch in batches:
                x_sc = batch['x_sc'].to(device)
                x_st = batch['x_st'].to(device)
                x_bulk = batch['x_bulk'].to(device)

                # Forward pass to get latent embeddings
                output = vae(x_sc, x_st, x_bulk, guidance_data)

                # Extract embeddings
                u_sc = output['u_sc'].cpu().numpy()
                u_st = output['u_st'].cpu().numpy()
                u_bulk = output['u_bulk'].cpu().numpy()

                # Append embeddings
                all_embeddings.append(u_sc)
                all_labels.extend([0] * len(u_sc))  # 0 = scRNA
                all_embeddings.append(u_st)
                all_labels.extend([1] * len(u_st))  # 1 = ST
                all_embeddings.append(u_bulk)
                all_labels.extend([2] * len(u_bulk))  # 2 = Bulk

                # Extract cell type labels if available
                if 'cell_type_sc' in batch:
                    all_cell_types.extend(batch['cell_type_sc'])
                if 'cell_type_st' in batch:
                    all_cell_types.extend(batch['cell_type_st'])
                if 'cell_type_bulk' in batch:
                    all_cell_types.extend(batch['cell_type_bulk'])

        # Concatenate all embeddings
        latent_matrix = np.concatenate(all_embeddings, axis=0)
        modality_labels = np.array(all_labels)
        cell_type_labels = np.array(all_cell_types) if all_cell_types else None

        return latent_matrix, modality_labels, cell_type_labels

    def compute_umap(self, latent_matrix: np.ndarray, n_neighbors: int = 15, min_dist: float = 0.5) -> np.ndarray:
        """Compute UMAP embedding from latent matrix.

        Parameters
        ----------
        latent_matrix : np.ndarray
            (n_samples, latent_dim) matrix of latent embeddings.
        n_neighbors : int, default 15
            Number of neighbors for UMAP graph construction.
        min_dist : float, default 0.5
            Minimum distance for UMAP embedding.

        Returns
        -------
        np.ndarray
            (n_samples, 2) UMAP embedding.

        Notes
        -----
        Uses umap.UMAP(random_state=42) for reproducibility.
        """
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=2,
            random_state=42,
        )
        umap_embedding = reducer.fit_transform(latent_matrix)
        return umap_embedding

    def plot_umap(
        self,
        umap_embedding: np.ndarray,
        modality_labels: np.ndarray,
        cell_type_labels: Optional[np.ndarray] = None,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 8)
    ) -> ad.AnnData:
        """Generate UMAP scatter plot colored by modality/cell type.

        Parameters
        ----------
        umap_embedding : np.ndarray
            (n_samples, 2) UMAP coordinates.
        modality_labels : np.ndarray
            (n_samples,) modality labels (0=scRNA, 1=ST, 2=Bulk).
        cell_type_labels : Optional[np.ndarray], default None
            (n_samples,) cell type labels (optional).
        save_path : Optional[str], default None
            Path to save PDF (scanpy will add .pdf suffix).
        figsize : Tuple[int, int], default (10, 8)
            Figure size in inches.

        Returns
        -------
        ad.AnnData
            AnnData object with UMAP embedding and obs labels.

        Notes
        -----
        Uses sc.pl.umap(adata, color='modality', palette='colorblind').
        Publication quality settings applied via set_publication_style().
        """
        from .publication_quality import set_publication_style
        set_publication_style()

        # Create AnnData for scanpy plotting
        adata = ad.AnnData(X=umap_embedding)
        adata.obsm['X_umap'] = umap_embedding
        adata.obs['modality'] = pd.Categorical(
            ['scRNA' if m == 0 else 'ST' if m == 1 else 'Bulk' for m in modality_labels]
        )

        if cell_type_labels is not None:
            adata.obs['cell_type'] = pd.Categorical(cell_type_labels)

        # Publication quality settings (already applied by set_publication_style)
        sc.set_figure_params(figsize=figsize, fontsize=12, dpi=300)

        # Use seaborn colorblind palette
        colorblind_palette = sns.color_palette('colorblind').as_hex()

        # Plot by modality
        if save_path:
            # scanpy saves to 'figures/' relative to current working directory
            # It adds 'umap' prefix to the filename
            # We use show=False to prevent display and ensure saving
            save_basename = Path(save_path).stem.replace('umap_', '')
            sc.pl.umap(adata, color='modality', save=save_basename, palette=colorblind_palette, frameon=True, show=False)

        # Plot by cell type if available
        if cell_type_labels is not None and save_path:
            celltype_basename = Path(save_path).stem.replace('umap_', '') + '_celltype'
            sc.pl.umap(adata, color='cell_type', save=celltype_basename, palette=colorblind_palette, frameon=True, show=False)

        return adata