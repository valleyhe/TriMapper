"""DualModalVAESequential: Memory-efficient dual-modal VAE with sequential processing.

Per 14-03: Process each modality independently to avoid simultaneous memory accumulation.
Reduces peak GPU memory from ~18GB to <1GB on 22GB GPU.

CRITICAL: Uses MSE loss instead of NB reconstruction to avoid torch.lgamma memory issues.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from .encoders import NBDataEncoder
from .utils import kl_gaussian, mse_reconstruction_loss


class DualModalVAESequential(nn.Module):
    """Memory-efficient dual-modal VAE with sequential modality processing.

    Key innovations:
    1. Process scRNA and ST in separate forward/backward passes
    2. Use MSE loss instead of NB reconstruction (avoids torch.lgamma memory issues)
    3. Lightweight architecture: hidden_dim=128, latent_dim=64

    Architecture changes for memory efficiency:
    - hidden_dim: 128 (reduced from 256)
    - latent_dim: 64 (reduced from 128)
    - NO graph encoder/decoder (disabled for memory)
    - MSE reconstruction instead of NB (avoids lgamma memory explosion)
    """

    def __init__(
        self,
        n_genes: int,
        latent_dim: int = 64,  # Reduced from 128
        hidden_dim: int = 128,  # Reduced from 256
        use_checkpointing: bool = True,  # Default enabled
    ) -> None:
        super().__init__()
        self.n_genes = n_genes
        self.latent_dim = latent_dim

        # Lightweight modality encoders
        self.enc_sc = NBDataEncoder(n_genes, latent_dim, hidden_dim, use_checkpointing=use_checkpointing)
        self.enc_st = NBDataEncoder(n_genes, latent_dim, hidden_dim, use_checkpointing=use_checkpointing)

        # Gene projection layers for reconstruction
        self.gene_proj_sc = nn.Linear(latent_dim, n_genes, bias=False)
        self.gene_proj_st = nn.Linear(latent_dim, n_genes, bias=False)

    @property
    def gene_count(self) -> int:
        return self.n_genes

    def compute_kl(self, mean: Tensor, log_var: Tensor) -> Tensor:
        return kl_gaussian(mean, log_var)

    def forward_sc_only(self, x_sc: Tensor) -> dict[str, Tensor]:
        """Forward pass for scRNA modality only.

        Process scRNA batch independently to reduce memory.
        Returns loss components for gradient accumulation.

        Args:
            x_sc: (batch_sc, n_genes) scRNA counts

        Returns:
            dict with z_sc, recon_loss_sc, kl_loss_sc
        """
        z_sc, mean_sc, log_var_sc = self.enc_sc(x_sc)
        kl_sc = self.compute_kl(mean_sc, log_var_sc)

        # MSE reconstruction (memory-efficient, no lgamma)
        pred_sc = torch.relu(self.gene_proj_sc(mean_sc))
        recon_sc = mse_reconstruction_loss(x_sc, pred_sc)

        return {
            "z_sc": z_sc,
            "recon_loss_sc": recon_sc.mean(),
            "kl_loss_sc": kl_sc.mean(),
        }

    def forward_st_only(self, x_st: Tensor) -> dict[str, Tensor]:
        """Forward pass for ST modality only.

        Process ST batch independently to reduce memory.
        Returns loss components for gradient accumulation.

        Args:
            x_st: (batch_st, n_genes) ST counts

        Returns:
            dict with z_st, recon_loss_st, kl_loss_st
        """
        z_st, mean_st, log_var_st = self.enc_st(x_st)
        kl_st = self.compute_kl(mean_st, log_var_st)

        # MSE reconstruction (memory-efficient, no lgamma)
        pred_st = torch.relu(self.gene_proj_st(mean_st))
        recon_st = mse_reconstruction_loss(x_st, pred_st)

        return {
            "z_st": z_st,
            "recon_loss_st": recon_st.mean(),
            "kl_loss_st": kl_st.mean(),
        }

    def forward_combined(
        self,
        x_sc: Tensor,
        x_st: Tensor,
    ) -> dict[str, Tensor]:
        """Combined forward pass (original DualModalVAE behavior).

        WARNING: This method processes both modalities simultaneously
        and may cause OOM on limited GPU memory.
        Use forward_sc_only + forward_st_only for memory-efficient training.
        """
        # Process scRNA
        sc_out = self.forward_sc_only(x_sc)
        # Process ST
        st_out = self.forward_st_only(x_st)

        # Combine outputs
        return {
            "u_sc": sc_out["z_sc"],
            "u_st": st_out["z_st"],
            "recon_loss": sc_out["recon_loss_sc"] + st_out["recon_loss_st"],
            "kl_loss": sc_out["kl_loss_sc"] + st_out["kl_loss_st"],
            "graph_recon_loss": torch.tensor(0.0, device=x_sc.device),
        }