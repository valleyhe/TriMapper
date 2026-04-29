"""TripleModalVAEEfficient: Memory-efficient triple-modal VAE with sequential processing.

Per 15-FA-01: Process each modality independently to avoid simultaneous memory accumulation.
Uses MSE loss instead of NB reconstruction to avoid torch.lgamma memory issues.

Fair ablation experiment design:
- Unified architecture: latent_dim=64, hidden_dim=128
- Unified loss: MSE reconstruction
- Sequential modality processing
- Includes Bulk encoder for triple-modal comparison
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from .encoders import NBDataEncoder, VanillaDataEncoder
from .utils import kl_gaussian, mse_reconstruction_loss


class TripleModalVAEEfficient(nn.Module):
    """Memory-efficient triple-modal VAE with sequential modality processing.

    Key innovations:
    1. Process scRNA, ST, Bulk in separate forward/backward passes
    2. Use MSE loss instead of NB reconstruction (avoids torch.lgamma memory issues)
    3. Lightweight architecture: hidden_dim=128, latent_dim=64
    4. Bulk encoder for composition prior (no reconstruction target)

    Architecture for fair ablation comparison:
    - latent_dim: 64 (unified across all FA configs)
    - hidden_dim: 128 (unified across all FA configs)
    - MSE reconstruction (unified across all FA configs)
    """

    def __init__(
        self,
        n_genes: int,
        latent_dim: int = 64,
        hidden_dim: int = 128,
        use_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        self.n_genes = n_genes
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        # Modality encoders (NB for count data, Vanilla for bulk)
        self.enc_sc = NBDataEncoder(n_genes, latent_dim, hidden_dim, use_checkpointing=use_checkpointing)
        self.enc_st = NBDataEncoder(n_genes, latent_dim, hidden_dim, use_checkpointing=use_checkpointing)
        self.enc_bulk = VanillaDataEncoder(n_genes, latent_dim, hidden_dim)

        # Gene projection layers for reconstruction (scRNA and ST only)
        self.gene_proj_sc = nn.Linear(latent_dim, n_genes, bias=False)
        self.gene_proj_st = nn.Linear(latent_dim, n_genes, bias=False)

    @property
    def gene_count(self) -> int:
        return self.n_genes

    def compute_kl(self, mean: Tensor, log_var: Tensor) -> Tensor:
        return kl_gaussian(mean, log_var)

    def forward_sc_only(self, x_sc: Tensor) -> dict[str, Tensor]:
        """Forward pass for scRNA modality only.

        Args:
            x_sc: (batch_sc, n_genes) scRNA counts

        Returns:
            dict with z_sc, recon_loss_sc, kl_loss_sc
        """
        z_sc, mean_sc, log_var_sc = self.enc_sc(x_sc)
        kl_sc = self.compute_kl(mean_sc, log_var_sc)

        # MSE reconstruction (memory-efficient)
        pred_sc = torch.relu(self.gene_proj_sc(mean_sc))
        recon_sc = mse_reconstruction_loss(x_sc, pred_sc)

        return {
            "z_sc": z_sc,
            "recon_loss_sc": recon_sc.mean(),
            "kl_loss_sc": kl_sc.mean(),
        }

    def forward_st_only(self, x_st: Tensor) -> dict[str, Tensor]:
        """Forward pass for ST modality only.

        Args:
            x_st: (batch_st, n_genes) ST counts

        Returns:
            dict with z_st, recon_loss_st, kl_loss_st
        """
        z_st, mean_st, log_var_st = self.enc_st(x_st)
        kl_st = self.compute_kl(mean_st, log_var_st)

        # MSE reconstruction (memory-efficient)
        pred_st = torch.relu(self.gene_proj_st(mean_st))
        recon_st = mse_reconstruction_loss(x_st, pred_st)

        return {
            "z_st": z_st,
            "recon_loss_st": recon_st.mean(),
            "kl_loss_st": kl_st.mean(),
        }

    def forward_bulk_only(self, x_bulk: Tensor) -> dict[str, Tensor]:
        """Forward pass for Bulk modality only.

        Args:
            x_bulk: (batch_bulk, n_genes) Bulk expression

        Returns:
            dict with z_bulk (no reconstruction for Bulk, but L2 regularization)
        """
        z_bulk = self.enc_bulk(x_bulk)

        # L2 regularization on z_bulk (ensures gradient flow for Bulk encoder)
        # This is a surrogate loss since Bulk has no reconstruction target
        kl_loss_bulk = z_bulk.pow(2).mean() * 0.01  # Small coefficient to not dominate

        return {
            "z_bulk": z_bulk,
            "recon_loss_bulk": torch.tensor(0.0, device=x_bulk.device),
            "kl_loss_bulk": kl_loss_bulk,
        }

    def forward_combined(
        self,
        x_sc: Tensor,
        x_st: Tensor,
        x_bulk: Tensor,
    ) -> dict[str, Tensor]:
        """Combined forward pass (original TripleModalVAE behavior).

        WARNING: This method processes all modalities simultaneously
        and may cause OOM on limited GPU memory.
        Use forward_sc_only + forward_st_only + forward_bulk_only for memory-efficient training.

        VAE-05 FIX: Handles empty tensors gracefully, returning zero losses.
        """
        # VAE-05 FIX: Entry validation for empty inputs
        device = x_sc.device if x_sc.numel() > 0 else (x_st.device if x_st.numel() > 0 else x_bulk.device)

        # Handle empty scRNA tensor
        if x_sc.shape[0] == 0:
            # Return empty embeddings with zero losses
            empty_sc = torch.empty(0, self.latent_dim, device=device)
            st_out = self.forward_st_only(x_st) if x_st.shape[0] > 0 else {
                "z_st": torch.empty(0, self.latent_dim, device=device),
                "recon_loss_st": torch.tensor(0.0, device=device),
                "kl_loss_st": torch.tensor(0.0, device=device),
            }
            bulk_out = self.forward_bulk_only(x_bulk) if x_bulk.shape[0] > 0 else {
                "z_bulk": torch.empty(0, self.latent_dim, device=device),
                "recon_loss_bulk": torch.tensor(0.0, device=device),
                "kl_loss_bulk": torch.tensor(0.0, device=device),
            }
            return {
                "u_sc": empty_sc,
                "u_st": st_out["z_st"],
                "u_bulk": bulk_out["z_bulk"],
                "recon_loss": torch.tensor(0.0, device=device),
                "kl_loss": torch.tensor(0.0, device=device),
            }

        # Process scRNA
        sc_out = self.forward_sc_only(x_sc)
        # Process ST
        st_out = self.forward_st_only(x_st) if x_st.shape[0] > 0 else {
            "z_st": torch.empty(0, self.latent_dim, device=device),
            "recon_loss_st": torch.tensor(0.0, device=device),
            "kl_loss_st": torch.tensor(0.0, device=device),
        }
        # Process Bulk
        bulk_out = self.forward_bulk_only(x_bulk) if x_bulk.shape[0] > 0 else {
            "z_bulk": torch.empty(0, self.latent_dim, device=device),
            "recon_loss_bulk": torch.tensor(0.0, device=device),
            "kl_loss_bulk": torch.tensor(0.0, device=device),
        }

        # Combine outputs
        return {
            "u_sc": sc_out["z_sc"],
            "u_st": st_out["z_st"],
            "u_bulk": bulk_out["z_bulk"],
            "recon_loss": sc_out["recon_loss_sc"] + st_out["recon_loss_st"],
            "kl_loss": sc_out["kl_loss_sc"] + st_out["kl_loss_st"],
        }