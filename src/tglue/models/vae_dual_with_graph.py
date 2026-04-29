"""DualModalVAEWithGraph: Memory-efficient dual-modal VAE with guidance graph.

Per 15-FA-02: Fair ablation comparison - remove Bulk but keep guidance graph.
Uses MSE loss and sequential processing for memory efficiency.

Architecture:
- latent_dim: 64 (unified)
- hidden_dim: 128 (unified)
- Guidance graph encoder/decoder (preserved)
- MSE reconstruction
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from .encoders import NBDataEncoder
from .graph_encoder import GraphEncoder
from .graph_decoder import GraphDecoder
from .utils import kl_gaussian, mse_reconstruction_loss


class DualModalVAEWithGraph(nn.Module):
    """Memory-efficient dual-modal VAE with guidance graph.

    Key features:
    1. scRNA + ST encoders (no Bulk)
    2. Guidance graph encoder/decoder preserved
    3. MSE reconstruction for memory efficiency
    4. Sequential modality processing
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

        # Modality encoders
        self.enc_sc = NBDataEncoder(n_genes, latent_dim, hidden_dim, use_checkpointing=use_checkpointing)
        self.enc_st = NBDataEncoder(n_genes, latent_dim, hidden_dim, use_checkpointing=use_checkpointing)

        # Guidance graph components
        self.graph_enc = GraphEncoder(n_genes, latent_dim)
        self.graph_dec = GraphDecoder(latent_dim)

        # Gene projection layers for reconstruction
        self.gene_proj_sc = nn.Linear(latent_dim, n_genes, bias=False)
        self.gene_proj_st = nn.Linear(latent_dim, n_genes, bias=False)

    @property
    def gene_count(self) -> int:
        return self.n_genes

    def compute_kl(self, mean: Tensor, log_var: Tensor) -> Tensor:
        return kl_gaussian(mean, log_var)

    def set_graph(self, edge_index: Tensor) -> None:
        """Set guidance graph for encoder."""
        self.graph_enc.set_graph(edge_index)

    def forward_sc_only(self, x_sc: Tensor) -> dict[str, Tensor]:
        """Forward pass for scRNA modality only."""
        z_sc, mean_sc, log_var_sc = self.enc_sc(x_sc)
        kl_sc = self.compute_kl(mean_sc, log_var_sc)

        # MSE reconstruction
        pred_sc = torch.relu(self.gene_proj_sc(mean_sc))
        recon_sc = mse_reconstruction_loss(x_sc, pred_sc)

        return {
            "z_sc": z_sc,
            "recon_loss_sc": recon_sc.mean(),
            "kl_loss_sc": kl_sc.mean(),
        }

    def forward_st_only(self, x_st: Tensor) -> dict[str, Tensor]:
        """Forward pass for ST modality only."""
        z_st, mean_st, log_var_st = self.enc_st(x_st)
        kl_st = self.compute_kl(mean_st, log_var_st)

        # MSE reconstruction
        pred_st = torch.relu(self.gene_proj_st(mean_st))
        recon_st = mse_reconstruction_loss(x_st, pred_st)

        return {
            "z_st": z_st,
            "recon_loss_st": recon_st.mean(),
            "kl_loss_st": kl_st.mean(),
        }

    def forward_graph_only(self, edge_index: Tensor) -> dict[str, Tensor]:
        """Forward pass for guidance graph only."""
        v_gene = self.graph_enc()
        graph_recon_loss = self.graph_dec(v_gene, edge_index)

        return {
            "v_gene": v_gene,
            "graph_recon_loss": graph_recon_loss,
        }

    def forward_combined(
        self,
        x_sc: Tensor,
        x_st: Tensor,
        edge_index: Tensor,
    ) -> dict[str, Tensor]:
        """Combined forward pass with graph."""
        # Process scRNA
        sc_out = self.forward_sc_only(x_sc)
        # Process ST
        st_out = self.forward_st_only(x_st)
        # Process graph
        graph_out = self.forward_graph_only(edge_index)

        return {
            "u_sc": sc_out["z_sc"],
            "u_st": st_out["z_st"],
            "v_gene": graph_out["v_gene"],
            "recon_loss": sc_out["recon_loss_sc"] + st_out["recon_loss_st"],
            "kl_loss": sc_out["kl_loss_sc"] + st_out["kl_loss_st"],
            "graph_recon_loss": graph_out["graph_recon_loss"],
        }