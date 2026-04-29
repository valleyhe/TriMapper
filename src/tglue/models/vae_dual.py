"""DualModalVAE: shared latent space for scRNA + ST only (AB-DUAL ablation).

Per D-14-01: Removes Bulk encoder from TripleModalVAE to quantify Bulk prior contribution.
This is a fresh nn.Module class (not a subclass) to avoid inheriting enc_bulk.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from .encoders import NBDataEncoder, _nb_negloglikelihood
from .graph_encoder import GraphEncoder
from .graph_decoder import GraphDecoder
from .utils import kl_gaussian


class DualModalVAE(nn.Module):
    """Dual-modal VAE for scRNA + ST only (AB-DUAL ablation experiment).

    Per D-14-01, this model has NO Bulk encoder to quantify the contribution
    of Bulk prior to the triple-modal integration.

    Architecture:
    - scRNA encoder: NBDataEncoder (Gaussian + NB recon)
    - ST encoder: NBDataEncoder (Gaussian + NB recon)
    - GraphEncoder: gene-level message passing on guidance graph
    - GraphDecoder: Bernoulli edge reconstruction

    forward() returns a dict with latent tensors and losses (no u_bulk key).
    """

    def __init__(
        self,
        n_genes: int,
        latent_dim: int = 128,
        enc_sc_hidden: int = 256,
        enc_st_hidden: int = 256,
        use_checkpointing: bool = False,  # D-14: Enable for memory efficiency
        disable_graph_recon: bool = False,  # D-14: Disable graph decoder for memory
    ) -> None:
        super().__init__()
        self.n_genes = n_genes
        self.latent_dim = latent_dim
        self.disable_graph_recon = disable_graph_recon

        # Modality encoders (NO Bulk encoder per D-14-01)
        self.enc_sc = NBDataEncoder(n_genes, latent_dim, enc_sc_hidden, use_checkpointing=use_checkpointing)
        self.enc_st = NBDataEncoder(n_genes, latent_dim, enc_st_hidden, use_checkpointing=use_checkpointing)

        # Guidance graph components
        self.graph_enc = GraphEncoder(n_genes, latent_dim)
        self.graph_dec = GraphDecoder(latent_dim)

        # Gene projection layers for NB reconstruction
        self.gene_proj_sc = nn.Linear(latent_dim, n_genes, bias=False)
        self.gene_proj_st = nn.Linear(latent_dim, n_genes, bias=False)

    @property
    def gene_count(self) -> int:
        return self.n_genes

    def compute_kl(self, mean: Tensor, log_var: Tensor) -> Tensor:
        return kl_gaussian(mean, log_var)

    def forward(
        self,
        x_sc: Tensor,
        x_st: Tensor,
        guidance_data,
    ) -> dict[str, Tensor]:
        """Run dual-modal forward pass (NO Bulk modality).

        Args:
            x_sc: (n_cells, n_genes) scRNA counts
            x_st: (n_spots, n_genes) ST counts
            guidance_data: PyG Data with edge_index, optionally edge_weight

        Returns:
            dict with keys:
                u_sc — scRNA latent tensor
                u_st — ST latent tensor
                v_gene — (n_genes, latent_dim) gene embeddings
                recon_loss — NB reconstruction for scRNA + ST
                kl_loss — KL divergence for scRNA + ST
                graph_recon_loss — Graph edge reconstruction loss

        NOTE: No u_bulk key in output (per D-14-01).
        """
        # --- scRNA encoder ---
        z_sc, mean_sc, log_var_sc = self.enc_sc(x_sc)
        kl_sc = self.compute_kl(mean_sc, log_var_sc)

        # --- ST encoder ---
        z_st, mean_st, log_var_st = self.enc_st(x_st)
        kl_st = self.compute_kl(mean_st, log_var_st)

        # --- Graph encoder ---
        if hasattr(guidance_data, "edge_index"):
            self.graph_enc.set_graph(
                guidance_data.edge_index,
                getattr(guidance_data, "edge_weight", None),
            )
        v_gene = self.graph_enc()

        # --- Graph decoder reconstruction loss ---
        # D-14: Optionally skip graph reconstruction for memory efficiency
        if self.disable_graph_recon:
            graph_recon_loss = torch.tensor(0.0, device=x_sc.device)
        else:
            edge_index = getattr(guidance_data, "edge_index", None)
            if (
                edge_index is not None
                and edge_index.numel() > 0
                and edge_index.shape[1] > 0
            ):
                graph_recon_loss = self.graph_dec(v_gene, edge_index)
            else:
                graph_recon_loss = torch.tensor(0.0, device=x_sc.device)

        # --- NB reconstruction losses ---
        pred_mean_sc = torch.relu(self.gene_proj_sc(mean_sc))
        pred_mean_st = torch.relu(self.gene_proj_st(mean_st))

        pi_sc = self.enc_sc.pi(mean_sc)
        pi_st = self.enc_st.pi(mean_st)

        recon_sc = _nb_negloglikelihood(x_sc, pred_mean_sc, pi_sc)
        recon_st = _nb_negloglikelihood(x_st, pred_mean_st, pi_st)

        recon_loss = recon_sc.mean() + recon_st.mean()
        kl_loss = kl_sc.mean() + kl_st.mean()

        return {
            "u_sc": z_sc,
            "u_st": z_st,
            "v_gene": v_gene,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
            "graph_recon_loss": graph_recon_loss,
        }