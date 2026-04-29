"""TripleModalVAE: shared latent space for scRNA, ST, and Bulk + guidance graph components."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from .encoders import NBDataEncoder, VanillaDataEncoder, _nb_negloglikelihood
from .graph_encoder import GraphEncoder
from .graph_decoder import GraphDecoder
from .utils import kl_gaussian


class TripleModalVAE(nn.Module):
    """Triple-modal VAE with shared Gaussian latent space (D=128).

    - scRNA and ST encoders: NBDataEncoder (Gaussian + NB recon)
    - Bulk encoder: VanillaDataEncoder (deterministic prior only, no recon)
    - GraphEncoder: gene-level message passing on guidance graph
    - GraphDecoder: Bernoulli edge reconstruction

    forward() returns a dict with latent tensors and losses.
    """

    def __init__(
        self,
        n_genes: int,
        latent_dim: int = 128,
        enc_sc_hidden: int = 256,
        enc_st_hidden: int = 256,
        enc_bulk_hidden: int = 128,
    ) -> None:
        super().__init__()
        self.n_genes = n_genes
        self.latent_dim = latent_dim

        # Modality encoders
        self.enc_sc = NBDataEncoder(n_genes, latent_dim, enc_sc_hidden)
        self.enc_st = NBDataEncoder(n_genes, latent_dim, enc_st_hidden)
        self.enc_bulk = VanillaDataEncoder(n_genes, latent_dim)

        # Guidance graph components
        self.graph_enc = GraphEncoder(n_genes, latent_dim)
        self.graph_dec = GraphDecoder(latent_dim)

        # Gene projection layers for NB reconstruction (BF-01 fix)
        # These project latent embeddings to gene-level means for reconstruction.
        # Previously created in forward() causing unregistered parameters -> NaN.
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
        x_bulk: Tensor,
        guidance_data,
    ) -> dict[str, Tensor]:
        """Run triple-modal forward pass.

        Args:
            x_sc: (n_spots, n_genes) scRNA counts
            x_st: (n_spots, n_genes) ST counts
            x_bulk: (n_clusters, n_genes) Bulk expression
            guidance_data: PyG Data with edge_index, optionally edge_weight

        Returns:
            dict with keys:
                u_sc, u_st, u_bulk — latent tensors
                v_gene — (n_genes, latent_dim) gene embeddings
                recon_loss — NB reconstruction for scRNA + ST
                kl_loss — KL divergence for scRNA + ST (Bulk has none)
        """
        # VAE-05: Empty batch guard — prevent BatchNorm1d crash and NaN losses
        if x_sc.shape[0] == 0 or x_st.shape[0] == 0:
            device = x_sc.device if x_sc.numel() > 0 else (
                x_st.device if x_st.numel() > 0 else x_bulk.device
            )
            zero = torch.tensor(0.0, device=device)
            latent_dim = self.latent_dim
            return {
                "u_sc": torch.empty(0, latent_dim, device=device),
                "u_st": torch.empty(0, latent_dim, device=device),
                "u_bulk": torch.empty(0, latent_dim, device=device),
                "v_gene": torch.empty(0, latent_dim, device=device),
                "recon_loss": zero,
                "kl_loss": zero,
                "graph_recon_loss": zero,
            }

        # --- scRNA encoder ---
        z_sc, mean_sc, log_var_sc = self.enc_sc(x_sc)
        kl_sc = self.compute_kl(mean_sc, log_var_sc)

        # --- ST encoder ---
        z_st, mean_st, log_var_st = self.enc_st(x_st)
        kl_st = self.compute_kl(mean_st, log_var_st)

        # --- Bulk prior (deterministic, no KL, no recon) ---
        u_bulk = self.enc_bulk(x_bulk)

        # --- Graph encoder ---
        if hasattr(guidance_data, "edge_index"):
            self.graph_enc.set_graph(
                guidance_data.edge_index,
                getattr(guidance_data, "edge_weight", None),
            )
        v_gene = self.graph_enc()

        # --- Graph decoder reconstruction loss ---
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
        # Use mean output as "predicted count distribution mean"
        # We pass the reparameterized z as context to build a prediction mean.
        # Strategy: decode mean_sc through a shared gene-projection to get a gene-level mean.
        # For simplicity, use mean_sc @ gene_proj as predicted gene mean.
        # A minimal approach: just reconstruct using mean_sc as proxy for the latent centroid.
        # BF-01: gene_proj layers now in __init__, properly registered for optimization.
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
            "u_bulk": u_bulk,
            "v_gene": v_gene,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
            "graph_recon_loss": graph_recon_loss,
        }
