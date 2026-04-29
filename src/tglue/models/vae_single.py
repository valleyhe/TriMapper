"""SingleModalVAE: ST-only VAE baseline (AB-SINGLE ablation).

Per D-14-02: No scRNA, no Bulk, no adversarial alignment.
Simple ST-only VAE for baseline comparison.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from .encoders import NBDataEncoder, _nb_negloglikelihood


def _kl_gaussian(mean: Tensor, log_var: Tensor) -> Tensor:
    """KL divergence KL(N(mu,sigma) || N(0,I)).

    KL = -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
    """
    return -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp(), dim=-1)


class SingleModalVAE(nn.Module):
    """Single-modal VAE for ST only (AB-SINGLE ablation experiment).

    Per D-14-02, this model has:
    - NO scRNA encoder (no enc_sc)
    - NO Bulk encoder (no enc_bulk)
    - NO guidance graph components (no graph_enc, graph_dec)
    - NO adversarial alignment (no discriminator)

    Simple baseline VAE: encode ST counts → decode with NB likelihood.
    """

    def __init__(
        self,
        n_genes: int,
        latent_dim: int = 128,
        enc_st_hidden: int = 256,
    ) -> None:
        super().__init__()
        self.n_genes = n_genes
        self.latent_dim = latent_dim

        # ONLY ST encoder (per D-14-02)
        self.enc_st = NBDataEncoder(n_genes, latent_dim, enc_st_hidden)

        # Gene projection for NB reconstruction
        self.gene_proj_st = nn.Linear(latent_dim, n_genes, bias=False)

    @property
    def gene_count(self) -> int:
        return self.n_genes

    def compute_kl(self, mean: Tensor, log_var: Tensor) -> Tensor:
        return _kl_gaussian(mean, log_var)

    def forward(
        self,
        x_st: Tensor,
    ) -> dict[str, Tensor]:
        """Run single-modal forward pass (ST only).

        Args:
            x_st: (n_spots, n_genes) ST counts

        Returns:
            dict with keys:
                u_st — ST latent tensor
                recon_loss — NB reconstruction loss
                kl_loss — KL divergence loss

        NOTE: No u_sc, u_bulk, v_gene keys in output (per D-14-02).
        """
        # --- ST encoder ---
        z_st, mean_st, log_var_st = self.enc_st(x_st)
        kl_st = self.compute_kl(mean_st, log_var_st)

        # --- NB reconstruction ---
        pred_mean_st = torch.relu(self.gene_proj_st(mean_st))
        pi_st = self.enc_st.pi(mean_st)
        recon_st = _nb_negloglikelihood(x_st, pred_mean_st, pi_st)

        recon_loss = recon_st.mean()
        kl_loss = kl_st.mean()

        return {
            "u_st": z_st,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
        }