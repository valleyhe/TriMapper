"""Encoder modules for TripleModalVAE.

NBDataEncoder: Gaussian encoder for scRNA and ST count data (reparameterized).
VanillaDataEncoder: Deterministic projection for Bulk expression (prior only, no stochasticity).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint


def _nb_negloglikelihood(x: Tensor, mean: Tensor, pi: Tensor) -> Tensor:
    """Negative binomial negative log-likelihood per sample.

    Parameterization: NB(mean=mu, theta=exp(pi)).
    log_p(x) = log Gamma(x+theta) - log Gamma(theta) - log Gamma(x+1)
               + theta * log(theta / (theta + mu)) + x * log(mu / (theta + mu))
    Using digamma/trigamma approximations for the log-gamma differences.

    P0-01: Clamp ratios to prevent log(0) edge cases.
    Per D-01 locked decision: Clamp ratios [1e-6, 1.0-1e-6] directly.
    """
    # NB is only defined for non-negative counts; clamp for numerical safety
    x = torch.clamp(x, min=0.0)
    theta = torch.clamp(torch.exp(pi), min=1e-6, max=1e6)  # Prevent lgamma overflow
    # Clip mean to avoid numerical issues
    mu = torch.clamp(mean, min=1e-6)

    # P0-01 FIX: Clamp ratios directly, not just mean
    # This prevents log(0) when theta << mu or theta >> mu
    eps_ratio = 1e-6
    ratio_theta_mu = torch.clamp(
        theta / (theta + mu), min=eps_ratio, max=1.0 - eps_ratio
    )
    ratio_mu_theta = torch.clamp(
        mu / (theta + mu), min=eps_ratio, max=1.0 - eps_ratio
    )

    # Log-likelihood using torch.lgamma (log gamma function)
    ll = (
        torch.lgamma(x + theta)
        - torch.lgamma(theta)
        - torch.lgamma(x + 1)
        + theta * torch.log(ratio_theta_mu)  # Safe log with clamped ratio
        + x * torch.log(ratio_mu_theta)      # Safe log with clamped ratio
    )
    # Sum over gene dimension; guard against numerical overflow
    nll = -ll.sum(dim=-1)
    if not torch.isfinite(nll).all():
        nll = torch.where(
            torch.isfinite(nll),
            nll,
            torch.tensor(1e6, device=nll.device, dtype=nll.dtype),
        )
    return nll


class NBDataEncoder(nn.Module):
    """Gaussian + optional NB-dispersion encoder for scRNA/ST count data.

    Architecture: 2-layer MLP + BatchNorm + LeakyReLU → Gaussian parameters.
    Produces reparameterized sample z ~ N(mean, exp(0.5*log_var)).
    Optionally outputs NB dispersion for negative binomial reconstruction loss.

    D-14: Gradient checkpointing support for memory efficiency.
    """

    def __init__(
        self,
        n_genes: int,
        latent_dim: int = 128,
        hidden_dim: int = 256,
        use_checkpointing: bool = False,  # D-14: default False for backward compat
    ) -> None:
        super().__init__()
        self.n_genes = n_genes
        self.latent_dim = latent_dim
        self.use_checkpointing = use_checkpointing

        # Define network as separate modules for checkpointing
        self.layer1 = nn.Sequential(
            nn.Linear(n_genes, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.LeakyReLU(0.2),
        )
        self.mean_head = nn.Linear(latent_dim, latent_dim)
        self.log_var_head = nn.Linear(latent_dim, latent_dim)
        # NB dispersion: one theta per gene
        self.pi_head = nn.Linear(latent_dim, n_genes)

        # Backward compatibility aliases (D-14: renamed for checkpointing but keep old API)
        # net was a single Sequential, now split into layer1 + layer2
        # pi was renamed to pi_head
        self._init_backward_compat_aliases()

    def _init_backward_compat_aliases(self):
        """Set up backward compatibility for pre-10-03 code."""
        # Create a combined net Sequential for backward compat (used by tests)
        self.net = nn.Sequential(
            self.layer1[0],  # Linear(n_genes, hidden_dim)
            self.layer1[1],  # BatchNorm1d
            self.layer1[2],  # LeakyReLU
            self.layer2[0],  # Linear(hidden_dim, latent_dim)
            self.layer2[1],  # BatchNorm1d
            self.layer2[2],  # LeakyReLU
        )
        # Rename back for backward compat access
        self.mean = self.mean_head
        self.log_var = self.log_var_head
        self.pi = self.pi_head

    def _forward_core(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Core forward logic (checkpointed region)."""
        h1 = self.layer1(x)
        h2 = self.layer2(h1)
        mean = self.mean_head(h2)
        log_var = self.log_var_head(h2)
        return mean, log_var

    def forward(
        self, x: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Encode count data to latent representation.

        Args:
            x: (batch, n_genes) — scRNA or ST count vector

        Returns:
            z: (batch, latent_dim) — reparameterized sample
            mean: (batch, latent_dim) — Gaussian mean
            log_var: (batch, latent_dim) — Gaussian log variance
        """
        if self.use_checkpointing:
            # D-14: use_reentrant=False required in PyTorch 2.4+
            mean, log_var = checkpoint(
                self._forward_core, x,
                use_reentrant=False,
            )
        else:
            mean, log_var = self._forward_core(x)

        # Reparameterization trick (outside checkpointed region - stochastic)
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mean + eps * std
        return z, mean, log_var

    def compute_nb_recon_loss(
        self, x: Tensor, mean: Tensor
    ) -> Tensor:
        """Compute per-sample NB negative log-likelihood using the pi head."""
        pi = self.pi_head(mean)
        return _nb_negloglikelihood(x, mean, pi)


class VanillaDataEncoder(nn.Module):
    """Deterministic projection for Bulk expression prior.

    Bulk is used ONLY as a prior for the latent space — it is NOT a reconstruction target.
    No stochastic sampling, no log_var. Returns a cluster-level mean embedding.

    D-14: Gradient checkpointing support for memory efficiency.
    """

    def __init__(
        self,
        n_genes: int,
        latent_dim: int = 128,
        use_checkpointing: bool = False,  # D-14: default False
    ) -> None:
        super().__init__()
        self.n_genes = n_genes
        self.latent_dim = latent_dim
        self.use_checkpointing = use_checkpointing
        self.proj = nn.Linear(n_genes, latent_dim, bias=False)

    def _forward_core(self, x_bulk: Tensor) -> Tensor:
        """Core forward logic (checkpointed region)."""
        return self.proj(x_bulk)

    def forward(self, x_bulk: Tensor) -> Tensor:
        """Project Bulk expression to prior mean embedding.

        Args:
            x_bulk: (n_clusters, n_genes) — Bulk expression per cluster

        Returns:
            prior_mean: (n_clusters, latent_dim) — deterministic embedding
        """
        if self.use_checkpointing:
            # D-14: use_reentrant=False required in PyTorch 2.4+
            return checkpoint(
                self._forward_core, x_bulk,
                use_reentrant=False,
            )
        else:
            return self._forward_core(x_bulk)
