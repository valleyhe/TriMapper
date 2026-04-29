"""Dual-Modal Discriminator for AB-DUAL ablation.

Two-way classifier that distinguishes scRNA (idx=0) and ST (idx=1)
latent embeddings. Used for adversarial alignment in DualModalVAE.

Per D-14-01: NO Bulk modality.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# Modality label constants (must match DualModalVAE)
SCRNA_MODALITY = 0
ST_MODALITY = 1


class DualModalDiscriminator(nn.Module):
    """2-way modality classifier MLP for AB-DUAL ablation.

    Architecture:
        concat(u_sc, u_st) → Linear(2*latent_dim, 256)
        → LeakyReLU(0.2) → Linear(256, 256) → LeakyReLU(0.2) → Linear(256, 2)

    Each input is first mean-pooled across the spot dimension to produce
    a single representative embedding per modality.
    """

    def __init__(self, latent_dim: int = 128, hidden_dim: int = 256) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        self.net = nn.Sequential(
            nn.Linear(2 * latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 2),  # 2-way: scRNA vs ST
        )

    def forward(
        self,
        u_sc: Tensor,
        u_st: Tensor,
    ) -> Tensor:
        """Classify concatenated modality embeddings.

        Args:
            u_sc: (batch, latent_dim) — scRNA latent embeddings
            u_st: (batch, latent_dim) — ST latent embeddings

        Returns:
            logits: (batch, 2) — unnormalized scores for [scRNA, ST]
        """
        # Mean-pool across spots to get one representative embedding per modality
        u_sc_mean = u_sc.mean(dim=0, keepdim=True)  # (1, latent_dim)
        u_st_mean = u_st.mean(dim=0, keepdim=True)  # (1, latent_dim)

        # Concatenate: (1, 2*latent_dim)
        concat = torch.cat([u_sc_mean, u_st_mean], dim=-1)

        logits = self.net(concat)  # (1, 2)
        return logits


def r1_gradient_penalty_dual(
    discriminator: nn.Module,
    u_sc: Tensor,
    u_st: Tensor,
    reg_weight: float = 1.0,
) -> Tensor:
    """R1 gradient penalty for dual-modal discriminator stability.

    Args:
        discriminator: The dual-modal discriminator
        u_sc: (batch, latent_dim) — real scRNA embeddings
        u_st: (batch, latent_dim) — real ST embeddings
        reg_weight: Multiplier for the penalty term

    Returns:
        penalty: scalar R1 penalty (reg_weight * ||grad||^2 / 2)
    """
    u_sc = u_sc.detach().requires_grad_(True)
    u_st = u_st.detach().requires_grad_(True)

    logits = discriminator(u_sc, u_st)  # (1, 2)
    loss = logits.sum()

    grad_sc, grad_st = torch.autograd.grad(
        outputs=loss,
        inputs=[u_sc, u_st],
        create_graph=True,
        retain_graph=True,
        grad_outputs=torch.ones_like(loss),
    )

    penalty = reg_weight * ((grad_sc.norm() ** 2) + (grad_st.norm() ** 2)) / 2
    return penalty


def adversarial_loss_for_vae_dual(
    u_sc: Tensor,
    u_st: Tensor,
    discriminator: DualModalDiscriminator,
    modality_idx: int,
) -> Tensor:
    """GLUE-style adversarial confusion loss that passes gradients to the VAE."""
    logits = discriminator(u_sc, u_st)
    target = torch.tensor([modality_idx], device=logits.device, dtype=torch.long)
    return F.cross_entropy(logits, target)


def adversarial_loss_for_disc_dual(
    u_sc: Tensor,
    u_st: Tensor,
    discriminator: DualModalDiscriminator,
    modality_idx: int,
) -> Tensor:
    """GLUE-style adversarial confusion loss that detaches VAE embeddings."""
    u_sc_det = u_sc.detach()
    u_st_det = u_st.detach()
    logits = discriminator(u_sc_det, u_st_det)
    target = torch.tensor([modality_idx], device=logits.device, dtype=torch.long)
    return F.cross_entropy(logits, target)