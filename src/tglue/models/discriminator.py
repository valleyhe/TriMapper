"""Modality Discriminator for Triple-Modal GLUE.

Three-way classifier that distinguishes scRNA (idx=0), ST (idx=1), and Bulk (idx=2)
latent embeddings. Used for adversarial alignment post VAE pre-warm.

R1 gradient penalty stabilizes GAN training by penalizing large gradients
near the decision boundary.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# Modality label constants (must match encoders.py)
SCRNA_MODALITY = 0
ST_MODALITY = 1
BULK_MODALITY = 2


class ModalityDiscriminator(nn.Module):
    """3-way modality classifier MLP.

    Architecture per D-03:
        concat(u_sc, u_st, u_bulk) → Linear(3*latent_dim, 256)
        → LeakyReLU(0.2) → Linear(256, 256) → LeakyReLU(0.2) → Linear(256, 3)

    Each input is first mean-pooled across the spot dimension to produce
    a single representative embedding per modality.
    """

    def __init__(self, latent_dim: int = 128, hidden_dim: int = 256) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        self.net = nn.Sequential(
            nn.Linear(3 * latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 3),
        )

    def forward(
        self,
        u_sc: Tensor,
        u_st: Tensor,
        u_bulk: Tensor,
    ) -> Tensor:
        """Classify concatenated modality embeddings.

        Args:
            u_sc: (batch, latent_dim) — scRNA latent embeddings
            u_st: (batch, latent_dim) — ST latent embeddings
            u_bulk: (batch, latent_dim) — Bulk latent embeddings

        Returns:
            logits: (batch, 3) — unnormalized scores for [scRNA, ST, Bulk]

        DISC-01 FIX: Handle empty inputs in mean computation.
        Empty tensor mean() returns NaN, replaced with zeros.
        """
        # DISC-01 FIX: Handle empty inputs in mean-pooling
        if u_sc.shape[0] == 0:
            u_sc_mean = torch.zeros(1, self.latent_dim, device=u_sc.device)
        else:
            u_sc_mean = u_sc.mean(dim=0, keepdim=True)  # (1, latent_dim)

        if u_st.shape[0] == 0:
            u_st_mean = torch.zeros(1, self.latent_dim, device=u_st.device)
        else:
            u_st_mean = u_st.mean(dim=0, keepdim=True)  # (1, latent_dim)

        if u_bulk.shape[0] == 0:
            u_bulk_mean = torch.zeros(1, self.latent_dim, device=u_bulk.device)
        else:
            u_bulk_mean = u_bulk.mean(dim=0, keepdim=True)  # (1, latent_dim)

        # DISC-01 FIX: Return zero logits for all-empty degenerate case
        if u_sc.shape[0] == 0 and u_st.shape[0] == 0 and u_bulk.shape[0] == 0:
            return torch.zeros(1, 3, device=u_sc.device)  # Zero logits

        # Concatenate: (1, 3*latent_dim)
        concat = torch.cat([u_sc_mean, u_st_mean, u_bulk_mean], dim=-1)

        logits = self.net(concat)  # (1, 3)
        return logits


def r1_gradient_penalty(
    discriminator: nn.Module,
    u_sc: Tensor,
    u_st: Tensor,
    u_bulk: Tensor,
    reg_weight: float = 1.0,
) -> Tensor:
    """R1 gradient penalty for discriminator stability.

    Computes the gradient of the discriminator output w.r.t. the real samples
    (concatenated modality embeddings) and penalizes its L2 norm squared.
    This regularizes against large gradients near the decision boundary
    (as used in SCGLUE / spectral regularization).

    The gradient is computed for each modality embedding separately and
    then summed.

    Args:
        discriminator: The modality discriminator
        u_sc: (batch, latent_dim) — real scRNA embeddings
        u_st: (batch, latent_dim) — real ST embeddings
        u_bulk: (batch, latent_dim) — real Bulk embeddings
        reg_weight: Multiplier for the penalty term

    Returns:
        penalty: scalar R1 penalty (reg_weight * ||grad||^2 / 2)
    """
    # Ensure inputs require gradient for autograd
    u_sc = u_sc.detach().requires_grad_(True)
    u_st = u_st.detach().requires_grad_(True)
    u_bulk = u_bulk.detach().requires_grad_(True)

    # Forward pass through discriminator
    logits = discriminator(u_sc, u_st, u_bulk)  # (1, 3)

    # Use sum of logits as the scalar output for gradient computation
    # (any scalar output works for R1 penalty)
    loss = logits.sum()

    # Compute gradients w.r.t. each input
    grad_sc, grad_st, grad_bulk = torch.autograd.grad(
        outputs=loss,
        inputs=[u_sc, u_st, u_bulk],
        create_graph=True,
        retain_graph=True,
        grad_outputs=torch.ones_like(loss),
    )

    # R1 penalty: reg_weight * (||grad_sc||^2 + ||grad_st||^2 + ||grad_bulk||^2) / 2
    penalty = reg_weight * (
        (grad_sc.norm() ** 2) + (grad_st.norm() ** 2) + (grad_bulk.norm() ** 2)
    ) / 2
    return penalty


def adversarial_loss_scglue(
    u_sc: Tensor,
    u_st: Tensor,
    u_bulk: Tensor,
    discriminator: ModalityDiscriminator,
    modality_idx: int,
) -> Tensor:
    """Deprecated wrapper around `adversarial_loss_for_disc`.

    Keeps the legacy behavior where gradients do not flow back to the VAE,
    for backward compatibility with existing call sites that expect the old
    GLUE-style loss shape.
    """
    # Detach VAE outputs — only discriminator learns from this signal
    return adversarial_loss_for_disc(
        u_sc,
        u_st,
        u_bulk,
        discriminator,
        modality_idx,
    )


def adversarial_loss_for_vae(
    u_sc: Tensor,
    u_st: Tensor,
    u_bulk: Tensor,
    discriminator: ModalityDiscriminator,
    modality_idx: int,
) -> Tensor:
    """GLUE-style adversarial confusion loss that passes gradients to the VAE."""
    logits = discriminator(u_sc, u_st, u_bulk)
    target = torch.tensor([modality_idx], device=logits.device, dtype=torch.long)
    return F.cross_entropy(logits, target)


def adversarial_loss_for_disc(
    u_sc: Tensor,
    u_st: Tensor,
    u_bulk: Tensor,
    discriminator: ModalityDiscriminator,
    modality_idx: int,
) -> Tensor:
    """GLUE-style adversarial confusion loss that detaches VAE embeddings."""
    u_sc_det = u_sc.detach()
    u_st_det = u_st.detach()
    u_bulk_det = u_bulk.detach()
    logits = discriminator(u_sc_det, u_st_det, u_bulk_det)
    target = torch.tensor([modality_idx], device=logits.device, dtype=torch.long)
    return F.cross_entropy(logits, target)
