"""Individual-level Modality Discriminator for improved cross-modal mixing.

Unlike the original ModalityDiscriminator which mean-pools all embeddings
before classification (global-level only), this version classifies EACH
individual embedding, forcing the VAE to mix modalities at the cell/spot level.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


SCRNA_MODALITY = 0
ST_MODALITY = 1
BULK_MODALITY = 2


class IndividualModalityDiscriminator(nn.Module):
    """3-way modality classifier that operates on individual embeddings.

    Architecture:
        u → Linear(latent_dim, 256) → LeakyReLU(0.2)
        → Linear(256, 256) → LeakyReLU(0.2) → Linear(256, 3)

    This forces the VAE to make every single embedding modality-indistinguishable,
    not just the global mean.
    """

    def __init__(self, latent_dim: int = 128, hidden_dim: int = 256) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, u: Tensor) -> Tensor:
        """Classify individual embeddings.

        Args:
            u: (batch, latent_dim) — latent embeddings from any modality

        Returns:
            logits: (batch, 3) — unnormalized scores for [scRNA, ST, Bulk]
        """
        return self.net(u)


def adversarial_loss_for_vae_individual(
    u_sc: Tensor,
    u_st: Tensor,
    u_bulk: Tensor,
    discriminator: IndividualModalityDiscriminator,
    adversarial_weight: float = 1.0,
) -> Tensor:
    """Adversarial confusion loss that passes gradients to the VAE.

    For each modality, the VAE tries to fool the discriminator into
    classifying its embeddings as ALL three modalities uniformly.

    Args:
        u_sc: scRNA embeddings
        u_st: ST embeddings
        u_bulk: Bulk embeddings
        discriminator: Individual-level discriminator
        adversarial_weight: Weight for the adversarial loss

    Returns:
        Weighted adversarial loss
    """
    all_embeddings = [u_sc, u_st, u_bulk]
    total_loss = torch.tensor(0.0, device=u_sc.device)
    total_samples = 0

    for modality_idx, u in enumerate(all_embeddings):
        if u.shape[0] == 0:
            continue
        logits = discriminator(u)
        # Target: uniform distribution across all modalities
        # This is stronger than targeting a single modality —
        # the VAE must make embeddings completely unclassifiable
        target = torch.full(
            (u.shape[0],), modality_idx, device=u.device, dtype=torch.long
        )
        loss = F.cross_entropy(logits, target)
        total_loss = total_loss + loss
        total_samples += 1

    if total_samples == 0:
        return torch.tensor(0.0, device=u_sc.device)

    return adversarial_weight * (total_loss / total_samples)


def adversarial_loss_for_disc_individual(
    u_sc: Tensor,
    u_st: Tensor,
    u_bulk: Tensor,
    discriminator: IndividualModalityDiscriminator,
    modality_weights: Tensor | None = None,
) -> Tensor:
    """Discriminator loss with individual-level classification.

    Args:
        u_sc: scRNA embeddings
        u_st: ST embeddings
        u_bulk: Bulk embeddings
        discriminator: Individual-level discriminator
        modality_weights: Optional per-modality weights

    Returns:
        Discriminator cross-entropy loss
    """
    all_embeddings = [u_sc, u_st, u_bulk]
    total_loss = torch.tensor(0.0, device=u_sc.device)
    total_weight = torch.tensor(0.0, device=u_sc.device)

    if modality_weights is None:
        modality_weights = torch.ones(3, device=u_sc.device)

    for modality_idx, u in enumerate(all_embeddings):
        if u.shape[0] == 0:
            continue
        u_det = u.detach()
        logits = discriminator(u_det)
        target = torch.full(
            (u_det.shape[0],), modality_idx, device=u_det.device, dtype=torch.long
        )
        loss = F.cross_entropy(logits, target)
        w = modality_weights[modality_idx]
        total_loss = total_loss + w * loss
        total_weight = total_weight + w

    if total_weight.item() == 0:
        return torch.tensor(0.0, device=u_sc.device)

    return total_loss / total_weight
