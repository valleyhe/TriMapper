"""Shared utility functions for VAE models.

This module contains common functions used across multiple VAE implementations
to reduce code duplication and maintenance overhead.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def kl_gaussian(mean: Tensor, log_var: Tensor) -> Tensor:
    """KL divergence KL(N(mu, sigma) || N(0, I)).

    KL = -0.5 * sum(1 + log_var - mu^2 - exp(log_var))

    Args:
        mean: Mean of the Gaussian distribution
        log_var: Log variance of the Gaussian distribution

    Returns:
        KL divergence per sample (summed over latent dimensions)
    """
    return -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp(), dim=-1)


def mse_reconstruction_loss(x: Tensor, pred: Tensor) -> Tensor:
    """MSE reconstruction loss (memory-efficient alternative to NB loss).

    Args:
        x: Target tensor
        pred: Predicted tensor

    Returns:
        Per-sample MSE loss (summed over feature dimensions)
    """
    return F.mse_loss(pred, x, reduction='none').sum(dim=-1)