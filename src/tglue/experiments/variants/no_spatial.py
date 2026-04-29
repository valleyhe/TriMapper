"""Identity pass-through spatial scaffold for AB-02 (D-AB02).

Replaces fusion_conv processing with pure identity transformation.
Returns u_st.detach() unchanged, no spatial loss computed.

Key: u_st.detach() prevents gradient leakage (Pitfall 2).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class NoSpatialScaffold(nn.Module):
    """Identity pass-through for spatial scaffold (AB-02 baseline).

    Replaces SpatialScaffold.fusion_conv with identity operation.
    Returns input tensor detached (no gradient).

    No trainable parameters, pure identity transformation.

    Parameters
    ----------
    latent_dim : int, default 128
        Latent dimension (stored for interface compatibility)
    """

    def __init__(self, latent_dim: int = 128) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        # No trainable parameters
        # Interface compatibility: spatial_adj is None for no-spatial variant
        self.spatial_adj = None

    def forward(self, u_st: Tensor) -> Tensor:
        """Return input tensor unchanged (detached).

        Parameters
        ----------
        u_st : Tensor
            (n_spots, latent_dim) ST latent embeddings

        Returns
        -------
        Tensor
            (n_spots, latent_dim) Same tensor, detached
        """
        # Pure identity: return detached input
        # u_st.detach() prevents gradient leakage (Pitfall 2)
        return u_st.detach()

    def compute_loss(self, u_st: Tensor) -> Tensor:
        """Spatial loss is zero (no smoothness constraint).

        Parameters
        ----------
        u_st : Tensor
            Any tensor (ignored)

        Returns
        -------
        Tensor
            Zero scalar tensor
        """
        return torch.zeros(1, device=u_st.device, dtype=u_st.dtype)