"""GraphEncoder: gene-level message passing on the guidance graph.

Produces (n_genes, latent_dim) vertex embeddings via learnable initial
embeddings + simplified GraphConv message passing over the guidance graph.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import LongTensor, Tensor


class GraphEncoder(nn.Module):
    """Gene-level graph encoder with learnable vertex embeddings.

    Uses 2 rounds of simplified GraphConv message passing:
        v_new = v + ReLU(W @ v_aggregated)
    where v_aggregated is the mean of neighbor embeddings.
    """

    def __init__(self, n_genes: int, latent_dim: int = 128) -> None:
        super().__init__()
        self.n_genes = n_genes
        self.latent_dim = latent_dim

        # Learnable initial vertex embeddings
        self.vertex_embed = nn.Embedding(n_genes, latent_dim)
        nn.init.xavier_normal_(self.vertex_embed.weight)

        # Message-passing linear layers (two rounds)
        self.W1 = nn.Linear(latent_dim, latent_dim)
        self.W2 = nn.Linear(latent_dim, latent_dim)

        # Edge index and weights stored at runtime
        self._edge_index: LongTensor | None = None
        self._edge_weight: Tensor | None = None

    def set_graph(self, edge_index: LongTensor, edge_weight: Tensor | None = None) -> None:
        """Store guidance graph connectivity for message passing.

        Args:
            edge_index: (2, n_edges) — source and target gene indices
            edge_weight: (n_edges,) optional edge weights
        """
        self._edge_index = edge_index
        self._edge_weight = edge_weight

    def _message_pass_once(self, v: Tensor, W: nn.Linear) -> Tensor:
        """Single round of message passing: aggregate neighbors + apply linear + residual."""
        if self._edge_index is None:
            return v

        edge_index = self._edge_index
        src, tgt = edge_index[0], edge_index[1]

        # Gather source node embeddings
        msg = v[src]  # (n_edges, latent_dim)
        if self._edge_weight is not None:
            msg = msg * self._edge_weight.unsqueeze(-1)

        # Aggregate by target node (mean)
        ones = torch.ones(src.size(0), device=v.device)
        deg = torch.zeros(v.size(0), device=v.device).scatter_add_(0, tgt, ones)
        deg = deg.clamp(min=1)
        aggr = torch.zeros_like(v).scatter_add_(0, tgt.unsqueeze(-1).expand_as(msg), msg)
        aggr = aggr / deg.unsqueeze(-1)

        # Linear transformation + residual
        out = v + torch.relu(W(aggr))
        return out

    def forward(self, x: Tensor | None = None) -> Tensor:
        """Run two rounds of message passing to produce final gene embeddings.

        Args:
            x: Ignored (compatibility with nn.Module interface).
               Uses self.vertex_embed.weight as starting embeddings.

        Returns:
            v_gene: (n_genes, latent_dim) — final gene embeddings
        """
        # Initial embeddings
        v = self.vertex_embed.weight  # (n_genes, latent_dim)

        # Round 1
        v = self._message_pass_once(v, self.W1)
        # Round 2
        v = self._message_pass_once(v, self.W2)

        return v
