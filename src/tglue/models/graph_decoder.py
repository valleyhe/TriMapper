"""GraphDecoder: Bernoulli edge reconstruction from gene embeddings.

Two-layer MLP maps (v_i || v_j) → Bernoulli logit for each edge.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import LongTensor, Tensor


class GraphDecoder(nn.Module):
    """Bernoulli edge reconstruction decoder.

    Scores edge (i,j) as MLP([v_i; v_j]) → logit(p(edge)).
    Trained with binary cross entropy against positive/negative edge labels.
    """

    def __init__(self, latent_dim: int = 128) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        hidden = 128
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        v_gene: Tensor,
        pos_edge_index: LongTensor,
        neg_edge_index: LongTensor | None = None,
        neg_ratio: float = 0.1,  # Only sample 10% of positives as negatives to reduce memory
    ) -> Tensor:
        """Compute BCE loss for positive and (optionally) negative edges.

        Args:
            v_gene: (n_genes, latent_dim) — gene embeddings from GraphEncoder
            pos_edge_index: (2, n_pos) — ground-truth positive edge indices
            neg_edge_index: (2, n_neg) optional negative edges.
                            If None, negative samples are generated randomly.
            neg_ratio: Fraction of positive edges to sample as negatives (default 0.1).

        Returns:
            recon_loss: scalar — BCE loss (mean over all edges)
        """
        if (
            pos_edge_index is None
            or pos_edge_index.numel() == 0
            or pos_edge_index.shape[1] == 0
        ):
            return torch.tensor(0.0, device=v_gene.device, dtype=v_gene.dtype)

        src_pos, tgt_pos = pos_edge_index[0], pos_edge_index[1]
        # Sample a subset of positive edges to reduce memory usage
        n_pos_total = src_pos.size(0)
        n_pos_sample = min(n_pos_total, 2000)  # D-14: Cap at 2000 positive edges per batch (was 5000)
        if n_pos_sample < n_pos_total:
            perm = torch.randperm(n_pos_total, device=v_gene.device)[:n_pos_sample]
            src_pos = src_pos[perm]
            tgt_pos = tgt_pos[perm]

        pos_scores = self.mlp(torch.cat([v_gene[src_pos], v_gene[tgt_pos]], dim=-1)).squeeze(-1)
        pos_labels = torch.ones_like(pos_scores)

        if neg_edge_index is not None:
            src_neg, tgt_neg = neg_edge_index[0], neg_edge_index[1]
            # Also sample subset of negatives
            n_neg_total = src_neg.size(0)
            n_neg_sample = min(n_neg_total, int(n_pos_sample * neg_ratio))
            if n_neg_sample < n_neg_total:
                perm = torch.randperm(n_neg_total, device=v_gene.device)[:n_neg_sample]
                src_neg = src_neg[perm]
                tgt_neg = tgt_neg[perm]
            neg_scores = self.mlp(torch.cat([v_gene[src_neg], v_gene[tgt_neg]], dim=-1)).squeeze(-1)
            neg_labels = torch.zeros_like(neg_scores)
        else:
            # Random negative sampling: sample fewer negatives
            n_genes = v_gene.size(0)
            n_neg = int(n_pos_sample * neg_ratio)
            src_neg = torch.randint(0, n_genes, (n_neg,), device=v_gene.device)
            tgt_neg = torch.randint(0, n_genes, (n_neg,), device=v_gene.device)
            neg_scores = self.mlp(torch.cat([v_gene[src_neg], v_gene[tgt_neg]], dim=-1)).squeeze(-1)
            neg_labels = torch.zeros_like(neg_scores)

        scores = torch.cat([pos_scores, neg_scores], dim=0)
        labels = torch.cat([pos_labels, neg_labels], dim=0)
        if scores.numel() == 0:
            return torch.tensor(0.0, device=v_gene.device, dtype=v_gene.dtype)
        return nn.functional.binary_cross_entropy_with_logits(scores, labels)

    def predict_edges(self, v_gene: Tensor, edge_index: LongTensor) -> Tensor:
        """Return Bernoulli probabilities for given edges.

        Args:
            v_gene: (n_genes, latent_dim)
            edge_index: (2, n_edges)

        Returns:
            probs: (n_edges,) — edge probabilities in (0, 1)
        """
        src, tgt = edge_index[0], edge_index[1]
        scores = self.mlp(torch.cat([v_gene[src], v_gene[tgt]], dim=-1)).squeeze(-1)
        return torch.sigmoid(scores)
