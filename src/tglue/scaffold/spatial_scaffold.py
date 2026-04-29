"""Spatial k-NN scaffold using squidpy for spatial transcriptomics.

SP-01: Build spatial k-NN graph from ST coordinates using squidpy.
SP-02: fusion_conv reads u_st.detach() only — no gradient flows back to VAE encoder.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import scipy.sparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


def build_spatial_knn(
    st_adata,
    n_neighbors: int = 6,
    coord_type: str = "grid",
) -> Tuple[scipy.sparse.csr_matrix, np.ndarray]:
    """Build spatial k-NN adjacency matrix from ST coordinates using squidpy.

    Parameters
    ----------
    st_adata : ann.AnnData
        Spatial transcriptomics AnnData object with `.obsm['spatial']` coordinates.
    n_neighbors : int, default 6
        Number of spatial neighbors (k for k-NN graph).
    coord_type : str, default 'grid'
        Coordinate type for squidpy:
        - 'grid' for Visium (row, col in .obsm['spatial'])
        - 'coordinates' for general x,y coordinates

    Returns
    -------
    Tuple[scipy.sparse.csr_matrix, np.ndarray]
        - spatial_adj: (n_spots, n_spots) sparse adjacency matrix
        - coords: (n_spots, 2) array of spatial coordinates
    """
    import squidpy as sq

    # squidpy computes spatial neighbors in-place
    # squidpy 1.6.5 uses n_neighs (not n_neighbors) and key_added (not key)
    sq.gr.spatial_neighbors(
        st_adata,
        n_neighs=n_neighbors,
        coord_type=coord_type,
        key_added="spatial",
    )

    # Extract adjacency matrix from squidpy results
    # st_adata.obsp['spatial_connectivities'] contains the k-NN adjacency
    adj = st_adata.obsp["spatial_connectivities"]
    if not isinstance(adj, scipy.sparse.csr_matrix):
        adj = scipy.sparse.csr_matrix(adj)

    # Extract coordinates
    coords = st_adata.obsm["spatial"].copy()
    if coords.shape[1] > 2:
        # DS-02 FIX: Warn user about 3D coordinate truncation
        logger.warning(
            f"Spatial coordinates have {coords.shape[1]} dimensions, "
            f"truncating to 2D (first two columns). "
            f"If you intended 3D spatial analysis, preprocess coordinates first."
        )
        coords = coords[:, :2]  # Take first 2 dimensions if >2D

    return adj, coords


class SpatialScaffold(nn.Module):
    """Spatial-aware latent integration via fusion_conv.

    fusion_conv reads u_st.detach() — NO gradient flows from spatial scaffold to VAE.
    This is the anti-pattern enforcement: spatial scaffold is downstream consumer only.

    In Phase 2, fusion_conv output will condition ST decoder.
    Here: scaffold architecture defined, initialized, partially wired but NOT trained end-to-end.
    """

    def __init__(
        self,
        latent_dim: int = 128,
        n_neighbors: int = 6,
        fusion_hidden: int = 128,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.n_neighbors = n_neighbors
        self.fusion_hidden = fusion_hidden

        # Spatial adjacency (set via set_spatial_graph)
        self.spatial_adj: Optional[scipy.sparse.csr_matrix] = None

        # fusion_conv: 2-layer MLP reading u_st.detach()
        # u_st.detach() -> fusion_hidden -> latent_dim
        self.fusion_conv = nn.Sequential(
            nn.Linear(latent_dim, fusion_hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(fusion_hidden, latent_dim),
        )

        # Spatial bias projection: project from neighbor aggregation to latent space
        self.spatial_bias_proj = nn.Linear(latent_dim, latent_dim)

    def set_spatial_graph(self, adj_matrix: scipy.sparse.csr_matrix) -> None:
        """Set the spatial k-NN adjacency matrix from squidpy.

        Parameters
        ----------
        adj_matrix : scipy.sparse.csr_matrix
            Shape (n_spots, n_spots), k-NN adjacency from build_spatial_knn.
        """
        self.spatial_adj = adj_matrix

    def _compute_spatial_bias(self, u_st_detached: Tensor) -> Tensor:
        """Aggregate neighboring latent representations to compute spatial bias.

        Parameters
        ----------
        u_st_detached : Tensor
            (n_spots, latent_dim) — detached from VAE graph.

        Returns
        -------
        Tensor
            (n_spots, latent_dim) — spatial bias term to add to fused representation.
        """
        if self.spatial_adj is None:
            return torch.zeros_like(u_st_detached)

        adj = self.spatial_adj
        n_spots = adj.shape[0]

        # Convert to COO for index access
        adj_coo = adj.tocoo()
        row_indices = adj_coo.row
        col_indices = adj_coo.col

        # WR-03 FIX: Use sparse matrix operations instead of inefficient Python for-loop
        # This is O(n_spots) efficient for large datasets

        # Convert adjacency to sparse tensor for efficient matrix multiplication
        # For undirected graph: adj[i,j]=1 means j is neighbor of i
        # We want: spatial_bias[i] = mean of u_st[j] for all j where adj[i,j]=1
        # This is: spatial_bias = (adj @ u_st) / degree

        # Create sparse tensor from scipy CSR matrix
        adj_csr = adj.tocsr()

        # Get row-wise degree (number of neighbors per spot)
        degree = np.asarray(adj_csr.sum(axis=1)).flatten().astype(np.float32)
        degree = np.maximum(degree, 1e-8)  # Avoid division by zero

        # Convert sparse matrix to PyTorch sparse tensor
        adj_coo = adj.tocoo()
        indices = torch.from_numpy(np.vstack([adj_coo.row, adj_coo.col]).astype(np.int64))
        values = torch.from_numpy(adj_coo.data.astype(np.float32))
        adj_sparse = torch.sparse_coo_tensor(
            indices, values,
            size=(n_spots, n_spots),
            dtype=torch.float32,
        ).to(u_st_detached.device)

        # Normalize adjacency: divide each row by degree
        # For sparse: normalized_adj[i,j] = adj[i,j] / degree[i]
        degree_tensor = torch.from_numpy(degree).to(u_st_detached.device)
        degree_inv = 1.0 / degree_tensor

        # Compute normalized sparse adjacency (row-normalized)
        # This is equivalent to A / diag(degree) for mean aggregation
        values_normalized = values.to(u_st_detached.device) * degree_inv[adj_coo.row]

        adj_normalized = torch.sparse_coo_tensor(
            indices, values_normalized,
            size=(n_spots, n_spots),
            dtype=torch.float32,
        ).to(u_st_detached.device)

        # Efficient sparse matrix multiplication: spatial_bias = adj_normalized @ u_st
        spatial_bias = torch.sparse.mm(adj_normalized, u_st_detached)

        spatial_bias = F.leaky_relu(self.spatial_bias_proj(spatial_bias), 0.2)
        return spatial_bias

    def forward(
        self,
        u_st: Tensor,
        return_fused: bool = True,
    ) -> Tensor:
        """Forward pass through spatial scaffold.

        CRITICAL: u_st is detached before any spatial processing.
        fusion_conv output is also detached — spatial scaffold does not
        contribute gradients back to VAE (anti-pattern enforcement).
        In Phase 2, fusion_conv output will condition ST decoder.

        Parameters
        ----------
        u_st : Tensor
            (n_spots, latent_dim) — ST latent from VAE encoder.
        return_fused : bool, default True
            If True, return spatial-conditioned latent.
            If False, return just fusion_conv output (no spatial bias).

        Returns
        -------
        Tensor
            Fused spatial-aware latent representation (detached, no grad).
        """
        # CRITICAL ANTI-PATTERN: detach u_st before spatial processing
        # This prevents gradients from flowing back to VAE encoder through u_st
        u_st_detached = u_st.detach()

        # fusion_conv: MLP conditioning on detached latent
        # fusion_conv itself has trainable parameters but reads only detached input
        fused = self.fusion_conv(u_st_detached)

        # Add spatial bias if graph is available
        if self.spatial_adj is not None:
            spatial_bias = self._compute_spatial_bias(u_st_detached)
            fused = fused + spatial_bias

        # Detach output: spatial scaffold does not backprop gradients to VAE
        # This ensures fused.requires_grad == False regardless of trainable params
        if return_fused:
            return fused.detach()
        return fused.detach()


def SpatialAwareLoss(
    u_st: Tensor,
    spatial_scaffold: SpatialScaffold,
    batch_indices: Optional[Tensor] = None,
    margin: float = 1.0,
) -> Tensor:
    """Spatial smoothness loss: neighboring spots should have similar latent representations.

    BF-02 FIX: Handles minibatch correctly by filtering edges to batch-local subset.

    Computes: loss = ((u_st[neighbors_i] - u_st[neighbors_j]) ** 2).mean()

    Parameters
    ----------
    u_st : Tensor
        (batch_size, latent_dim) — ST latent representations for THIS batch.
    spatial_scaffold : SpatialScaffold
        Scaffold with spatial adjacency graph set.
    batch_indices : Tensor | None, default None
        (batch_size,) — global indices of spots in this batch.
        If None, assumes u_st contains all spots (full dataset).
    margin : float, default 1.0
        Hinge margin for contrastive-like smoothness (not used in basic version).

    Returns
    -------
    Tensor
        Scalar smoothness loss (non-negative).
    """
    if spatial_scaffold.spatial_adj is None:
        return torch.tensor(0.0, device=u_st.device, requires_grad=u_st.requires_grad)

    adj = spatial_scaffold.spatial_adj
    u = u_st  # Use attached u_st for loss computation (backprop goes to VAE)

    # BF-02: Filter to batch-local edges if batch_indices provided
    if batch_indices is not None:
        batch_indices_np = batch_indices.cpu().numpy()
        batch_set = set(batch_indices_np)

        # Get all edges from adjacency
        row, col = adj.nonzero()

        # Filter to edges where BOTH endpoints are in batch
        edge_mask = np.array([
            i in batch_set and j in batch_set
            for i, j in zip(row, col)
        ])

        row_batch = row[edge_mask]
        col_batch = col[edge_mask]

        if len(row_batch) == 0:
            # No edges within this batch
            return torch.tensor(0.0, device=u.device)

        # Remap global indices to batch-local indices (0..batch_size-1)
        idx_map = {global_idx: batch_idx for batch_idx, global_idx in enumerate(batch_indices_np)}
        row_local = np.array([idx_map[i] for i in row_batch])
        col_local = np.array([idx_map[j] for j in col_batch])

        # Use batch-local indices
        diff = u[row_local] - u[col_local]
    else:
        # Full dataset case (original behavior)
        # BF-02 SAFETY: Check that u has same size as adjacency
        if u.shape[0] != adj.shape[0]:
            # u is batch embedding, but no batch_indices provided
            # This is a configuration error - return zero loss
            import logging
            logging.getLogger(__name__).warning(
                f"SpatialAwareLoss: u.shape[0]={u.shape[0]} != adj.shape[0]={adj.shape[0]}, "
                f"but batch_indices=None. Returning zero loss."
            )
            return torch.tensor(0.0, device=u.device)
        row, col = adj.nonzero()
        if len(row) == 0:
            return torch.tensor(0.0, device=u.device)
        diff = u[row] - u[col]

    # Smoothness loss = mean squared distance between neighbors
    sq_dist = (diff ** 2).sum(dim=-1)
    loss = sq_dist.mean()
    return loss


def get_spatial_neighbors_batch(
    spatial_adj: scipy.sparse.csr_matrix,
    batch_indices: Tensor,
) -> Tensor:
    """Get neighbor indices for a batch of spot indices.

    Parameters
    ----------
    spatial_adj : scipy.sparse.csr_matrix
        (n_spots, n_spots) spatial k-NN adjacency.
    batch_indices : Tensor
        (batch_size,) — indices of spots to get neighbors for.

    Returns
    -------
    Tensor
        (batch_size, n_neighbors) — indices of neighbors for each spot in batch.
    """
    device = batch_indices.device
    adj = spatial_adj.tocsc()  # CSC for efficient column slicing

    batch_indices_np = batch_indices.cpu().numpy()
    neighbor_list = []
    for idx in batch_indices_np:
        # Get neighbors for this spot (column of adjacency)
        col = adj.getcol(idx)
        nbrs = col.indices
        # Optionally pad/truncate to n_neighbors
        if len(nbrs) >= spatial_adj.shape[1]:
            nbrs = nbrs[: spatial_adj.shape[1]]
        neighbor_list.append(nbrs)

    # Pad to consistent length
    max_len = max(len(n) for n in neighbor_list)
    padded = np.array([np.pad(n, (0, max_len - len(n)), constant_values=-1) for n in neighbor_list])
    return torch.tensor(padded, dtype=torch.long, device=device)
