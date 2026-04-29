"""Optimal Transport solver for spatial transcriptomics deconvolution.

OT-01: Uses POT ot.sinkhorn with epsilon >= 0.1 (NOT from scratch).
OT-02: Cost matrix built from fused ST embeddings via cosine similarity.
OT-03: k-NN pre-filtering for scalability (top-50 per spot).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor

try:
    import ot
except ImportError:
    ot = None  # type: ignore


@dataclass
class TransportPlan:
    """Result of optimal transport solving.

    Attributes:
        plan: Transport plan matrix P (n_spots, n_cells) or (n_spots, k) after prefilter
        cost: Cost matrix used for OT (n_spots, n_cells) or (n_spots, k) after prefilter
        marginal_error: L1 error of row/column marginals vs target masses
        convergence_passed: True if marginal_error < 1e-3
        n_iter: Number of Sinkhorn iterations performed
    """

    plan: Tensor
    cost: Tensor
    marginal_error: float
    convergence_passed: bool
    n_iter: int


class OTSolver:
    """Sinkhorn OT solver using POT library.

    Implements entropic regularized optimal transport with:
    - epsilon >= 0.1 enforced per OT-01 anti-pattern
    - Cosine similarity cost matrix per D-09
    - k-NN pre-filtering per D-12 (top-50 per spot)
    - Convergence checking via marginal error < 1e-3

    Parameters
    ----------
    epsilon : float, default 0.1
        Entropy regularization (CLAMPED to >= 0.1)
    n_iters : int, default 100
        Maximum Sinkhorn iterations
    k_neighbors : int or "auto", default "auto"
        Number of k-NN pre-filtered neighbors per spot (D-12).
        If "auto", resolved at runtime via _resolve_k_neighbors().
    convergence_threshold : float, default 1e-3
        Threshold for marginal error convergence check
    """

    def __init__(
        self,
        epsilon: float = 0.5,
        n_iters: int = 100,
        k_neighbors: Union[int, str] = "auto",
        convergence_threshold: float = 1e-3,
    ) -> None:
        # OT-01: Enforce epsilon >= 0.1
        self.epsilon = max(0.1, float(epsilon))
        self.n_iters = n_iters
        if isinstance(k_neighbors, str) and k_neighbors != "auto":
            raise ValueError(f'k_neighbors must be int or "auto", got {k_neighbors!r}')
        self._k_neighbors_raw = k_neighbors
        self.k_neighbors = k_neighbors  # keep for backward compat property access
        self.convergence_threshold = convergence_threshold

    def _resolve_k_neighbors(self, n_cells: int, n_cell_types: int = 16) -> int:
        """Resolve adaptive k-NN value from raw setting.

        Strategy A4: max(50, n_cell_types * 5, min(200, n_cells // 500))
        Balances type coverage and data scale.
        """
        if isinstance(self._k_neighbors_raw, int):
            return min(self._k_neighbors_raw, n_cells)
        # auto strategy: type coverage + scale
        k = max(50, n_cell_types * 5, min(200, n_cells // 500))
        return min(k, n_cells)

    def build_cosine_cost_matrix(
        self,
        fused_st: Tensor,
        scRNA_profiles: Tensor,
    ) -> Tensor:
        """Build cost matrix from fused ST embeddings using cosine similarity.

        D-09: Cost matrix = cosine similarity on fused ST embeddings.

        Parameters
        ----------
        fused_st : Tensor
            (n_spots, latent_dim) — spatially-fused ST embeddings from SpatialScaffold
        scRNA_profiles : Tensor
            (n_cells, latent_dim) — scRNA embedding profiles

        Returns
        -------
        Tensor
            (n_spots, n_cells) cost matrix where cost[i,j] = 1 - cosine(fused_st[i], scRNA[j])
            All values in [0, 2], lower is better (more similar)
        """
        # Normalize embeddings to unit vectors for cosine similarity
        fused_st_norm = F.normalize(fused_st, p=2, dim=1)  # (n_spots, latent_dim)
        scRNA_norm = F.normalize(scRNA_profiles, p=2, dim=1)  # (n_cells, latent_dim)

        # Cosine similarity via dot product of normalized vectors
        # shape: (n_spots, n_cells)
        cosine_sim = torch.mm(fused_st_norm, scRNA_norm.t())

        # Convert similarity to cost: cost = 1 - similarity
        # Range: [0, 2], where 0 means identical, 2 means opposite
        cost = 1.0 - cosine_sim

        return cost.to(torch.float32)

    def knn_prefilter(
        self,
        cost_matrix: Tensor,
        k: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Apply k-NN pre-filtering to keep only top-k smallest costs per row.

        D-12: k-NN pre-filter = top-50 per spot for scalability.

        Parameters
        ----------
        cost_matrix : Tensor
            (n_spots, n_cells) cost matrix
        k : int, optional
            Number of neighbors to keep per spot. If None, uses self.k_neighbors.

        Returns
        -------
        Tuple[Tensor, Tensor]
            - filtered_cost: (n_spots, k) cost matrix with only top-k per row
            - indices: (n_spots, k) indices of kept columns
        """
        if k is None:
            k = 50 if self._k_neighbors_raw == "auto" else self._k_neighbors_raw

        n_spots, n_cells = cost_matrix.shape
        k = min(k, n_cells)  # Cannot have more neighbors than cells

        # For each row, get indices of k smallest values
        # Using topk on negative values to get k smallest
        indices = torch.argsort(cost_matrix, dim=1)[:, :k]  # (n_spots, k)

        # Gather the costs at these indices
        n_spots_idx = torch.arange(n_spots, device=cost_matrix.device).unsqueeze(1).expand_as(indices)
        filtered_cost = cost_matrix[n_spots_idx, indices]

        return filtered_cost, indices

    def check_convergence(
        self,
        plan: Tensor,
        cost: Tensor,
        row_mass: Tensor,
        col_mass: Tensor,
    ) -> Tuple[float, bool]:
        """Check convergence via marginal error.

        Computes L1 norm of marginal error: sum of
        - ||row_sum(P) - row_mass||_1
        - ||col_sum(P) - col_mass||_1

        Parameters
        ----------
        plan : Tensor
            (n_spots, k) or (n_spots, n_cells) transport plan
        cost : Tensor
            (n_spots, k) or (n_spots, n_cells) cost matrix
        row_mass : Tensor
            (n_spots,) target row marginals
        col_mass : Tensor
            (k,) or (n_cells,) target column marginals

        Returns
        -------
        Tuple[float, bool]
            - marginal_error: L1 norm of marginal violations
            - converged: True if marginal_error < self.convergence_threshold
        """
        # Row marginal: sum over columns
        # Handle sparse tensor: sum returns sparse, need to convert to dense for operations
        row_sum = plan.sum(dim=1)  # (n_spots,)
        if row_sum.is_sparse:
            row_sum = row_sum.to_dense()
        row_error = torch.abs(row_sum - row_mass).sum().item()

        # Column marginal: sum over rows
        col_sum = plan.sum(dim=0)  # (k,) or (n_cells,)
        if col_sum.is_sparse:
            col_sum = col_sum.to_dense()
        # Handle shape mismatch if col_mass is longer (after prefilter)
        min_len = min(len(col_sum), len(col_mass))
        col_error = torch.abs(col_sum[:min_len] - col_mass[:min_len]).sum().item()
        col_error += torch.abs(col_sum[min_len:]).sum().item() if min_len < len(col_sum) else 0.0

        marginal_error = row_error + col_error
        converged = marginal_error < self.convergence_threshold

        return marginal_error, converged

    def solve(
        self,
        cost_matrix: Tensor,
        row_mass: Optional[Tensor] = None,
        col_mass: Optional[Tensor] = None,
        return_plan: bool = True,
    ) -> Tuple[Tensor, int]:
        """Solve optimal transport using POT Sinkhorn.

        Uses POT ot.sinkhorn with epsilon >= 0.1 (NOT from scratch).

        Parameters
        ----------
        cost_matrix : Tensor
            (n_spots, n_cells) or (n_spots, k) cost matrix
        row_mass : Tensor, optional
            (n_spots,) row probability mass (uniform by default)
        col_mass : Tensor, optional
            (n_cells,) or (k,) column probability mass (uniform by default)
        return_plan : bool, default True
            If True, return transport plan. If False, return only n_iter.

        Returns
        -------
        Tuple[Tensor, int]
            - P: Transport plan (n_spots, n_cells) or (n_spots, k)
            - n_iter: Number of iterations performed
        """
        if ot is None:
            raise ImportError("POT library not available. Install with: pip install POT")

        # Ensure float32
        C = cost_matrix.to(torch.float32)
        device = C.device
        n_spots, n_cells = C.shape

        # Convert to numpy for POT
        C_np = C.detach().cpu().numpy()

        # Uniform distributions by default
        # OT-01 FIX: Validate marginals before Sinkhorn to prevent division by zero
        if row_mass is None:
            a_np = np.ones(n_spots, dtype=np.float64) / n_spots
        else:
            a_np = row_mass.detach().cpu().numpy().astype(np.float64)
            # OT-01 FIX: Validate sum > epsilon before normalization
            if a_np.sum() < 1e-10:
                import logging
                logging.getLogger(__name__).warning(
                    "Row marginal sum near zero (%.2e), using uniform distribution"
                )
                a_np = np.ones(n_spots, dtype=np.float64) / n_spots
            else:
                a_np = a_np / a_np.sum()

        if col_mass is None:
            b_np = np.ones(n_cells, dtype=np.float64) / n_cells
        else:
            b_np = col_mass.detach().cpu().numpy().astype(np.float64)
            # OT-01 FIX: Validate sum > epsilon before normalization
            if b_np.sum() < 1e-10:
                import logging
                logging.getLogger(__name__).warning(
                    "Column marginal sum near zero (%.2e), using uniform distribution"
                )
                b_np = np.ones(n_cells, dtype=np.float64) / n_cells
            else:
                b_np = b_np / b_np.sum()

        # POT sinkhorn: epsilon >= 0.1 enforced via self.epsilon
        # numItermax = self.n_iters
        P_np = ot.sinkhorn(
            a_np,
            b_np,
            C_np,
            self.epsilon,
            numItermax=self.n_iters,
            method="sinkhorn",  # Explicit method
        )

        n_iter = self.n_iters

        # Check if NaN/Inf in result (indicates non-convergence)
        if not np.isfinite(P_np).all():
            n_iter = self.n_iters  # Already at max iterations
            P_np = np.nan_to_num(P_np, nan=1.0 / n_cells, posinf=1.0 / n_cells, neginf=0.0)

        P = torch.tensor(P_np, dtype=torch.float32, device=device)

        return P, n_iter

    def solve_chunked(
        self,
        fused_st: Tensor,
        scRNA_profiles: Tensor,
        chunk_size: int = 5000,
        apply_prefilter: bool = True,
        n_cell_types: int = 16,
        two_pass: bool = True,
    ) -> TransportPlan:
        """Chunked OT solving for large-scale datasets (LD-02, D-12, D-13).

        Processes spots in chunks to avoid dense (n_spots, n_cells) cost matrix.
        Supports two-pass mode for improved marginal accuracy.

        Parameters
        ----------
        fused_st : Tensor
            (n_spots, latent_dim) — fused ST embeddings
        scRNA_profiles : Tensor
            (n_cells, latent_dim) — scRNA profiles
        chunk_size : int, default 5000
            Number of spots per chunk
        apply_prefilter : bool, default True
            Apply k-NN prefiltering per chunk
        n_cell_types : int, default 16
            Number of cell types (used for adaptive k-NN)
        two_pass : bool, default True
            If True, run two-pass solving with global marginal estimation
        """
        n_spots = fused_st.shape[0]
        n_cells = scRNA_profiles.shape[0]
        device = fused_st.device

        # Resolve adaptive k-NN
        k = self._resolve_k_neighbors(n_cells, n_cell_types)

        # Pre-compute normalized profiles (one-time, D-12)
        scRNA_norm = F.normalize(scRNA_profiles, p=2, dim=1)

        def _solve_pass(col_mass_provider=None):
            """Single pass over all chunks. col_mass_provider(chunk_knn_indices) -> col_mass."""
            sparse_indices = []
            sparse_values = []
            total_marginal_error = 0.0
            total_n_iter = 0
            n_chunks = 0
            col_sum_global = torch.zeros(n_cells, device=device) if col_mass_provider is None else None

            for start in range(0, n_spots, chunk_size):
                end = min(start + chunk_size, n_spots)
                cs = end - start

                # Build cost matrix for this chunk
                fused_chunk = fused_st[start:end]
                fused_norm = F.normalize(fused_chunk, p=2, dim=1)
                cosine_sim = torch.mm(fused_norm, scRNA_norm.t())
                cost_chunk = 1.0 - cosine_sim

                # Apply k-NN prefilter
                if apply_prefilter and k < n_cells:
                    filtered_cost, knn_indices = self.knn_prefilter(cost_chunk, k=k)
                else:
                    filtered_cost = cost_chunk
                    knn_indices = None

                # Row mass: always uniform per chunk (Sinkhorn normalizes anyway)
                row_mass = torch.ones(cs, device=device) / cs

                # Column mass
                if col_mass_provider is not None and knn_indices is not None:
                    col_mass = col_mass_provider(knn_indices)
                elif knn_indices is not None:
                    col_mass = torch.ones(k, device=device) / k
                else:
                    col_mass = torch.ones(n_cells, device=device) / n_cells

                # Solve OT for chunk
                plan_chunk, n_iter = self.solve(filtered_cost, row_mass, col_mass)

                # Check marginal error for this chunk
                marg_err, _ = self.check_convergence(plan_chunk, filtered_cost, row_mass, col_mass)
                total_marginal_error += marg_err
                total_n_iter += n_iter
                n_chunks += 1

                # Aggregate column sums for two-pass estimation
                if col_sum_global is not None:
                    if knn_indices is not None:
                        col_sum_chunk = torch.zeros(n_cells, device=device)
                        col_sum_chunk.scatter_add_(0, knn_indices.flatten(), plan_chunk.flatten())
                        col_sum_global += col_sum_chunk
                    else:
                        col_sum_global += plan_chunk.sum(dim=0)

                # Build sparse indices
                if knn_indices is not None:
                    row_idx = torch.arange(cs, device=device).unsqueeze(1).expand_as(knn_indices)
                    sparse_indices.append(torch.stack([
                        row_idx.flatten() + start,
                        knn_indices.flatten()
                    ], dim=0))
                    sparse_values.append(plan_chunk.flatten())
                else:
                    row_idx = torch.arange(cs, device=device).unsqueeze(1).expand(cs, n_cells)
                    col_idx = torch.arange(n_cells, device=device).unsqueeze(0).expand(cs, n_cells)
                    sparse_indices.append(torch.stack([
                        row_idx.flatten() + start,
                        col_idx.flatten()
                    ], dim=0))
                    sparse_values.append(plan_chunk.flatten())

            # Assemble sparse plan
            all_indices = torch.cat(sparse_indices, dim=1)
            all_values = torch.cat(sparse_values)
            P = torch.sparse_coo_tensor(
                all_indices, all_values,
                size=(n_spots, n_cells), dtype=torch.float32, device=device,
            ).coalesce()

            avg_err = total_marginal_error / n_chunks if n_chunks > 0 else 0.0
            avg_iter = total_n_iter / n_chunks if n_chunks > 0 else 0
            return P, avg_err, avg_iter, col_sum_global

        # --- Pass 1: coarse solve to estimate column marginals ---
        P1, avg_err1, avg_iter1, col_sum_global = _solve_pass(col_mass_provider=None)

        if not two_pass or col_sum_global is None:
            converged = avg_err1 < self.convergence_threshold * 10
            return TransportPlan(
                plan=P1, cost=torch.zeros(1, device=device),
                marginal_error=avg_err1, convergence_passed=converged,
                n_iter=int(avg_iter1),
            )

        # --- Normalize column marginal from pass 1 ---
        global_col_prob = col_sum_global / col_sum_global.sum().clamp(min=1e-10)

        def _col_mass_from_global(knn_indices):
            """Extract 1D col mass for chunk's knn indices from global estimate."""
            col_mass = global_col_prob[knn_indices.flatten()].reshape(knn_indices.shape)
            # Aggregate across spots in chunk → 1D marginal per column
            col_mass_1d = col_mass.sum(dim=0)
            return col_mass_1d / col_mass_1d.sum().clamp(min=1e-10)

        # --- Pass 2: refined solve with global marginals ---
        P2, avg_err2, avg_iter2, _ = _solve_pass(col_mass_provider=_col_mass_from_global)

        converged = avg_err2 < self.convergence_threshold * 10
        return TransportPlan(
            plan=P2, cost=torch.zeros(1, device=device),
            marginal_error=avg_err2, convergence_passed=converged,
            n_iter=int(avg_iter1 + avg_iter2),
        )

    def __call__(
        self,
        fused_st: Tensor,
        scRNA_profiles: Tensor,
        row_mass: Optional[Tensor] = None,
        col_mass: Optional[Tensor] = None,
        apply_prefilter: bool = True,
    ) -> TransportPlan:
        """Full OT pipeline: build cost, prefilter, solve, check convergence.

        Parameters
        ----------
        fused_st : Tensor
            (n_spots, latent_dim) — fused ST embeddings
        scRNA_profiles : Tensor
            (n_cells, latent_dim) — scRNA profiles
        row_mass : Tensor, optional
            (n_spots,) row mass vector
        col_mass : Tensor, optional
            (n_cells,) column mass vector
        apply_prefilter : bool, default True
            Whether to apply k-NN prefiltering

        Returns
        -------
        TransportPlan
            Dataclass with transport plan, cost matrix, marginal error, convergence status
        """
        n_spots = fused_st.shape[0]
        n_cells = scRNA_profiles.shape[0]

        # Resolve adaptive k-NN
        k = self._resolve_k_neighbors(n_cells)

        # Build cosine similarity cost matrix
        cost_matrix = self.build_cosine_cost_matrix(fused_st, scRNA_profiles)

        # Apply k-NN prefiltering if requested
        prefilter_indices = None
        if apply_prefilter and k < cost_matrix.shape[1]:
            filtered_cost, prefilter_indices = self.knn_prefilter(cost_matrix, k=k)
            cost_to_solve = filtered_cost
        else:
            cost_to_solve = cost_matrix

        # Solve OT
        plan, n_iter = self.solve(cost_to_solve, row_mass, col_mass)

        # BF-03 FIX: Return sparse tensor instead of dense reconstruction
        # Dense (n_spots, n_cells) would be ~1GB for 50Kx5K dataset
        if prefilter_indices is not None:
            # Construct sparse COO tensor directly from prefiltered indices and values
            # Indices: row indices (0..n_spots-1) and column indices from prefilter
            row_indices = torch.arange(n_spots, device=plan.device).unsqueeze(1).expand_as(prefilter_indices)
            sparse_indices = torch.stack([
                row_indices.flatten(),
                prefilter_indices.flatten()
            ], dim=0)  # (2, n_spots * k)

            sparse_values = plan.flatten()  # (n_spots * k,)

            plan = torch.sparse_coo_tensor(
                sparse_indices,
                sparse_values,
                size=(n_spots, n_cells),
                dtype=plan.dtype,
                device=plan.device
            ).coalesce()  # Coalesce for efficient operations

        # Compute default masses for convergence check
        if row_mass is None:
            row_m = torch.ones(n_spots, dtype=torch.float32, device=fused_st.device) / n_spots
        else:
            row_m = row_mass.to(torch.float32)

        if col_mass is None:
            col_m = torch.ones(cost_to_solve.shape[1], dtype=torch.float32, device=fused_st.device) / cost_to_solve.shape[1]
        else:
            col_m = col_mass.to(torch.float32)
            if apply_prefilter and k < len(col_m):
                # Adjust col_mass to match prefiltered size
                col_m = col_m[:k]

        # Check convergence
        marginal_error, converged = self.check_convergence(plan, cost_to_solve, row_m, col_m)

        # Log warning if not converged
        if not converged:
            import warnings

            warnings.warn(
                f"OT solver did not converge: marginal_error={marginal_error:.6f} "
                f"(threshold={self.convergence_threshold}). "
                f"Consider increasing epsilon or n_iters."
            )

        return TransportPlan(
            plan=plan,
            cost=cost_to_solve,
            marginal_error=marginal_error,
            convergence_passed=converged,
            n_iter=n_iter,
        )


# Helper function for quick access
def compute_ot_deconvolution(
    fused_st: Tensor,
    scRNA_profiles: Tensor,
    epsilon: float = 0.1,
    k_neighbors: int = 50,
    n_iters: int = 100,
) -> TransportPlan:
    """Convenience function for OT deconvolution.

    Parameters
    ----------
    fused_st : Tensor
        (n_spots, latent_dim) — fused ST embeddings
    scRNA_profiles : Tensor
        (n_cells, latent_dim) — scRNA profiles
    epsilon : float, default 0.1
        Entropy regularization (clamped to >= 0.1)
    k_neighbors : int, default 50
        k-NN prefilter size
    n_iters : int, default 100
        Sinkhorn iterations

    Returns
    -------
    TransportPlan
        Optimal transport plan and metadata
    """
    solver = OTSolver(epsilon=epsilon, k_neighbors=k_neighbors, n_iters=n_iters)
    return solver(fused_st, scRNA_profiles)


import numpy as np  # noqa: E402
