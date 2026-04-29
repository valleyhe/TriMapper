"""Condition-level prior computation for bulk data integration.

Implements the full pipeline:
1. Encode all scRNA and ST embeddings
2. Run OT deconvolution to get spot-level cell type proportions
3. Aggregate by condition to get pred_condition_proportions
4. Compare with bulk_condition_proportions via KL

This module is designed to be called at epoch start, with cached results
used throughout the epoch training steps.

Chunk 2: Added condition alignment by name (not row position).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
import numpy as np

logger = logging.getLogger(__name__)


def align_condition_tensors(
    pred_props: Tensor,
    pred_names: List[str],
    bulk_props: Tensor,
    bulk_names: List[str],
) -> Tuple[Tensor, Tensor, List[str]]:
    """Align predicted and bulk condition tensors by condition name.

    Chunk 2: Aligns by condition NAMES, not row position.
    This handles:
    - Different ordering of conditions
    - Partial overlap (some conditions missing from either side)
    - No overlap (returns empty tensors with valid=False)

    Parameters
    ----------
    pred_props : Tensor
        (n_pred_conditions, n_cell_types) predicted condition proportions
    pred_names : List[str]
        Condition names for pred_props rows
    bulk_props : Tensor
        (n_bulk_conditions, n_cell_types) bulk-derived condition proportions
    bulk_names : List[str]
        Condition names for bulk_props rows

    Returns
    -------
    Tuple[Tensor, Tensor, List[str]]
        - aligned_pred: (n_common,) subset of pred_props aligned with bulk
        - aligned_bulk: (n_common,) subset of bulk_props aligned with pred
        - common_names: List of condition names that appear in both
    """
    # Find intersection
    pred_set = set(pred_names)
    bulk_set = set(bulk_names)
    common = pred_set.intersection(bulk_set)

    if len(common) == 0:
        # No overlap: return empty tensors
        logger.warning(
            f"No common conditions between pred ({pred_names}) and bulk ({bulk_names})"
        )
        return (
            torch.zeros(0, pred_props.shape[1], dtype=pred_props.dtype),
            torch.zeros(0, bulk_props.shape[1], dtype=bulk_props.dtype),
            [],
        )

    # Sort common conditions for consistent ordering
    # Use alphabetical order for reproducibility
    common_names = sorted(common)

    # Build aligned tensors
    aligned_pred = torch.zeros(len(common_names), pred_props.shape[1], dtype=pred_props.dtype)
    aligned_bulk = torch.zeros(len(common_names), bulk_props.shape[1], dtype=bulk_props.dtype)

    for i, name in enumerate(common_names):
        pred_idx = pred_names.index(name)
        bulk_idx = bulk_names.index(name)
        aligned_pred[i] = pred_props[pred_idx]
        aligned_bulk[i] = bulk_props[bulk_idx]

    logger.info(
        f"Aligned conditions: {common_names} "
        f"(pred has {len(pred_names)}, bulk has {len(bulk_names)}, common={len(common_names)}"
    )

    return aligned_pred, aligned_bulk, common_names


@dataclass
class ConditionPriorState:
    """Cached state for condition-level bulk prior computation.

    Chunk 4: Added explicit bulk_condition_names and pred_condition_names
    to avoid positional fallback when row counts match.

    Attributes:
        spot_type_proportions: (n_st, n_cell_types) spot-level proportions from OT deconv
        pred_condition_proportions: (n_conditions, n_cell_types) aggregated by ST condition
        bulk_condition_proportions: (n_conditions, n_cell_types) from bulk preprocessing
        pred_condition_names: List of predicted condition names (from ST aggregation)
        bulk_condition_names: List of bulk condition names (from trainer input)
        condition_names: Legacy field, now pred_condition_names (kept for backward compat)
        cell_type_names: List of canonical cell type names in order
        ot_valid: Whether OT deconvolution succeeded this epoch
        epoch: Current epoch number
    """

    spot_type_proportions: Optional[Tensor] = None
    pred_condition_proportions: Optional[Tensor] = None
    bulk_condition_proportions: Optional[Tensor] = None
    pred_condition_names: Optional[List[str]] = None  # Chunk 4: explicit predicted names
    bulk_condition_names: Optional[List[str]] = None  # Chunk 4: explicit bulk names
    condition_names: Optional[List[str]] = None  # Legacy: pred names (backward compat)
    cell_type_names: Optional[List[str]] = None
    ot_valid: bool = False
    epoch: int = 0


def encode_full_embeddings(
    vae,
    X_sc: Tensor,
    X_st: Tensor,
    device: str,
    batch_size: int = 512,
) -> Tuple[Tensor, Tensor]:
    """Encode full scRNA and ST datasets to latent embeddings.

    Parameters
    ----------
    vae : TripleModalVAE
        VAE model with enc_sc and enc_st encoders
    X_sc : Tensor
        (n_cells, n_genes) scRNA expression
    X_st : Tensor
        (n_spots, n_genes) ST expression
    device : str
        Device to compute on
    batch_size : int
        Batch size for encoding (memory optimization)

    Returns
    -------
    Tuple[Tensor, Tensor]
        - u_sc: (n_cells, latent_dim) scRNA embeddings
        - u_st: (n_spots, latent_dim) ST embeddings
    """
    vae.eval()

    # Encode scRNA in batches
    n_sc = X_sc.shape[0]
    u_sc_all = []
    with torch.no_grad():
        for i in range(0, n_sc, batch_size):
            batch = X_sc[i:i+batch_size].to(device)
            z, mean, log_var = vae.enc_sc(batch)
            u_sc_all.append(mean.cpu())  # Move to CPU to save GPU memory
    u_sc = torch.cat(u_sc_all, dim=0)

    # Encode ST in batches
    n_st = X_st.shape[0]
    u_st_all = []
    with torch.no_grad():
        for i in range(0, n_st, batch_size):
            batch = X_st[i:i+batch_size].to(device)
            z, mean, log_var = vae.enc_st(batch)
            u_st_all.append(mean.cpu())
    u_st = torch.cat(u_st_all, dim=0)

    return u_sc, u_st


def encode_full_embeddings_from_iterators(
    vae,
    sc_iter,
    st_iter,
    device: str,
) -> Tuple[Tensor, Tensor]:
    """Encode full scRNA and ST datasets from chunked iterators.

    Parameters
    ----------
    vae : TripleModalVAE
        VAE model with enc_sc and enc_st encoders
    sc_iter : Iterator[Tensor]
        Iterator yielding scRNA expression chunks
    st_iter : Iterator[Tensor]
        Iterator yielding ST expression chunks
    device : str
        Device to compute on

    Returns
    -------
    Tuple[Tensor, Tensor]
        - u_sc: (n_cells, latent_dim) scRNA embeddings
        - u_st: (n_spots, latent_dim) ST embeddings
    """
    vae.eval()

    u_sc_all = []
    with torch.no_grad():
        for batch in sc_iter:
            batch = batch.to(device)
            z, mean, log_var = vae.enc_sc(batch)
            u_sc_all.append(mean.cpu())
    u_sc = torch.cat(u_sc_all, dim=0) if u_sc_all else torch.empty(0)

    u_st_all = []
    with torch.no_grad():
        for batch in st_iter:
            batch = batch.to(device)
            z, mean, log_var = vae.enc_st(batch)
            u_st_all.append(mean.cpu())
    u_st = torch.cat(u_st_all, dim=0) if u_st_all else torch.empty(0)

    return u_sc, u_st


def run_ot_deconvolution(
    u_st: Tensor,
    u_sc: Tensor,
    epsilon: float = 0.5,
    k_neighbors: int = 50,
    chunk_size: int = 5000,
    subsample_sc: int = 10000,
    subsample_st: int = 10000,
) -> Tuple[Tensor, bool]:
    """Run OT deconvolution to get spot-level cell assignments.

    Uses random subsampling of scRNA and ST embeddings for tractable OT.
    This makes OT feasible on CPU by reducing matrix size from
    (n_st × n_sc) to (subsample_st × subsample_sc).

    Parameters
    ----------
    u_st : Tensor
        (n_spots, latent_dim) ST embeddings
    u_sc : Tensor
        (n_cells, latent_dim) scRNA embeddings
    epsilon : float
        OT entropy regularization (>= 0.5 for stability)
    k_neighbors : int
        k-NN prefilter size for scalability
    chunk_size : int
        Chunk size for OT solver
    subsample_sc : int
        Number of scRNA cells to subsample (default 10K). None = no subsample.
    subsample_st : int
        Number of ST spots to subsample (default 10K). None = no subsample.

    Returns
    -------
    Tuple[Tensor, bool]
        - transport_plan: Sparse tensor (n_st, n_cells) with subsampled OT
        - valid: Whether OT succeeded (not NaN/Inf)
    """
    from ..deconv.ot_solver import OTSolver

    n_st, n_sc = u_st.shape[0], u_sc.shape[0]

    # Subsample for tractable OT
    if subsample_st is not None and n_st > subsample_st:
        st_idx = torch.randperm(n_st)[:subsample_st]
        u_st_sub = u_st[st_idx]
        logger.info(f"OT subsample ST: {n_st} → {subsample_st}")
    else:
        st_idx = None
        u_st_sub = u_st

    if subsample_sc is not None and n_sc > subsample_sc:
        sc_idx = torch.randperm(n_sc)[:subsample_sc]
        u_sc_sub = u_sc[sc_idx]
        logger.info(f"OT subsample scRNA: {n_sc} → {subsample_sc}")
    else:
        sc_idx = None
        u_sc_sub = u_sc

    solver = OTSolver(epsilon=epsilon, k_neighbors=k_neighbors)

    try:
        # Use chunked solver for scalability
        result = solver.solve_chunked(
            u_st_sub, u_sc_sub, chunk_size=chunk_size, apply_prefilter=True
        )

        # Check validity
        plan_values = result.plan._values() if result.plan.is_sparse else result.plan
        if not torch.isfinite(plan_values).all():
            logger.warning("OT transport plan contains NaN/Inf")
            return None, False

        # If subsampled, expand transport plan to full size
        if st_idx is not None or sc_idx is not None:
            # Create full-size sparse tensor
            # Get indices and values from subsampled plan
            if result.plan.is_sparse:
                sub_indices = result.plan.indices()  # (2, nnz)
                sub_values = result.plan.values()    # (nnz,)
                
                # Map subsampled indices back to original
                if st_idx is not None:
                    full_row_idx = st_idx[sub_indices[0]]
                else:
                    full_row_idx = sub_indices[0]
                
                if sc_idx is not None:
                    full_col_idx = sc_idx[sub_indices[1]]
                else:
                    full_col_idx = sub_indices[1]
                
                # Create full-size sparse tensor
                full_indices = torch.stack([full_row_idx, full_col_idx], dim=0)
                full_plan = torch.sparse_coo_tensor(
                    full_indices, sub_values,
                    size=(n_st, n_sc), dtype=torch.float32
                ).coalesce()
            else:
                # Dense plan from subsampled → expand to sparse full
                full_plan = torch.zeros(n_st, n_sc, dtype=torch.float32)
                if st_idx is not None and sc_idx is not None:
                    for i, si in enumerate(st_idx.tolist()):
                        for j, sj in enumerate(sc_idx.tolist()):
                            full_plan[si, sj] = result.plan[i, j]
                elif st_idx is not None:
                    for i, si in enumerate(st_idx.tolist()):
                        full_plan[si, :] = result.plan[i, :]
                elif sc_idx is not None:
                    for j, sj in enumerate(sc_idx.tolist()):
                        full_plan[:, sj] = result.plan[:, j]
                full_plan = full_plan.to_sparse()
            
            return full_plan, result.convergence_passed
        else:
            return result.plan, result.convergence_passed

    except (RuntimeError, torch._C._LinAlgError) as e:
        # SAFE-01 FIX: Specific exception for OT numerical failures
        # RuntimeError: PyTorch runtime errors
        # LinAlgError: linear algebra failures in sinkhorn
        logger.warning(f"OT deconvolution failed due to numerical issue: {e}")
        return None, False


def transport_to_spot_proportions(
    transport_plan: Tensor,
    cell_type_onehot: Tensor,
    sparse: bool = True,
) -> Tensor:
    """Convert transport plan to spot-level cell type proportions.

    Parameters
    ----------
    transport_plan : Tensor
        (n_spots, n_cells) transport plan, may be sparse
    cell_type_onehot : Tensor
        (n_cells, n_cell_types) one-hot cell type matrix from scRNA
    sparse : bool
        Whether transport_plan is sparse COO tensor

    Returns
    -------
    Tensor
        (n_spots, n_cell_types) spot-level cell type proportions
    """
    if sparse and transport_plan.is_sparse:
        # Sparse matrix multiplication
        # P @ cell_type_onehot where P is sparse COO
        # Use torch.sparse.mm for sparse @ dense multiplication
        spot_props = torch.sparse.mm(transport_plan, cell_type_onehot)
        # Convert sparse result to dense for normalization
        spot_props = spot_props.to_dense()
    else:
        # Dense multiplication
        spot_props = transport_plan @ cell_type_onehot

    # Normalize to simplex (rows sum to 1)
    row_sums = spot_props.sum(dim=1, keepdim=True)
    row_sums = torch.clamp(row_sums, min=1e-8)
    spot_props = spot_props / row_sums

    return spot_props


def aggregate_spot_proportions_by_condition(
    spot_proportions: Tensor,
    st_conditions: np.ndarray,
    condition_names: Optional[List[str]] = None,
) -> Tuple[Tensor, List[str]]:
    """Aggregate spot-level proportions to condition-level.

    Parameters
    ----------
    spot_proportions : Tensor
        (n_spots, n_cell_types) spot-level proportions
    st_conditions : np.ndarray
        (n_spots,) array of canonical condition labels
    condition_names : List[str], optional
        Canonical condition ordering

    Returns
    -------
    Tuple[Tensor, List[str]]
        - condition_proportions: (n_conditions, n_cell_types)
        - condition_names: List of condition names in order
    """
    from ..deconv.label_mapping import get_canonical_conditions, get_canonical_cell_types

    conditions = get_canonical_conditions()

    # Get unique conditions and ensure canonical ordering
    unique_conditions = list(set(str(c) for c in st_conditions))

    if condition_names is None:
        # Use canonical ordering: only include conditions that exist
        condition_names = [c for c in conditions.names if c in unique_conditions]
        # Add any extra conditions not in canonical (should not happen)
        for c in unique_conditions:
            if c not in condition_names:
                condition_names.append(c)

    n_conditions = len(condition_names)
    n_cell_types = spot_proportions.shape[1]

    condition_proportions = torch.zeros(n_conditions, n_cell_types)

    for i, cond in enumerate(condition_names):
        mask = np.where(st_conditions == cond)[0]
        if len(mask) > 0:
            condition_proportions[i] = spot_proportions[mask].mean(dim=0)
        else:
            condition_proportions[i] = torch.ones(n_cell_types) / n_cell_types

    # Normalize to simplex
    row_sums = condition_proportions.sum(dim=1, keepdim=True)
    condition_proportions = condition_proportions / (row_sums + 1e-8)

    return condition_proportions, condition_names


def compute_epoch_condition_prior(
    vae,
    X_sc: Tensor,
    X_st: Tensor,
    cell_type_labels: np.ndarray,
    st_conditions: np.ndarray,
    bulk_condition_proportions: Optional[Tensor],
    bulk_condition_names: Optional[List[str]],
    device: str,
    epsilon: float = 0.5,
    epoch: int = 0,
    fallback_state: Optional[ConditionPriorState] = None,
    subsample_sc: int = 10000,
    subsample_st: int = 10000,
) -> ConditionPriorState:
    """Full epoch-level condition prior computation.

    This is the main entry point called at epoch start.

    Parameters
    ----------
    vae : TripleModalVAE
        VAE model
    X_sc : Tensor
        scRNA expression data
    X_st : Tensor
        ST expression data
    cell_type_labels : np.ndarray
        (n_cells,) scRNA cell type labels (canonical names)
    st_conditions : np.ndarray
        (n_spots,) ST condition labels (canonical names)
    bulk_condition_proportions : Tensor, optional
        (n_conditions, n_cell_types) from bulk preprocessing
    bulk_condition_names : List[str], optional
        Condition names for bulk proportions
    device : str
        Device for computation
    epsilon : float
        OT entropy regularization
    epoch : int
        Current epoch number
    fallback_state : ConditionPriorState, optional
        Previous epoch state to use if OT fails
    subsample_sc : int
        Number of scRNA cells to subsample for OT (default 10K)
    subsample_st : int
        Number of ST spots to subsample for OT (default 10K)

    Returns
    -------
    ConditionPriorState
        Cached state for this epoch's bulk prior computation
    """
    from ..deconv.label_mapping import get_canonical_cell_types

    cell_types = get_canonical_cell_types()

    # 1. Encode all embeddings
    logger.info(f"Epoch {epoch}: Encoding full embeddings for condition prior...")
    u_sc, u_st = encode_full_embeddings(vae, X_sc, X_st, device)

    # 2. Build cell type one-hot matrix from scRNA labels
    cell_type_onehot_np = cell_types.to_onehot(cell_type_labels)
    cell_type_onehot = torch.tensor(cell_type_onehot_np, dtype=torch.float32)

    # 3. Run OT deconvolution with subsampling
    logger.info(f"Epoch {epoch}: Running OT deconvolution (subsample: sc={subsample_sc}, st={subsample_st})...")
    transport_plan, ot_valid = run_ot_deconvolution(
        u_st, u_sc, epsilon=epsilon,
        subsample_sc=subsample_sc, subsample_st=subsample_st,
    )

    if not ot_valid or transport_plan is None:
        logger.warning(f"Epoch {epoch}: OT deconvolution failed, using fallback")

        if fallback_state is not None and fallback_state.ot_valid:
            logger.info(f"Epoch {epoch}: Reusing previous epoch's condition prior")
            return fallback_state

        # No valid fallback: use uniform proportions
        n_st = X_st.shape[0]
        n_cell_types = cell_types.n_types
        spot_proportions = torch.ones(n_st, n_cell_types) / n_cell_types
        ot_valid = False
    else:
        # 4. Convert transport to spot proportions
        logger.info(f"Epoch {epoch}: Computing spot-level proportions...")
        spot_proportions = transport_to_spot_proportions(
            transport_plan, cell_type_onehot, sparse=transport_plan.is_sparse
        )
        ot_valid = True

    # 5. Aggregate to condition level
    logger.info(f"Epoch {epoch}: Aggregating to condition proportions...")
    pred_condition_proportions, pred_condition_names = aggregate_spot_proportions_by_condition(
        spot_proportions, st_conditions
    )

    # 6. Build state with explicit pred and bulk condition names (Chunk 4)
    state = ConditionPriorState(
        spot_type_proportions=spot_proportions,
        pred_condition_proportions=pred_condition_proportions,
        bulk_condition_proportions=bulk_condition_proportions,
        pred_condition_names=pred_condition_names,  # Chunk 4: from ST aggregation
        bulk_condition_names=bulk_condition_names,   # Chunk 4: from trainer input
        condition_names=pred_condition_names,        # Legacy: keep for backward compat
        cell_type_names=cell_types.names,
        ot_valid=ot_valid,
        epoch=epoch,
    )

    logger.info(
        f"Epoch {epoch}: Condition prior ready. "
        f"Pred conditions: {pred_condition_names}, "
        f"Bulk conditions: {bulk_condition_names}, "
        f"OT valid: {ot_valid}"
    )

    return state


def compute_epoch_condition_prior_streaming(
    vae,
    dataset,
    cell_type_labels: np.ndarray,
    st_conditions: np.ndarray,
    bulk_condition_proportions: Optional[Tensor],
    bulk_condition_names: Optional[List[str]],
    device: str,
    epsilon: float = 0.5,
    epoch: int = 0,
    fallback_state: Optional[ConditionPriorState] = None,
    chunk_size: int = 512,
    subsample_sc: int = 10000,
    subsample_st: int = 10000,
    st_indices_for_ot: Optional[np.ndarray] = None,
) -> ConditionPriorState:
    """Streaming epoch-level condition prior computation using chunked iterators.

    Parameters
    ----------
    vae : TripleModalVAE
        VAE model
    dataset
        Dataset with iter_expression_chunks() helper
    cell_type_labels : np.ndarray
        (n_cells,) scRNA cell type labels (canonical names)
    st_conditions : np.ndarray
        (n_spots,) ST condition labels (canonical names)
    bulk_condition_proportions : Tensor, optional
        (n_conditions, n_cell_types) from bulk preprocessing
    bulk_condition_names : List[str], optional
        Condition names for bulk proportions
    device : str
        Device for computation
    epsilon : float
        OT entropy regularization
    epoch : int
        Current epoch number
    fallback_state : ConditionPriorState, optional
        Previous epoch state to use if OT fails
    chunk_size : int
        Chunk size for expression loading
    subsample_sc : int
        Number of scRNA cells to subsample for OT (default 10K)
    subsample_st : int
        Number of ST spots to subsample for OT (default 10K)
    st_indices_for_ot : np.ndarray, optional
        ST spot indices to use for OT. If None, uses dataset.st_indices.

    Returns
    -------
    ConditionPriorState
        Cached state for this epoch's bulk prior computation
    """
    from ..deconv.label_mapping import get_canonical_cell_types

    cell_types = get_canonical_cell_types()

    # 1. Encode all embeddings from chunked iterators
    logger.info(f"Epoch {epoch}: Encoding full embeddings (streaming chunks)...")
    sc_iter = dataset.iter_expression_chunks(
        "scrna", chunk_size=chunk_size, as_tensor=True, device=device
    )
    st_iter = dataset.iter_expression_chunks(
        "st", obs_indices=st_indices_for_ot, chunk_size=chunk_size, as_tensor=True, device=device
    )
    u_sc, u_st = encode_full_embeddings_from_iterators(vae, sc_iter, st_iter, device)

    # 2. Build cell type one-hot matrix from scRNA labels
    cell_type_onehot_np = cell_types.to_onehot(cell_type_labels)
    cell_type_onehot = torch.tensor(cell_type_onehot_np, dtype=torch.float32)

    # 3. Run OT deconvolution with subsampling
    logger.info(f"Epoch {epoch}: Running OT deconvolution (subsample: sc={subsample_sc}, st={subsample_st})...")
    transport_plan, ot_valid = run_ot_deconvolution(
        u_st, u_sc, epsilon=epsilon,
        subsample_sc=subsample_sc, subsample_st=subsample_st,
    )

    if not ot_valid or transport_plan is None:
        logger.warning(f"Epoch {epoch}: OT deconvolution failed, using fallback")

        if fallback_state is not None and fallback_state.ot_valid:
            logger.info(f"Epoch {epoch}: Reusing previous epoch's condition prior")
            return fallback_state

        # No valid fallback: use uniform proportions
        n_st = u_st.shape[0]
        n_cell_types = cell_types.n_types
        spot_proportions = torch.ones(n_st, n_cell_types) / n_cell_types
        ot_valid = False
    else:
        # 4. Convert transport to spot proportions
        logger.info(f"Epoch {epoch}: Computing spot-level proportions...")
        spot_proportions = transport_to_spot_proportions(
            transport_plan, cell_type_onehot, sparse=transport_plan.is_sparse
        )
        ot_valid = True

    # 5. Aggregate to condition level
    logger.info(f"Epoch {epoch}: Aggregating to condition proportions...")
    pred_condition_proportions, pred_condition_names = aggregate_spot_proportions_by_condition(
        spot_proportions, st_conditions
    )

    # 6. Build state with explicit pred and bulk condition names (Chunk 4)
    state = ConditionPriorState(
        spot_type_proportions=spot_proportions,
        pred_condition_proportions=pred_condition_proportions,
        bulk_condition_proportions=bulk_condition_proportions,
        pred_condition_names=pred_condition_names,
        bulk_condition_names=bulk_condition_names,
        condition_names=pred_condition_names,
        cell_type_names=cell_types.names,
        ot_valid=ot_valid,
        epoch=epoch,
    )

    logger.info(
        f"Epoch {epoch}: Condition prior ready. "
        f"Pred conditions: {pred_condition_names}, "
        f"Bulk conditions: {bulk_condition_names}, "
        f"OT valid: {ot_valid}"
    )

    return state


def compute_bulk_prior_loss_from_state(
    state: ConditionPriorState,
    epoch: int,
    lambda_start: float = 0.01,
    lambda_max: float = 0.1,
    warmup_start: int = 20,
    warmup_end: int = 40,
) -> Tuple[Tensor, Dict[str, float]]:
    """Compute bulk prior KL loss from condition prior state.

    Chunk 2 REVISED: Aligns by condition NAMES, not row position.
    Handles different ordering and partial overlap.

    Parameters
    ----------
    state : ConditionPriorState
        Cached condition prior state
    epoch : int
        Current epoch
    lambda_start : float
        Starting lambda value
    lambda_max : float
        Maximum lambda after warmup
    warmup_start : int
        Epoch when warmup starts
    warmup_end : int
        Epoch when warmup ends

    Returns
    -------
    Tuple[Tensor, Dict[str, float]]
        - kl_loss: Scalar KL loss
        - metrics: Dict with per-condition KL and lambda
            - bulk_prior_loss: loss value
            - bulk_prior_lambda: lambda weight
            - bulk_prior_valid: whether OT was valid
            - bulk_prior_conditions_used: list of common conditions
            - bulk_prior_n_conditions: number of conditions used
    """
    from ..deconv.bulk_prior import compute_condition_level_kl

    if state.pred_condition_proportions is None:
        return torch.tensor(0.0), {
            "bulk_prior_loss": 0.0,
            "bulk_prior_lambda": 0.0,
            "bulk_prior_valid": False,
            "bulk_prior_conditions_used": [],
            "bulk_prior_n_conditions": 0,
        }

    if state.bulk_condition_proportions is None:
        return torch.tensor(0.0), {
            "bulk_prior_loss": 0.0,
            "bulk_prior_lambda": 0.0,
            "bulk_prior_valid": False,
            "bulk_prior_conditions_used": [],
            "bulk_prior_n_conditions": 0,
        }

    if state.condition_names is None:
        return torch.tensor(0.0), {
            "bulk_prior_loss": 0.0,
            "bulk_prior_lambda": 0.0,
            "bulk_prior_valid": False,
            "bulk_prior_conditions_used": [],
            "bulk_prior_n_conditions": 0,
        }

    # Chunk 4: Use explicit pred and bulk condition names (no positional fallback)
    pred_props = state.pred_condition_proportions
    bulk_props = state.bulk_condition_proportions

    # Get condition names - use new explicit fields, fallback to legacy for compat
    pred_names = state.pred_condition_names or state.condition_names
    bulk_names = state.bulk_condition_names

    if bulk_names is None:
        logger.warning(
            f"bulk_condition_names not set in state. "
            f"Cannot align tensors. Returning zero loss."
        )
        return torch.tensor(0.0), {
            "bulk_prior_loss": 0.0,
            "bulk_prior_lambda": 0.0,
            "bulk_prior_valid": False,
            "bulk_prior_conditions_used": [],
            "bulk_prior_n_conditions": 0,
        }

    if pred_names is None:
        logger.warning(
            f"pred_condition_names not set in state. "
            f"Cannot align tensors. Returning zero loss."
        )
        return torch.tensor(0.0), {
            "bulk_prior_loss": 0.0,
            "bulk_prior_lambda": 0.0,
            "bulk_prior_valid": False,
            "bulk_prior_conditions_used": [],
            "bulk_prior_n_conditions": 0,
        }

    # Align by condition names (Chunk 4: explicit alignment, NO positional fallback)
    aligned_pred, aligned_bulk, common_names = align_condition_tensors(
        pred_props, pred_names, bulk_props, bulk_names
    )

    if len(common_names) == 0:
        # No overlap: return zero loss
        return torch.tensor(0.0), {
            "bulk_prior_loss": 0.0,
            "bulk_prior_lambda": 0.0,
            "bulk_prior_valid": False,
            "bulk_prior_conditions_used": [],
            "bulk_prior_n_conditions": 0,
        }

    # Compute lambda from warmup schedule
    if epoch < warmup_start:
        lambda_ = lambda_start
    elif epoch < warmup_end:
        progress = (epoch - warmup_start) / (warmup_end - warmup_start)
        lambda_ = lambda_start + progress * (lambda_max - lambda_start)
    else:
        lambda_ = lambda_max

    # Compute KL on aligned conditions
    kl_loss, kl_per_condition = compute_condition_level_kl(
        aligned_pred,
        aligned_bulk,
        lambda_,
    )

    metrics = {
        "bulk_prior_loss": kl_loss.item(),
        "bulk_prior_lambda": lambda_,
        "bulk_prior_valid": state.ot_valid,
        "bulk_prior_conditions_used": common_names,
        "bulk_prior_n_conditions": len(common_names),
    }

    return kl_loss, metrics


def refresh_condition_prior_for_epoch(
    trainer,
    dataset,
    epoch: int,
    device: str,
    ot_prior_start_epoch: int = 20,
    chunk_size: int = 512,
    all_st_indices: Optional[np.ndarray] = None,
    all_st_conditions: Optional[np.ndarray] = None,
) -> None:
    """Compute and cache condition prior state for this epoch.

    Task 3 from docs/activate-bulk-prior-loss-plan.md.
    Called at epoch start by training scripts.

    Parameters
    ----------
    trainer : TripleModalTrainer
        Trainer with vae and metadata
    dataset : TripleModalDataset or MapStyleTripleModalDataset
        Dataset with get_expression_matrix() helper
    epoch : int
        Current epoch number
    device : str
        Device for computation
    ot_prior_start_epoch : int
        Skip prior before this epoch (warmup)
    chunk_size : int
        Chunk size for encoding (not used in current impl)
    all_st_indices : np.ndarray, optional
        Full ST spot indices (train+val) for OT deconvolution.
        If None, uses dataset.st_indices (training split only).
        Passing all spots ensures all conditions contribute to the
        condition-level bulk prior, even if spatial split separates them.
    all_st_conditions : np.ndarray, optional
        Condition labels for all_st_indices. Must match length.
        If None, uses trainer.st_condition_labels.
    """
    # Gate: Skip before warmup
    if epoch < ot_prior_start_epoch:
        trainer.condition_prior_state = None
        logger.debug(f"Epoch {epoch}: Condition prior inactive (epoch < {ot_prior_start_epoch})")
        return

    # Gate 1: Check required metadata
    if trainer.scrna_cell_type_labels is None:
        logger.warning(f"Epoch {epoch}: Condition prior disabled (missing scrna_cell_type_labels)")
        trainer.condition_prior_state = None
        return

    if trainer.st_condition_labels is None:
        logger.warning(f"Epoch {epoch}: Condition prior disabled (missing st_condition_labels)")
        trainer.condition_prior_state = None
        return

    if trainer.bulk_condition_proportions is None:
        logger.warning(f"Epoch {epoch}: Condition prior disabled (missing bulk_condition_proportions)")
        trainer.condition_prior_state = None
        return

    # Save previous state as fallback
    fallback_state = trainer.condition_prior_state

    # Determine ST indices/conditions for OT deconvolution
    # Use all ST spots (train+val) so all conditions contribute to prior
    st_indices_for_ot = all_st_indices if all_st_indices is not None else dataset.st_indices
    st_conditions_for_ot = all_st_conditions if all_st_conditions is not None else trainer.st_condition_labels

    try:
        logger.info(f"Epoch {epoch}: Computing condition prior (streaming chunks)...")

        stream_failed = False
        # Prefer streaming path if iter_expression_chunks is available.
        # Some datasets expose the method but cannot actually stream in current mode.
        if hasattr(dataset, "iter_expression_chunks"):
            try:
                state = compute_epoch_condition_prior_streaming(
                    vae=trainer.vae,
                    dataset=dataset,
                    cell_type_labels=trainer.scrna_cell_type_labels,
                    st_conditions=st_conditions_for_ot,
                    bulk_condition_proportions=trainer.bulk_condition_proportions,
                    bulk_condition_names=trainer.bulk_condition_names,
                    device=device,
                    epsilon=trainer.ot_epsilon,
                    epoch=epoch,
                    fallback_state=fallback_state,
                    chunk_size=chunk_size,
                    st_indices_for_ot=st_indices_for_ot,
                )
            except RuntimeError as stream_error:
                stream_failed = True
                logger.warning(
                    f"Epoch {epoch}: Streaming condition prior unavailable: {stream_error}. "
                    f"Falling back to materialized matrices."
                )

        if (not hasattr(dataset, "iter_expression_chunks")) or stream_failed:
            # Fallback to materialized path for backward compatibility
            X_sc = dataset.get_expression_matrix("scrna", as_tensor=True, device=device)
            X_st = dataset.get_expression_matrix("st", st_indices_for_ot, as_tensor=True, device=device)

            logger.info(
                f"Epoch {epoch}: Expression matrices: "
                f"X_sc={X_sc.shape}, X_st={X_st.shape}, "
                f"OT uses {'all' if all_st_indices is not None else 'train-only'} ST spots"
            )

            state = compute_epoch_condition_prior(
                vae=trainer.vae,
                X_sc=X_sc,
                X_st=X_st,
                cell_type_labels=trainer.scrna_cell_type_labels,
                st_conditions=st_conditions_for_ot,
                bulk_condition_proportions=trainer.bulk_condition_proportions,
                bulk_condition_names=trainer.bulk_condition_names,
                device=device,
                epsilon=trainer.ot_epsilon,
                epoch=epoch,
                fallback_state=fallback_state,
            )

        trainer.condition_prior_state = state
        logger.info(
            f"Epoch {epoch}: Condition prior ready. "
            f"Conditions: {state.condition_names}, "
            f"OT valid: {state.ot_valid}"
        )

    except (RuntimeError, ValueError, torch._C._LinAlgError) as e:
        # SAFE-01 FIX: Specific exception for condition prior failures
        # RuntimeError: PyTorch runtime errors
        # ValueError: invalid input values
        # LinAlgError: linear algebra failures
        logger.warning(f"Epoch {epoch}: Condition prior computation failed: {e}")
        # Keep previous fallback state
        if fallback_state is not None:
            logger.info(f"Epoch {epoch}: Keeping previous condition prior state")
        else:
            trainer.condition_prior_state = None
