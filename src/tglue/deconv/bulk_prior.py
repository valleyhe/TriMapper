"""Bulk RNA-seq prior integration via ssGSEA preprocessing and cluster-level KL divergence.

Implements:
- D-10: Bulk preprocessing via ssGSEA to produce cell type proportions
- D-11: Bulk prior lambda warm-up schedule (0.01->0.1 from epoch 20-40)
- Anti-pattern enforcement: Bulk prior at cluster level ONLY (never per-spot)
- Condition-level bulk prior aggregation

REVISED: Uses validated gseapy-based ssGSEA from preprocessing/ssgsea_bulk.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict

import torch
from torch import Tensor
import numpy as np

# Import validated ssGSEA implementation
from ..preprocessing.ssgsea_bulk import preprocess_bulk_ssgsea as preprocess_bulk_ssgsea_validated


@dataclass
class BulkPriorConfig:
    """Configuration for Bulk prior integration.

    Attributes:
        lambda_start: Initial lambda value during burn-in (default: 0.01)
        lambda_max: Maximum lambda value after warm-up (default: 0.1)
        warmup_start: Epoch when bulk prior starts ramping from lambda_start (default: 20)
        warmup_end: Epoch when bulk prior reaches lambda_max (default: 40)
        use_ssgsea: Whether to use ssGSEA for bulk preprocessing (default: True)
        online_prior_temperature: Temperature scaling factor that softens the online prior contribution (default: 0.1)
        cell_type_markers: Optional dict mapping cell type names to gene lists
    """

    lambda_start: float = 0.01
    lambda_max: float = 0.1
    warmup_start: int = 20
    warmup_end: int = 40
    use_ssgsea: bool = True
    online_prior_temperature: float = 0.1
    cell_type_markers: Optional[dict] = None


@dataclass
class BulkPriorOutput:
    """Output from bulk prior computation.

    Attributes:
        kl_loss: Scalar KL loss weighted by lambda
        kl_per_cluster: Per-cluster KL divergence values (n_clusters,)
        bulk_proportions: Bulk-derived cell type proportions (n_clusters, n_types)
        current_lambda: Lambda value used for this computation
    """

    kl_loss: Tensor
    kl_per_cluster: Tensor
    bulk_proportions: Tensor
    current_lambda: float


def compute_bulk_prior_lambda(epoch: int, config: BulkPriorConfig) -> float:
    """Compute bulk prior lambda based on warm-up schedule.

    D-11: Lambda warm-up schedule:
    - Epoch 0..warmup_start (0-20): lambda = lambda_start (0.01)
    - Epoch warmup_start..warmup_end (20-40): linear ramp from lambda_start to lambda_max
    - Epoch >= warmup_end (40+): lambda = lambda_max (0.1)

    Parameters
    ----------
    epoch : int
        Current training epoch
    config : BulkPriorConfig
        Configuration with warm-up parameters

    Returns
    -------
    float
        Lambda value for bulk prior loss at current epoch
    """
    if epoch < config.warmup_start:
        # Burn-in phase: minimal constraint
        return config.lambda_start
    elif epoch < config.warmup_end:
        # Ramp phase: linear interpolation
        progress = (epoch - config.warmup_start) / (config.warmup_end - config.warmup_start)
        return config.lambda_start + progress * (config.lambda_max - config.lambda_start)
    else:
        # Full prior strength after warm-up
        return config.lambda_max


def preprocess_bulk_ssgsea(
    bulk_adata,
    cell_type_markers: dict,
    n_bulk_samples: Optional[int] = None,
) -> Tensor:
    """Preprocess bulk RNA-seq data using ssGSEA to estimate cell type proportions.

    DEPRECATED: This function now calls preprocess_bulk_ssgsea_validated from
    preprocessing/ssgsea_bulk.py for proper gene-name matching.

    For new code, use preprocess_bulk_with_metadata() which provides richer output
    including condition-level aggregation.

    Parameters
    ----------
    bulk_adata : AnnData
        Bulk RNA-seq data of shape (n_bulk_samples, n_genes)
    cell_type_markers : dict
        Dictionary mapping cell type names to gene lists
    n_bulk_samples : int, optional
        Number of bulk samples (ignored, inferred from bulk_adata)

    Returns
    -------
    Tensor
        Bulk-derived cell type proportions of shape (n_bulk_samples, n_cell_types)

    Notes
    -----
    This is a compatibility wrapper. Gene name matching is handled by
    preprocess_bulk_ssgsea_validated via gseapy.
    """
    # Call validated implementation
    result = preprocess_bulk_ssgsea_validated(
        bulk_adata,
        gene_sets=cell_type_markers,
        normalize=True,
        return_names=False,
    )
    return result.proportions


def compute_cluster_level_kl(
    cluster_proportions: Tensor,
    bulk_proportions: Tensor,
    lambda_: float,
) -> Tuple[Tensor, Tensor]:
    """Compute KL divergence between cluster proportions and bulk-derived proportions.

    Anti-pattern enforcement: Bulk prior at CLUSTER LEVEL only, never per-spot.

    KL divergence per cluster: KL(cluster_props || bulk_props)
    KL(p||q) = sum_t p[t] * log(p[t] / q[t])

    Parameters
    ----------
    cluster_proportions : Tensor
        (n_clusters, n_cell_types) cluster-level cell type proportions from deconvolution
    bulk_proportions : Tensor
        (n_clusters, n_cell_types) cell type proportions from bulk ssGSEA
    lambda_ : float
        Weight for the bulk prior loss

    Returns
    -------
    Tuple[Tensor, Tensor]
        - kl_loss: Scalar KL loss = mean(per_cluster_kl) * lambda_
        - kl_per_cluster: (n_clusters,) KL divergence for each cluster
    """
    # Ensure inputs are on the same device
    device = cluster_proportions.device

    # Add small epsilon for numerical stability in KL computation
    eps = 1e-8

    # Ensure both are valid probability distributions
    cluster_props = torch.clamp(cluster_proportions, min=eps)
    bulk_props = torch.clamp(bulk_proportions, min=eps)

    # WR-02 FIX: Check for near-zero row sums before log computation
    # This prevents division by zero if input tensors have rows that sum to approximately zero
    row_sums_cluster = cluster_props.sum(dim=1, keepdim=True)
    row_sums_bulk = bulk_props.sum(dim=1, keepdim=True)

    if (row_sums_cluster < 1e-6).any():
        import logging
        logging.getLogger(__name__).warning(
            "Near-zero probability distribution detected in cluster_props for KL computation"
        )
        # Replace near-zero rows with uniform distribution
        near_zero_mask = (row_sums_cluster < 1e-6).squeeze(-1)
        cluster_props[near_zero_mask] = 1.0 / cluster_props.shape[1]

    if (row_sums_bulk < 1e-6).any():
        import logging
        logging.getLogger(__name__).warning(
            "Near-zero probability distribution detected in bulk_props for KL computation"
        )
        near_zero_mask = (row_sums_bulk < 1e-6).squeeze(-1)
        bulk_props[near_zero_mask] = 1.0 / bulk_props.shape[1]

    # Re-normalize to ensure valid distributions
    cluster_props = cluster_props / cluster_props.sum(dim=1, keepdim=True)
    bulk_props = bulk_props / bulk_props.sum(dim=1, keepdim=True)

    # P0-03 FIX: Use log_softmax for numerical stability
    # log_softmax handles epsilon internally and prevents log(0)
    import torch.nn.functional as F

    # Convert to log-space for stability
    log_cluster = F.log_softmax(cluster_props, dim=1)  # Safe log
    log_bulk = F.log_softmax(bulk_props, dim=1)        # Safe log

    # KL divergence: KL(cluster || bulk) = sum_t cluster[t] * (log(cluster[t] / bulk[t]))
    kl_per_cluster = (cluster_props * (log_cluster - log_bulk)).sum(dim=1)

    # Total loss: mean of per-cluster KLs weighted by lambda
    kl_loss = kl_per_cluster.mean() * lambda_

    return kl_loss, kl_per_cluster


def compute_bulk_prior_loss(
    cluster_proportions: Tensor,
    bulk_proportions: Tensor,
    lambda_: float,
) -> BulkPriorOutput:
    """Compute bulk prior loss at cluster level.

    Combines cluster-level KL divergence computation with lambda weighting.

    Parameters
    ----------
    cluster_proportions : Tensor
        (n_clusters, n_cell_types) from OT deconvolution
    bulk_proportions : Tensor
        (n_bulk_samples, n_cell_types) from ssGSEA preprocessing
    lambda_ : float
        Current lambda from warm-up schedule

    Returns
    -------
    BulkPriorOutput
        Dataclass containing kl_loss, kl_per_cluster, bulk_proportions, current_lambda
    """
    kl_loss, kl_per_cluster = compute_cluster_level_kl(
        cluster_proportions, bulk_proportions, lambda_
    )

    return BulkPriorOutput(
        kl_loss=kl_loss,
        kl_per_cluster=kl_per_cluster,
        bulk_proportions=bulk_proportions,
        current_lambda=lambda_,
    )


@dataclass
class BulkPreprocessingOutput:
    """Rich output from bulk preprocessing.

    Attributes:
        sample_proportions: (n_bulk_samples, n_cell_types) cell type proportions per sample
        cell_type_names: List of canonical cell type names (ordered)
        sample_ids: List of sample identifiers
        sample_conditions: List of canonical condition labels for each sample
        condition_proportions: (n_conditions, n_cell_types) aggregated proportions by condition
        condition_names: List of canonical condition names (ordered)
    """

    sample_proportions: Tensor
    cell_type_names: List[str]
    sample_ids: List[str]
    sample_conditions: List[str]
    condition_proportions: Optional[Tensor] = None
    condition_names: Optional[List[str]] = None


def aggregate_bulk_by_condition(
    sample_proportions: Tensor,
    sample_conditions: List[str],
    condition_names: Optional[List[str]] = None,
) -> Tuple[Tensor, List[str]]:
    """Aggregate bulk sample-level proportions to condition-level.

    Parameters
    ----------
    sample_proportions : Tensor
        (n_bulk_samples, n_cell_types) cell type proportions per sample
    sample_conditions : List[str]
        List of canonical condition labels for each sample
    condition_names : List[str], optional
        Canonical condition ordering. If None, inferred from unique conditions.

    Returns
    -------
    Tuple[Tensor, List[str]]
        - condition_proportions: (n_conditions, n_cell_types) aggregated proportions
        - condition_names: List of canonical condition names in order

    Notes
    -----
    Aggregation is by mean (average proportions across samples in same condition).
    Result is renormalized to valid simplex (rows sum to 1).
    """
    # Get unique conditions and ordering
    unique_conditions = list(set(sample_conditions))

    if condition_names is None:
        # Use canonical ordering if available
        from .label_mapping import get_canonical_conditions
        conditions = get_canonical_conditions()
        # Sort by canonical order
        condition_names = [c for c in conditions.names if c in unique_conditions]
        # Add any conditions not in canonical (should not happen)
        for c in unique_conditions:
            if c not in condition_names:
                condition_names.append(c)
    else:
        # Filter to only conditions that exist
        condition_names = [c for c in condition_names if c in unique_conditions]

    n_conditions = len(condition_names)
    n_cell_types = sample_proportions.shape[1]

    # Aggregate by condition
    condition_proportions = torch.zeros(n_conditions, n_cell_types, dtype=sample_proportions.dtype)

    for i, cond in enumerate(condition_names):
        # Find samples belonging to this condition
        mask = [j for j, c in enumerate(sample_conditions) if c == cond]
        if len(mask) > 0:
            # Mean aggregation
            condition_proportions[i] = sample_proportions[mask].mean(dim=0)
        else:
            # Fallback: uniform (should not happen if condition_names filtered correctly)
            condition_proportions[i] = torch.ones(n_cell_types) / n_cell_types

    # Renormalize to valid simplex
    row_sums = condition_proportions.sum(dim=1, keepdim=True)
    condition_proportions = condition_proportions / (row_sums + 1e-8)

    return condition_proportions, condition_names


def preprocess_bulk_with_metadata(
    bulk_adata,
    cell_type_markers: dict | str | None = None,
    condition_col: str = "condition",
    sample_id_col: str = "sample",
) -> BulkPreprocessingOutput:
    """Preprocess bulk RNA-seq data with full metadata output.

    REVISED: Uses validated gseapy-based ssGSEA from preprocessing/ssgsea_bulk.py.
    Returns tensors with columns reordered to canonical cell type order.

    Extends preprocess_bulk_ssgsea to include sample conditions and
    pre-aggregated condition-level proportions.

    Parameters
    ----------
    bulk_adata : AnnData
        Bulk RNA-seq data with obs containing condition and sample metadata
    cell_type_markers : dict | str | None
        Gene sets for ssGSEA:
        - None: Use bundled markers (skin_markers.gmt for Rosacea)
        - str: GMT file path
        - dict: {cell_type_name: [gene1, gene2, ...]}
    condition_col : str
        Column name for condition metadata
    sample_id_col : str
        Column name for sample ID metadata

    Returns
    -------
    BulkPreprocessingOutput
        Rich output with sample and condition level proportions.
        sample_proportions columns are reordered to canonical cell type order.

    Notes
    -----
    This function now uses the validated gseapy-based ssGSEA implementation
    from preprocessing/ssgsea_bulk.py, which properly handles gene name matching
    and produces non-uniform cell type proportions from real marker genes.

    Key fix: Columns are reordered to canonical cell type order so that
    KL divergence compares semantically aligned cell types.
    """
    from .label_mapping import get_canonical_cell_types, get_canonical_conditions

    cell_types = get_canonical_cell_types()
    conditions = get_canonical_conditions()

    # Call validated ssGSEA implementation
    # This uses gseapy for real gene-name matching and enrichment scoring
    # Now returns both proportions and cell type names from GMT
    ssgsea_output = preprocess_bulk_ssgsea_validated(
        bulk_adata, gene_sets=cell_type_markers, normalize=True, return_names=True
    )

    raw_sample_proportions = ssgsea_output.proportions
    raw_cell_type_names = ssgsea_output.cell_type_names

    # Reorder tensor columns to canonical cell type ordering
    # This is the key fix: tensor columns must match canonical names
    sample_proportions, cell_type_names = reorder_columns_to_canonical(
        raw_sample_proportions, raw_cell_type_names, cell_types.names
    )

    # Get metadata
    n_samples = bulk_adata.shape[0]

    # Sample IDs
    if sample_id_col in bulk_adata.obs.columns:
        sample_ids = bulk_adata.obs[sample_id_col].tolist()
    else:
        sample_ids = [f"sample_{i}" for i in range(n_samples)]

    # Sample conditions (normalized to canonical)
    if condition_col in bulk_adata.obs.columns:
        raw_conditions = bulk_adata.obs[condition_col].tolist()
        sample_conditions = [conditions.normalize(str(c)) for c in raw_conditions]
    else:
        sample_conditions = ["Unknown"] * n_samples

    # Aggregate to condition level using the reordered tensor
    condition_proportions, condition_names = aggregate_bulk_by_condition(
        sample_proportions,
        sample_conditions,
        conditions.names,
    )

    return BulkPreprocessingOutput(
        sample_proportions=sample_proportions,
        cell_type_names=cell_type_names,
        sample_ids=sample_ids,
        sample_conditions=sample_conditions,
        condition_proportions=condition_proportions,
        condition_names=condition_names,
    )


def reorder_columns_to_canonical(
    sample_proportions: Tensor,
    raw_names: List[str],
    canonical_names: List[str],
) -> Tuple[Tensor, List[str]]:
    """Reorder tensor columns to match canonical cell type names.

    This is the key fix for semantic alignment: the ssGSEA output returns
    proportions in the order of the GMT gene sets, which may not match
    canonical ordering. KL divergence would then compare mismatched columns.

    Parameters
    ----------
    sample_proportions : Tensor
        (n_samples, n_cell_types) proportions in raw column order
    raw_names : List[str]
        Cell type names corresponding to columns of sample_proportions
    canonical_names : List[str]
        Canonical ordering from get_canonical_cell_types()

    Returns
    -------
    Tuple[Tensor, List[str]]
        - reordered_tensor: (n_samples, n_aligned_types) with columns reordered
        - aligned_names: List of cell type names in canonical order

    Notes
    -----
    - Only columns present in both raw and canonical are included
    - Extra names (in raw but not canonical) are appended after canonical intersection
    - Missing canonical names are skipped
    - Tensor dtype and device are preserved
    """
    if sample_proportions.shape[1] != len(raw_names):
        raise ValueError(
            f"sample_proportions has {sample_proportions.shape[1]} columns "
            f"but raw_names has {len(raw_names)} elements"
        )

    # Build aligned order: canonical intersection first, then extras
    aligned_names: List[str] = []

    # Add canonical names that exist in raw (preserving canonical order)
    for name in canonical_names:
        if name in raw_names:
            aligned_names.append(name)

    # Add extra names not in canonical (preserving raw order)
    for name in raw_names:
        if name not in aligned_names:
            aligned_names.append(name)

    # Build column indices for reordering
    indices = []
    for aligned_name in aligned_names:
        raw_idx = raw_names.index(aligned_name)
        indices.append(raw_idx)

    # Reorder tensor columns
    if len(indices) == 0:
        # Edge case: no overlap, return empty tensor
        reordered_tensor = torch.empty(
            sample_proportions.shape[0], 0,
            dtype=sample_proportions.dtype,
            device=sample_proportions.device
        )
    else:
        reordered_tensor = sample_proportions[:, indices]

    return reordered_tensor, aligned_names


def reorder_to_canonical(
    raw_names: List[str],
    canonical_names: List[str],
) -> List[str]:
    """Reorder cell type names to canonical ordering (name-only, no tensor).

    DEPRECATED: Use reorder_columns_to_canonical() which also reorders tensor.

    Parameters
    ----------
    raw_names : List[str]
        Cell type names from ssGSEA output (order depends on GMT/dict)
    canonical_names : List[str]
        Canonical ordering from get_canonical_cell_types()

    Returns
    -------
    List[str]
        Reordered cell type names (intersection with canonical)
    """
    # Find intersection
    intersection = [n for n in canonical_names if n in raw_names]
    # Add any extra names not in canonical
    for n in raw_names:
        if n not in intersection:
            intersection.append(n)
    return intersection


def compute_condition_level_kl(
    pred_condition_proportions: Tensor,
    bulk_condition_proportions: Tensor,
    lambda_: float,
) -> Tuple[Tensor, Tensor]:
    """Compute KL divergence between predicted and bulk condition proportions.

    This is the correct bulk prior for triple-modal integration:
    - pred_condition_proportions: from OT deconv + ST condition aggregation
    - bulk_condition_proportions: from ssGSEA + bulk condition aggregation
    - KL(pred || bulk) per condition

    Parameters
    ----------
    pred_condition_proportions : Tensor
        (n_conditions, n_cell_types) predicted proportions per condition
    bulk_condition_proportions : Tensor
        (n_conditions, n_cell_types) bulk-derived proportions per condition
    lambda_ : float
        Weight for the bulk prior loss

    Returns
    -------
    Tuple[Tensor, Tensor]
        - kl_loss: Scalar KL loss = mean(per_condition_kl) * lambda_
        - kl_per_condition: (n_conditions,) KL divergence for each condition
    """
    device = pred_condition_proportions.device
    eps = 1e-8

    # Ensure valid probability distributions
    pred_props = torch.clamp(pred_condition_proportions, min=eps)
    bulk_props = torch.clamp(bulk_condition_proportions, min=eps)

    # Re-normalize
    pred_props = pred_props / pred_props.sum(dim=1, keepdim=True)
    bulk_props = bulk_props / bulk_props.sum(dim=1, keepdim=True)

    # KL divergence per condition: KL(pred || bulk)
    kl_per_condition = (pred_props * (torch.log(pred_props) - torch.log(bulk_props))).sum(dim=1)

    # Total loss: mean of per-condition KLs weighted by lambda
    kl_loss = kl_per_condition.mean() * lambda_

    return kl_loss, kl_per_condition
