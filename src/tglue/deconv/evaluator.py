"""Deconvolution evaluation metrics for OT-based cell type proportion estimation.

EV-03: Pearson correlation vs ground truth (threshold > 0.5)
EV-04: Bulk alignment KL divergence (should decrease during training)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import torch
from scipy.stats import pearsonr
from torch import Tensor


def compute_pearson_correlation(
    proportions_pred: Tensor,
    proportions_true: Tensor,
) -> float:
    """Compute Pearson correlation between predicted and true cell type proportions.

    EV-03: Deconvolution accuracy metric - Pearson correlation vs ground truth.

    Flattens both tensors to 1D and computes Pearson correlation coefficient.
    Handles edge case where std=0 (constant values) by returning NaN with warning.

    Parameters
    ----------
    proportions_pred : Tensor
        (n_spots, n_types) predicted cell type proportions.
    proportions_true : Tensor
        (n_spots, n_types) ground truth cell type proportions.

    Returns
    -------
    float
        Pearson correlation coefficient r in [-1, 1].
        Returns NaN if either input has zero standard deviation (constant values).
    """
    # Convert to numpy and flatten
    pred_np = proportions_pred.detach().cpu().numpy().flatten()
    true_np = proportions_true.detach().cpu().numpy().flatten()

    # Check for constant inputs (std = 0)
    if np.std(pred_np) == 0 or np.std(true_np) == 0:
        warnings.warn(
            "Constant input detected in Pearson correlation computation. "
            "Returning NaN.",
            UserWarning,
        )
        return float("nan")

    r, _ = pearsonr(pred_np, true_np)
    return float(r)


def compute_bulk_alignment_kl(
    cluster_proportions: Tensor,
    bulk_proportions: Tensor,
) -> Tuple[float, Tensor]:
    """Compute KL divergence between cluster proportions and Bulk prior.

    EV-04: Bulk alignment metric - KL divergence should decrease during training.

    Computes per-cluster KL divergence: KL(cluster || bulk) = sum p * log(p / q)
    Uses natural logarithm (same as torch.nn.functional.kl_div with log_target=True).

    Parameters
    ----------
    cluster_proportions : Tensor
        (n_clusters, n_types) cluster cell type proportions.
    bulk_proportions : Tensor
        (n_clusters, n_types) Bulk prior proportions per cluster.

    Returns
    -------
    Tuple[float, Tensor]
        - mean_kl: Mean KL divergence across clusters.
        - per_cluster_kl: (n_clusters,) KL divergence per cluster.
    """
    # Ensure non-negative and normalized (numerical stability)
    p = cluster_proportions + 1e-10
    p = p / p.sum(dim=1, keepdim=True)

    q = bulk_proportions + 1e-10
    q = q / q.sum(dim=1, keepdim=True)

    # KL divergence: sum over types: p * log(p / q)
    # Using natural log, same as F.kl_div with log_target=False and reduction='none'
    kl_per_cluster = (p * (p / q).log()).sum(dim=1)

    return float(kl_per_cluster.mean()), kl_per_cluster


def kl_divergence(p: Tensor, q: Tensor, eps: float = 1e-10) -> Tensor:
    """Compute KL divergence KL(p || q) between two probability distributions.

    Parameters
    ----------
    p : Tensor
        (..., n_types) first distribution (numerator).
    q : Tensor
        (..., n_types) second distribution (denominator).
    eps : float
        Small constant for numerical stability.

    Returns
    -------
    Tensor
        KL divergence per batch element.
    """
    p = p + eps
    q = q + eps
    p = p / p.sum(dim=-1, keepdim=True)
    q = q / q.sum(dim=-1, keepdim=True)
    return (p * (p / q).log()).sum(dim=-1)


@dataclass
class DeconvolutionMetrics:
    """Dataclass for deconvolution evaluation metrics.

    Attributes
    ----------
    pearson_r : float
        Pearson correlation coefficient between estimated and true proportions.
    p_value : float
        P-value of the Pearson correlation.
    kl_mean : float
        Mean KL divergence across clusters.
    kl_per_cluster : Tensor
        Per-cluster KL divergence values.
    is_valid : bool
        Whether all metrics are valid (not NaN/Inf).
    threshold_met : bool
        Whether Pearson correlation > 0.5 (EV-03 requirement).
    """

    pearson_r: float
    p_value: float
    kl_mean: float
    kl_per_cluster: Tensor
    is_valid: bool
    threshold_met: bool

    def __str__(self) -> str:
        return (
            f"DeconvolutionMetrics(pearson_r={self.pearson_r:.4f}, "
            f"p_value={self.p_value:.4e}, kl_mean={self.kl_mean:.4f}, "
            f"threshold_met={self.threshold_met})"
        )


def evaluate_deconvolution(
    transport_plan: Tensor,
    cell_type_onehot: Tensor,
    ground_truth: Tensor,
    bulk_prior: Optional[Tensor] = None,
) -> DeconvolutionMetrics:
    """Evaluate deconvolution quality against ground truth and Bulk prior.

    Combines EV-03 (Pearson correlation) and EV-04 (Bulk alignment KL) metrics.

    Parameters
    ----------
    transport_plan : Tensor
        (n_spots, n_cells) OT transport plan.
    cell_type_onehot : Tensor
        (n_cells, n_types) binary cell type indicator matrix.
    ground_truth : Tensor
        (n_spots, n_types) ground truth cell type proportions.
    bulk_prior : Tensor, optional
        (n_clusters, n_types) Bulk prior proportions for KL evaluation.

    Returns
    -------
    DeconvolutionMetrics
        Dataclass containing all evaluation metrics.
    """
    # Convert transport plan to proportions using matrix multiplication
    # transport_plan @ cell_type_onehot gives (n_spots, n_types)
    proportions = transport_plan @ cell_type_onehot

    # Project to simplex (non-negative, sum to 1)
    proportions = _simplex_projection(proportions)

    # Compute Pearson correlation (EV-03)
    if ground_truth is not None:
        pearson_r, p_value = _pearson_with_pvalue(proportions, ground_truth)
        threshold_met = pearson_r > 0.5
    else:
        pearson_r = 0.0
        p_value = 1.0
        threshold_met = False

    # Compute Bulk alignment KL (EV-04)
    if bulk_prior is not None:
        kl_mean, kl_per_cluster = compute_bulk_alignment_kl(proportions, bulk_prior)
    else:
        kl_mean = 0.0
        kl_per_cluster = torch.zeros(proportions.shape[0])

    # Check validity
    is_valid = (
        np.isfinite(pearson_r)
        and np.isfinite(p_value)
        and np.isfinite(kl_mean)
    )

    return DeconvolutionMetrics(
        pearson_r=pearson_r,
        p_value=p_value,
        kl_mean=kl_mean,
        kl_per_cluster=kl_per_cluster,
        is_valid=is_valid,
        threshold_met=threshold_met,
    )


def _simplex_projection(x: Tensor) -> Tensor:
    """Project rows to probability simplex (non-negative, sum to 1).

    Parameters
    ----------
    x : Tensor
        (n, k) input tensor.

    Returns
    -------
    Tensor
        (n, k) projected tensor where each row sums to 1.
    """
    n, k = x.shape
    x_flat = x.reshape(-1, k)

    # Sort for projection algorithm
    sorted_x, _ = torch.sort(x_flat, dim=1, descending=True)
    cumsum = torch.cumsum(sorted_x, dim=1)

    # Find threshold
    t = torch.arange(1, k + 1, device=x.device).float()
    cond = sorted_x - (cumsum - 1) / t > 0
    idx = cond.long().sum(dim=1, keepdim=True)
    idx = idx.clamp(min=1)

    # Compute tau and project
    tau = (cumsum.gather(1, idx - 1) - 1) / idx.float()
    tau = tau.clamp(min=0)

    proj = (x_flat - tau).clamp(min=0)
    proj = proj / proj.sum(dim=1, keepdim=True)

    return proj.reshape(n, k)


def _pearson_with_pvalue(
    pred: Tensor,
    true: Tensor,
) -> Tuple[float, float]:
    """Compute Pearson correlation with p-value.

    Parameters
    ----------
    pred : Tensor
        (n_spots, n_types) predicted proportions.
    true : Tensor
        (n_spots, n_types) true proportions.

    Returns
    -------
    Tuple[float, float]
        (r, p_value) Pearson correlation and p-value.
    """
    pred_np = pred.detach().cpu().numpy().flatten()
    true_np = true.detach().cpu().numpy().flatten()

    if np.std(pred_np) == 0 or np.std(true_np) == 0:
        return 0.0, 1.0

    r, p = pearsonr(pred_np, true_np)
    return float(r), float(p)


def log_deconvolution_metrics(
    metrics: DeconvolutionMetrics,
    logger,
    epoch: int,
    prefix: str = "deconvolution/",
) -> None:
    """Log deconvolution metrics to logger.

    Parameters
    ----------
    metrics : DeconvolutionMetrics
        Metrics dataclass to log.
    logger : object
        Logger with .log() or .add_scalar() method.
        Supports:
        - torch.utils.tensorboard.SummaryWriter
        - wandb.sdk.wandb_run.Run
    epoch : int
        Current training epoch.
    prefix : str
        Prefix for metric keys.
    """
    # Try TensorBoard style logging
    try:
        logger.add_scalar(f"{prefix}pearson_r", metrics.pearson_r, epoch)
        logger.add_scalar(f"{prefix}p_value", metrics.p_value, epoch)
        logger.add_scalar(f"{prefix}kl_mean", metrics.kl_mean, epoch)
        logger.add_scalar(f"{prefix}threshold_met", float(metrics.threshold_met), epoch)
        return
    except (AttributeError, TypeError, RuntimeError):  # SAFE-01: Logger API mismatch or backend error
        pass

    # Try wandb style logging
    try:
        logger.log({
            f"{prefix}pearson_r": metrics.pearson_r,
            f"{prefix}kl_mean": metrics.kl_mean,
            "epoch": epoch,
        })
        return
    except (AttributeError, TypeError, RuntimeError):  # SAFE-01: Logger API mismatch or backend error
        pass

    # Fallback: print
    print(
        f"[Epoch {epoch}] {prefix}pearson_r={metrics.pearson_r:.4f}, "
        f"{prefix}kl_mean={metrics.kl_mean:.4f}, "
        f"threshold_met={metrics.threshold_met}"
    )


# Alias for backward compatibility
compute_cell_type_proportions = _simplex_projection
