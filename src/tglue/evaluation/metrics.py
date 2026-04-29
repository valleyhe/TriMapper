"""Alignment evaluation metrics: ASW, NMI, GC.

EV-01: ASW (Average Silhouette Width), NMI (Normalized Mutual Information), GC (Graph Connectivity)
All metrics return scalar values that can be logged during training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.neighbors import NearestNeighbors
from torch import Tensor


def compute_asw(
    latent: Tensor,
    modality_labels: Optional[Tensor] = None,
    batch_labels: Optional[Tensor] = None,
    per_label: bool = False,
) -> Union[Tensor, dict]:
    """Compute Average Silhouette Width for latent embeddings.

    ASW measures how well latent embeddings cluster by modality.
    Higher ASW (closer to 1) = better modality separation.
    ASW = mean over all samples of s(i) = (b(i) - a(i)) / max(a(i), b(i))

    Parameters
    ----------
    latent : Tensor
        (n_samples, latent_dim) — latent representations.
    modality_labels : Tensor, optional
        (n_samples,) — modality labels (0=scRNA, 1=ST, 2=Bulk).
        If None, batch_labels must be provided.
    batch_labels : Tensor, optional
        (n_samples,) — batch labels for batch-aware ASW.
        If None, modality_labels must be provided.
    per_label : bool, default False
        If True, return dict {modality_idx: asw_score} for each modality.

    Returns
    -------
    Union[Tensor, dict]
        If per_label=False: scalar mean ASW across all samples.
        If per_label=True: dict mapping modality label to ASW for that modality.
    """
    if modality_labels is None and batch_labels is None:
        raise ValueError("Either modality_labels or batch_labels must be provided.")

    labels = modality_labels if modality_labels is not None else batch_labels
    if isinstance(labels, Tensor):
        labels = labels.cpu().numpy()

    X = latent.detach().cpu().numpy()
    n_samples = X.shape[0]

    if per_label:
        unique_labels = np.unique(labels)
        result = {}
        for label in unique_labels:
            mask = labels == label
            if mask.sum() < 2:
                # Can't compute silhouette with < 2 samples of same label
                result[int(label)] = torch.tensor(float("nan"))
                continue
            # Use all other labels as "other" for silhouette
            # sklearn silhouette_score needs at least 2 clusters
            other_mask = ~mask
            if other_mask.sum() < 1:
                # Only one label present
                result[int(label)] = torch.tensor(float("nan"))
                continue
            try:
                score = silhouette_score(X, labels, sample_size=min(5000, n_samples))
            except (ValueError, RuntimeError) as e:
                # SAFE-01 FIX: Specific exception for silhouette failures
                # ValueError: insufficient samples, single label
                # RuntimeError: numerical instability
                score = float("nan")
            result[int(label)] = torch.tensor(score)
        return result

    try:
        score = silhouette_score(X, labels, sample_size=min(5000, n_samples))
    except (ValueError, RuntimeError) as e:
        # SAFE-01 FIX: Specific exception for silhouette failures
        # ValueError: insufficient samples, single label
        # RuntimeError: numerical instability
        score = float("nan")
    return torch.tensor(score)


def compute_nmi(
    labels_true: np.ndarray,
    labels_pred: Optional[np.ndarray] = None,
    n_clusters: Optional[int] = None,
) -> float:
    """Compute Normalized Mutual Information between true and predicted labels.

    NMI measures cluster-level modality alignment.
    NMI = 2 * I(true, pred) / (H(true) + H(pred)) in [0, 1], where 1 = perfect alignment.

    Parameters
    ----------
    labels_true : np.ndarray
        (n_samples,) — true modality labels.
    labels_pred : np.ndarray, optional
        (n_samples,) — predicted cluster labels.
        If None, use k-means on latent to generate predicted labels.
    n_clusters : int, optional
        Number of k-means clusters if labels_pred is None.
        Default: min(10, n_modalities).

    Returns
    -------
    float
        NMI score in [0, 1].
    """
    if labels_pred is None:
        raise ValueError("labels_pred must be provided if not using k-means on latent")
    return normalized_mutual_info_score(labels_true, labels_pred)


def compute_spatial_ari(
    latent: Tensor,
    spatial_labels: np.ndarray,
    n_clusters: Optional[int] = None,
) -> float:
    """Compute Adjusted Rand Index for spatial domain clustering.

    ARI measures agreement between latent k-means clusters and true spatial domains.
    Higher ARI (closer to 1) = better spatial domain identification.

    Parameters
    ----------
    latent : Tensor
        (n_spots, latent_dim) — ST latent embeddings (fused or unfused).
    spatial_labels : np.ndarray
        (n_spots,) — true spatial domain labels (e.g., from annotation).
    n_clusters : int, optional
        Number of k-means clusters. Default: len(np.unique(spatial_labels)).

    Returns
    -------
    float
        ARI score in [-1, 1], where 1 = perfect agreement.
    """
    X = latent.detach().cpu().numpy() if isinstance(latent, Tensor) else latent

    # Generate predicted clusters via k-means
    if n_clusters is None:
        n_clusters = len(np.unique(spatial_labels))

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels_pred = kmeans.fit_predict(X)

    return adjusted_rand_score(spatial_labels, labels_pred)


def compute_spatial_nmi(
    latent: Tensor,
    spatial_labels: np.ndarray,
    n_clusters: Optional[int] = None,
) -> float:
    """Compute Normalized Mutual Information for spatial domain clustering.

    NMI measures cluster-level alignment with true spatial domains.
    Higher NMI (closer to 1) = better spatial domain identification.

    Parameters
    ----------
    latent : Tensor
        (n_spots, latent_dim) — ST latent embeddings.
    spatial_labels : np.ndarray
        (n_spots,) — true spatial domain labels.
    n_clusters : int, optional
        Number of k-means clusters.

    Returns
    -------
    float
        NMI score in [0, 1].
    """
    X = latent.detach().cpu().numpy() if isinstance(latent, Tensor) else latent

    if n_clusters is None:
        n_clusters = len(np.unique(spatial_labels))

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels_pred = kmeans.fit_predict(X)

    return normalized_mutual_info_score(spatial_labels, labels_pred)


def _compute_nmi_from_latent(
    latent: Tensor,
    labels_true: np.ndarray,
    n_clusters: Optional[int] = None,
) -> float:
    """Compute NMI using k-means to generate predicted cluster labels.

    Parameters
    ----------
    latent : Tensor
        (n_samples, latent_dim) — latent representations.
    labels_true : np.ndarray
        (n_samples,) — true modality labels.
    n_clusters : int, optional
        Number of k-means clusters. Default: min(10, n_unique_labels).

    Returns
    -------
    float
        NMI score.
    """
    if isinstance(latent, Tensor):
        X = latent.detach().cpu().numpy()
    else:
        X = latent

    n_unique = len(np.unique(labels_true))
    if n_clusters is None:
        n_clusters = min(10, n_unique)

    # k-means clustering on latent space
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels_pred = kmeans.fit_predict(X)

    return normalized_mutual_info_score(labels_true, labels_pred)


def compute_gc(
    latent: Tensor,
    guidance_edge_index: Tensor,
    k: int = 5,
) -> Tensor:
    """Compute Graph Connectivity (GC) score.

    GC measures how well the latent space preserves guidance graph structure.
    GC = overlap coefficient between guidance graph adjacency and latent k-NN adjacency.
    Higher GC (closer to 1) = latent space better preserves guidance structure.

    Algorithm:
    1. Build k-NN graph from latent representations.
    2. Compute overlap: (# guidance edges present in latent k-NN) / (# guidance edges).

    Parameters
    ----------
    latent : Tensor
        (n_samples, latent_dim) — latent representations.
    guidance_edge_index : Tensor
        (2, n_edges) — edge index from guidance graph.
    k : int, default 5
        Number of nearest neighbors for latent k-NN graph.

    Returns
    -------
    Tensor
        Scalar GC score in [0, 1].
    """
    device = latent.device
    X = latent.detach().cpu().numpy()
    n_samples = X.shape[0]

    if guidance_edge_index.shape[0] != 2:
        raise ValueError("guidance_edge_index must be shape (2, n_edges)")

    # Build k-NN graph from latent representations
    k_actual = min(k, n_samples - 1)
    nn = NearestNeighbors(n_neighbors=k_actual + 1, algorithm="ball_tree")  # +1 includes self
    nn.fit(X)
    _, indices = nn.kneighbors(X)
    # Remove self-connections (first column)
    indices = indices[:, 1:]

    # Convert guidance edges to frozenset pairs for fast lookup
    edge_sets = set()
    edge_index_np = guidance_edge_index.cpu().numpy()
    for i in range(edge_index_np.shape[1]):
        src, dst = edge_index_np[0, i], edge_index_np[1, i]
        edge_sets.add(frozenset([src, dst]))

    # Count guidance edges preserved in latent k-NN
    preserved = 0
    total = len(edge_sets)

    if total == 0:
        return torch.tensor(1.0, device=device)  # No guidance edges = trivially preserved

    for i in range(n_samples):
        neighbors = indices[i]
        for nbr in neighbors:
            if frozenset([i, nbr]) in edge_sets:
                preserved += 1

    # Divide by 2 because each edge appears twice in undirected k-NN (i->j and j->i)
    # But guidance edges are undirected too, so we count each once
    # Actually preserved counts each undirected edge once from the perspective of source node
    # So we need to compare properly
    gc_score = preserved / total

    return torch.tensor(gc_score, device=device)


@dataclass
class AlignmentMetrics:
    """Alignment metrics dataclass for logging during training.

    Attributes
    ----------
    asw : float
        Average Silhouette Width (modality separation).
    nmi : float
        Normalized Mutual Information (cluster-level alignment).
    gc : float
        Graph Connectivity (guidance graph preservation).
    epoch : int
        Training epoch these metrics correspond to.
    """

    asw: float
    nmi: float
    gc: float
    epoch: int


def evaluate_alignment(
    vae,
    dataloader,
    guidance_data,
    device: str = "cuda",
    latent_key: str = "u_st",
) -> AlignmentMetrics:
    """Evaluate alignment metrics on validation data.

    Parameters
    ----------
    vae : TripleModalVAE
        VAE model to extract latent representations from.
    dataloader : DataLoader
        Validation dataloader yielding batches with x_sc, x_st, x_bulk.
    guidance_data : GuidanceGraph or Data
        Guidance graph for GC computation.
    device : str, default 'cuda'
        Device for computation.
    latent_key : str, default 'u_st'
        Key in VAE forward output to use for latent representation.

    Returns
    -------
    AlignmentMetrics
        Dataclass with asw, nmi, gc, epoch=0 (epoch should be set by caller).
    """
    vae.eval()
    all_latents = []
    all_modality_labels = []

    modality_counter = 0

    with torch.no_grad():
        for batch in dataloader:
            x_sc = batch.get("x_sc")
            x_st = batch.get("x_st")
            x_bulk = batch.get("x_bulk")

            # Get latent representations
            if x_sc is not None:
                x_sc = x_sc.to(device)
                # Encode just scRNA
                z_sc, _, _ = vae.enc_sc(x_sc)
                all_latents.append(z_sc.cpu())
                all_modality_labels.extend([0] * z_sc.shape[0])
                modality_counter += 1

            if x_st is not None:
                x_st = x_st.to(device)
                z_st, _, _ = vae.enc_st(x_st)
                all_latents.append(z_st.cpu())
                all_modality_labels.extend([1] * z_st.shape[0])

            if x_bulk is not None:
                x_bulk = x_bulk.to(device)
                u_bulk = vae.enc_bulk(x_bulk)
                all_latents.append(u_bulk.cpu())
                all_modality_labels.extend([2] * u_bulk.shape[0])

    # Concatenate all latents
    if not all_latents:
        raise ValueError("No latent representations collected from dataloader.")

    latent = torch.cat(all_latents, dim=0)
    modality_labels = np.array(all_modality_labels)

    # Get guidance edge index
    if hasattr(guidance_data, "edge_index"):
        guidance_edge_index = guidance_data.edge_index
    else:
        guidance_edge_index = getattr(guidance_data, "edge_index", None)

    # Compute metrics
    asw_score = compute_asw(latent, modality_labels=torch.from_numpy(modality_labels))
    nmi_score = _compute_nmi_from_latent(latent, modality_labels)
    gc_score = compute_gc(latent, guidance_edge_index) if guidance_edge_index is not None else torch.tensor(0.0)

    return AlignmentMetrics(
        asw=asw_score.item() if torch.is_tensor(asw_score) else float(asw_score),
        nmi=float(nmi_score),
        gc=gc_score.item() if torch.is_tensor(gc_score) else float(gc_score),
        epoch=0,  # Caller should set epoch
    )


def k_sweep_validation(
    trainer,
    dataloader,
    st_adata,
    spatial_labels: np.ndarray,
    k_values: list = [4, 6, 8, 12],
    device: str = "cuda",
) -> dict:
    """Run k-sweep validation over spatial k-NN graph construction.

    For each k in k_values:
    1. Build spatial k-NN graph via squidpy (using build_spatial_knn)
    2. Set spatial graph on trainer.spatial_scaffold
    3. Collect fused_st embeddings from trainer forward pass
    4. Compute spatial ARI and NMI

    Parameters
    ----------
    trainer : TripleModalTrainer
        Trainer with integrated SpatialScaffold.
    dataloader : DataLoader
        Validation dataloader.
    st_adata : ann.AnnData
        ST AnnData with .obsm['spatial'] coordinates.
    spatial_labels : np.ndarray
        (n_spots,) — true spatial domain labels.
    k_values : list, default [4, 6, 8, 12]
        k values to sweep (D-08).
    device : str, default 'cuda'

    Returns
    -------
    dict
        {k: {'ari': float, 'nmi': float, 'n_clusters': int}} for each k.
    """
    from ..scaffold.spatial_scaffold import build_spatial_knn

    results = {}
    n_clusters = len(np.unique(spatial_labels))

    for k in k_values:
        # Build spatial k-NN graph with this k
        adj, coords = build_spatial_knn(st_adata, n_neighbors=k)

        # Set spatial graph on scaffold
        trainer.spatial_scaffold.set_spatial_graph(adj)

        # Collect fused embeddings
        trainer.vae.eval()
        fused_embeddings = []
        with torch.no_grad():
            for batch in dataloader:
                x_st = batch.get('x_st')
                if x_st is None:
                    continue
                x_st = x_st.to(device)
                vae_out = trainer.vae(x_st, torch.zeros_like(x_st), torch.zeros_like(x_st), None)
                fused_st = trainer.spatial_scaffold(vae_out['u_st'])
                fused_embeddings.append(fused_st.cpu())

        # Concatenate all embeddings
        fused_latent = torch.cat(fused_embeddings, dim=0)

        # Truncate to match spatial_labels length if needed
        if len(fused_latent) > len(spatial_labels):
            fused_latent = fused_latent[:len(spatial_labels)]

        # Compute metrics
        ari = compute_spatial_ari(fused_latent, spatial_labels, n_clusters=n_clusters)
        nmi = compute_spatial_nmi(fused_latent, spatial_labels, n_clusters=n_clusters)

        results[k] = {
            'ari': ari,
            'nmi': nmi,
            'n_clusters': n_clusters,
        }
        print(f"k={k}: ARI={ari:.4f}, NMI={nmi:.4f}")

    return results


def evaluate_spatial_domains(
    latent: Tensor,
    spatial_labels: np.ndarray,
    n_clusters: Optional[int] = None,
) -> dict:
    """Evaluate spatial domain clustering quality.

    Parameters
    ----------
    latent : Tensor
        (n_spots, latent_dim) — ST embeddings (fused or unfused).
    spatial_labels : np.ndarray
        (n_spots,) — true spatial domain labels.
    n_clusters : int, optional
        Number of k-means clusters.

    Returns
    -------
    dict
        {'ari': float, 'nmi': float, 'n_clusters': int}
    """
    if n_clusters is None:
        n_clusters = len(np.unique(spatial_labels))

    ari = compute_spatial_ari(latent, spatial_labels, n_clusters=n_clusters)
    nmi = compute_spatial_nmi(latent, spatial_labels, n_clusters=n_clusters)

    return {
        'ari': ari,
        'nmi': nmi,
        'n_clusters': n_clusters,
    }


def log_metrics(
    metrics: AlignmentMetrics,
    logger,
    prefix: str = "alignment/",
) -> None:
    """Log alignment metrics to logger (TensorBoard or wandb).

    Parameters
    ----------
    metrics : AlignmentMetrics
        Metrics dataclass to log.
    logger : object
        Logger with .log() or .add_scalar() method.
        Supports:
        - torch.utils.tensorboard.SummaryWriter
        - wandb.sdk.wandb_run.Run
    prefix : str, default 'alignment/'
        Prefix for metric keys.
    """
    step = metrics.epoch
    asw = metrics.asw
    nmi = metrics.nmi
    gc = metrics.gc

    # Try TensorBoard style logging
    try:
        logger.add_scalar(f"{prefix}asw", asw, step)
        logger.add_scalar(f"{prefix}nmi", nmi, step)
        logger.add_scalar(f"{prefix}gc", gc, step)
        return
    except (AttributeError, TypeError, RuntimeError):  # SAFE-01: Logger API mismatch or backend error
        pass

    # Try wandb style logging
    try:
        logger.log({f"{prefix}asw": asw, f"{prefix}nmi": nmi, f"{prefix}gc": gc, "epoch": step})
        return
    except (AttributeError, TypeError, RuntimeError):  # SAFE-01: Logger API mismatch or backend error
        pass

    # Fallback: print
    print(f"[Epoch {step}] {prefix}asw={asw:.4f}, {prefix}nmi={nmi:.4f}, {prefix}gc={gc:.4f}")
