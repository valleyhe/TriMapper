"""HE histopathology overlay visualization for spatial results.

Overlays spatial domain segmentation, cell type deconvolution proportions,
and SC-ST mapping density onto per-sample HE (hematoxylin-eosin) images.

Coordinate alignment:
    Uses similarity transform (flip/swap + rotation + uniform scale + translate).
    Visium samples (N001, N002): coords already in HE pixel space, swap only.
    StereoSeq samples (R001, R003, R006): automated registration using
    mutual-information-based optimization on downsampled rasterized spot density
    vs HE tissue image (differential evolution + local Nelder-Mead refinement).

All spatial plots use matplotlib scatter (not scanpy spatial), per D-02.
All functions call set_publication_style() at start.
All functions return the output Path.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from ..visualization.publication_quality import set_publication_style

logger = logging.getLogger(__name__)

# Per-sample similarity transform parameters.
# Format: (pre_transform, rotation_deg, scale, tx_frac, ty_frac)
#
# pre_transform: axis mapping applied before rotation:
#   "swap"      -> (y, x)   i.e. spatial_x->HE_y, spatial_y->HE_x
#   "flip_y"    -> (x, -y)  vertical flip
#   "flip_x"    -> (-x, y)  horizontal flip
#   "flip_both" -> (-x, -y) 180° rotation (= flip_x + flip_y)
#   "none"      -> (x, y)   identity
#
# Scale and translation are for max_dim=4000 downsampled HE images.
# Determined via automated MI-based registration (differential evolution
# on downsampled rasterized spot density vs HE tissue image).
_SIMILARITY_PARAMS = {
    "R001": ("flip_y", 224.859696, 0.186445, 0.575043, 0.391946),   # MI=0.506
    "R003": ("flip_x", 86.415150, 0.243981, 0.502084, 0.497789),    # MI=0.409
    "R006": ("swap", 0.861044, 0.324180, 0.415289, 0.480462),       # MI=0.470
    "N001": ("swap", 0, 1.186508, 0.470139, 0.516265),
    "N002": ("swap", 0, 1.181702, 0.492434, 0.427765),
}


def _apply_pre_transform(x: np.ndarray, y: np.ndarray, name: str):
    """Apply axis swap/flip before rotation."""
    if name == "swap":
        return y, x
    elif name == "swap_fx":
        return -y, x
    elif name == "swap_fy":
        return y, -x
    elif name == "flip_x":
        return -x, y
    elif name == "flip_y":
        return x, -y
    elif name == "flip_both":
        return -x, -y
    return x, y


def get_he_path(sample_id: str, data_dir: Path) -> Path:
    """Resolve HE image file path for a given sample.

    Parameters
    ----------
    sample_id : str
        Sample identifier (R001, R003, R006, N001, N002).
    data_dir : Path
        Base directory containing HE TIF files.

    Returns
    -------
    Path
        Full path to the HE image file.

    Raises
    ------
    FileNotFoundError
        If the HE image file does not exist.
    """
    filename = f"{sample_id}_HE.tif" if sample_id.startswith("R") else f"{sample_id}.tif"
    he_path = data_dir / filename
    if not he_path.exists():
        raise FileNotFoundError(
            f"HE image not found for sample {sample_id}: {he_path}"
        )
    return he_path


def _load_he_downsampled(he_path: Path, max_dim: int = 4000) -> np.ndarray:
    """Load and downsample an HE image.

    Parameters
    ----------
    he_path : Path
        Path to the HE TIF image.
    max_dim : int
        Maximum dimension (pixels) for the longest axis.

    Returns
    -------
    np.ndarray
        Downsampled image as (H, W, 3) uint8 array.
    """
    old_max = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = 300_000_000  # Allow large TIF (~900MB RGB)
    try:
        img = Image.open(he_path)

        # Downsample if needed
        w, h = img.size
        if max(w, h) > max_dim:
            scale = max_dim / max(w, h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            img = img.resize((new_w, new_h), Image.LANCZOS)
            logger.info(f"Downsampled HE {he_path.name}: {w}x{h} -> {new_w}x{new_h}")

        # Convert to RGB numpy array
        if img.mode != "RGB":
            img = img.convert("RGB")
        return np.asarray(img)
    finally:
        Image.MAX_IMAGE_PIXELS = old_max


def align_coords_to_image(
    coords: np.ndarray,
    img_shape: tuple[int, int],
    sample_id: str,
) -> np.ndarray:
    """Align spatial coordinates to HE image pixel space via similarity transform.

    Applies: center → pre_transform (swap/flip) → rotate → uniform scale → translate.

    Parameters
    ----------
    coords : np.ndarray
        Raw spatial coordinates, shape (n_spots, 2).
    img_shape : tuple[int, int]
        HE image shape as (height, width).
    sample_id : str
        Sample identifier.

    Returns
    -------
    np.ndarray
        Pixel coordinates in image space, shape (n_spots, 2).
    """
    img_h, img_w = img_shape[:2]
    x = coords[:, 0].astype(float)
    y = coords[:, 1].astype(float)

    if sample_id not in _SIMILARITY_PARAMS:
        logger.warning(f"No alignment params for {sample_id}, using raw coords")
        return coords.astype(float)

    pre_name, rot_deg, scale, tx_frac, ty_frac = _SIMILARITY_PARAMS[sample_id]

    # Center coordinates
    cx, cy = x.mean(), y.mean()
    px, py = _apply_pre_transform(x - cx, y - cy, pre_name)

    # Apply rotation
    theta = np.radians(rot_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    rx = cos_t * px - sin_t * py
    ry = sin_t * px + cos_t * py

    # Apply uniform scale + translate
    img_x = scale * rx + tx_frac * img_w
    img_y = scale * ry + ty_frac * img_h

    return np.column_stack([img_x, img_y])


def _compute_spot_size(coords_img: np.ndarray, n_spots: int) -> float:
    """Compute spot size for scatter plot based on nearest-neighbor distance.

    Parameters
    ----------
    coords_img : np.ndarray
        Pixel coordinates in image space, shape (n_spots, 2).
    n_spots : int
        Number of spots (for subsampling if very large).

    Returns
    -------
    float
        Matplotlib scatter marker size.
    """
    from scipy.spatial import cKDTree

    # Subsample for speed if too many spots
    max_for_nn = 5000
    if n_spots > max_for_nn:
        idx = np.random.choice(n_spots, max_for_nn, replace=False)
        subset = coords_img[idx]
    else:
        subset = coords_img

    tree = cKDTree(subset)
    # k=2 gives nearest neighbor (first is self)
    dists, _ = tree.query(subset, k=2)
    avg_nn_dist = np.mean(dists[:, 1])

    return max(avg_nn_dist * 0.4, 1.0)


def plot_he_domain_overlay(
    st_adata,
    sample_id: str,
    data_dir: Path,
    output_dir: Path,
    he_max_dim: int = 4000,
) -> Path:
    """Plot spatial domains overlaid on HE image for one sample.

    Parameters
    ----------
    st_adata : sc.AnnData
        AnnData with obsm["spatial"], obs["domain"], obs["sample"].
    sample_id : str
        Sample identifier.
    data_dir : Path
        Directory containing HE TIF files.
    output_dir : Path
        Directory to save PDF.
    he_max_dim : int
        Maximum HE image dimension after downsampling.

    Returns
    -------
    Path
        Output file path.
    """
    set_publication_style()

    # Filter to sample
    mask = st_adata.obs["sample"] == sample_id
    if mask.sum() == 0:
        logger.warning(f"No spots for sample {sample_id}, skipping")
        return output_dir / f"he_domain_{sample_id}.pdf"

    coords = st_adata.obsm["spatial"][mask.values]
    domains = st_adata.obs["domain"].values[mask.values]
    domain_codes = pd.Categorical(domains.astype(str)).codes

    # Load and align
    he_path = get_he_path(sample_id, data_dir)
    he_img = _load_he_downsampled(he_path, he_max_dim)
    coords_img = align_coords_to_image(coords, he_img.shape, sample_id)
    spot_size = _compute_spot_size(coords_img, mask.sum())

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(he_img)
    scatter = ax.scatter(
        coords_img[:, 0], coords_img[:, 1],
        c=domain_codes, cmap="tab20",
        s=spot_size, alpha=0.6, rasterized=True,
    )
    ax.set_title(f"HE + Spatial Domains: {sample_id}", fontsize=14)
    ax.axis("off")
    plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04, label="Domain")

    out_path = output_dir / f"he_domain_{sample_id}.pdf"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {out_path}")
    return out_path


def plot_he_deconvolution_overlay(
    st_adata,
    sample_id: str,
    cell_type_names: list[str],
    data_dir: Path,
    output_dir: Path,
    he_max_dim: int = 4000,
    ncols: int = 4,
) -> Path:
    """Plot cell type deconvolution proportions overlaid on HE image.

    Parameters
    ----------
    st_adata : sc.AnnData
        AnnData with obsm["spatial"], obsm["cell_type_proportions"], obs["sample"].
    sample_id : str
        Sample identifier.
    cell_type_names : list[str]
        Cell type labels for subplot titles.
    data_dir : Path
        Directory containing HE TIF files.
    output_dir : Path
        Directory to save PDF.
    he_max_dim : int
        Maximum HE image dimension after downsampling.
    ncols : int
        Number of columns in subplot grid.

    Returns
    -------
    Path
        Output file path.
    """
    set_publication_style()

    # Filter to sample
    mask = st_adata.obs["sample"] == sample_id
    if mask.sum() == 0:
        logger.warning(f"No spots for sample {sample_id}, skipping deconvolution")
        return output_dir / f"he_deconv_{sample_id}.pdf"

    coords = st_adata.obsm["spatial"][mask.values]
    proportions = st_adata.obsm["cell_type_proportions"][mask.values]

    # Load and align
    he_path = get_he_path(sample_id, data_dir)
    he_img = _load_he_downsampled(he_path, he_max_dim)
    coords_img = align_coords_to_image(coords, he_img.shape, sample_id)
    spot_size = _compute_spot_size(coords_img, mask.sum())

    n_types = min(len(cell_type_names), 16)
    nrows = int(np.ceil(n_types / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 4))
    axes = axes.flatten()

    for i in range(n_types):
        ax = axes[i]
        ax.imshow(he_img)
        sc_plot = ax.scatter(
            coords_img[:, 0], coords_img[:, 1],
            c=proportions[:, i], cmap="Reds",
            s=spot_size, alpha=0.5, rasterized=True,
            vmin=0, vmax=proportions[:, i].max() if proportions[:, i].max() > 0 else 1,
        )
        ax.set_title(cell_type_names[i], fontsize=9)
        ax.axis("off")
        plt.colorbar(sc_plot, ax=ax, fraction=0.046, pad=0.04)

    for j in range(n_types, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    out_path = output_dir / f"he_deconv_{sample_id}.pdf"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {out_path}")
    return out_path


def plot_he_mapping_overlay(
    st_adata,
    sample_id: str,
    mapping_density: np.ndarray,
    data_dir: Path,
    output_dir: Path,
    he_max_dim: int = 4000,
) -> Path:
    """Plot SC-ST mapping density overlaid on HE image.

    Parameters
    ----------
    st_adata : sc.AnnData
        AnnData with obsm["spatial"], obs["sample"].
    sample_id : str
        Sample identifier.
    mapping_density : np.ndarray
        Per-spot mapping count array, shape (n_spots_for_sample,).
    data_dir : Path
        Directory containing HE TIF files.
    output_dir : Path
        Directory to save PDF.
    he_max_dim : int
        Maximum HE image dimension after downsampling.

    Returns
    -------
    Path
        Output file path.
    """
    set_publication_style()

    # Filter to sample
    mask = st_adata.obs["sample"] == sample_id
    if mask.sum() == 0:
        logger.warning(f"No spots for sample {sample_id}, skipping mapping overlay")
        return output_dir / f"he_mapping_{sample_id}.pdf"

    coords = st_adata.obsm["spatial"][mask.values]

    # Load and align
    he_path = get_he_path(sample_id, data_dir)
    he_img = _load_he_downsampled(he_path, he_max_dim)
    coords_img = align_coords_to_image(coords, he_img.shape, sample_id)
    spot_size = _compute_spot_size(coords_img, mask.sum())

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(he_img)
    scatter = ax.scatter(
        coords_img[:, 0], coords_img[:, 1],
        c=mapping_density, cmap="hot",
        s=spot_size, alpha=0.6, rasterized=True,
    )
    ax.set_title(f"HE + Mapping Density: {sample_id}", fontsize=14)
    ax.axis("off")
    plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04, label="Mapping count")

    out_path = output_dir / f"he_mapping_{sample_id}.pdf"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {out_path}")
    return out_path
