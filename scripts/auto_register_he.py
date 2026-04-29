#!/usr/bin/env python3
"""Fast automated registration of StereoSeq spots to HE images.

Two-phase approach:
1. Image moments: compute centroid, orientation, scale of both spot density
   and tissue mask → derive analytical similarity transform
2. Local optimization (Nelder-Mead) around the moments-based estimate using
   normalized cross-correlation on a downsampled grid

Much faster than brute-force global search because moments give a good
initial estimate that local optimization can refine in ~50 evaluations.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from PIL import Image
from skimage import filters, morphology
from scipy import ndimage
from scipy.optimize import minimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scanpy as sc

DATA_DIR = Path("data/rosacea")
RESULTS_DIR = Path("results/rosacea/full_outputs_100k")
MAX_DIM = 4000
# Downsample factor for NCC computation (speed)
NCC_DS = 4

print("Loading ST data...", flush=True)
st = sc.read_h5ad(RESULTS_DIR / "st_full_results.h5ad")
st.obsm["spatial"] = np.asarray(st.obsm["spatial"])
print(f"  {st.n_obs} spots, samples: {st.obs['sample'].unique().tolist()}", flush=True)


def load_he(he_path, max_dim=MAX_DIM):
    old_max = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = 300_000_000
    try:
        img = Image.open(he_path)
        w, h = img.size
        if max(w, h) > max_dim:
            s = max_dim / max(w, h)
            img = img.resize((int(w * s), int(h * s)), Image.LANCZOS)
        if img.mode != "RGB":
            img = img.convert("RGB")
        return np.asarray(img)
    finally:
        Image.MAX_IMAGE_PIXELS = old_max


def get_he_path(sid):
    fn = f"{sid}_HE.tif" if sid.startswith("R") else f"{sid}.tif"
    return DATA_DIR / fn


def tissue_mask(img, sigma=10):
    """Smooth tissue probability mask."""
    gray = np.mean(img, axis=2).astype(np.uint8)
    threshold = filters.threshold_otsu(gray)
    binary = gray < threshold
    binary = morphology.remove_small_objects(binary, min_size=2000)
    binary = morphology.remove_small_holes(binary, area_threshold=2000)
    mask = ndimage.gaussian_filter(binary.astype(float), sigma=sigma)
    return mask / mask.max() if mask.max() > 0 else mask


def apply_pre(px, py, name):
    if name == "swap":
        return py.copy(), px.copy()
    elif name == "swap_fx":
        return -py.copy(), px.copy()
    elif name == "swap_fy":
        return py.copy(), -px.copy()
    elif name == "flip_x":
        return -px.copy(), py.copy()
    elif name == "flip_y":
        return px.copy(), -py.copy()
    elif name == "flip_both":
        return -px.copy(), -py.copy()
    return px.copy(), py.copy()


def image_moments(img):
    """Compute centroid, orientation, and scale from a 2D image."""
    m00 = img.sum()
    if m00 < 1e-10:
        return None
    h, w = img.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    cx = (xx * img).sum() / m00
    cy = (yy * img).sum() / m00
    dx = xx - cx
    dy = yy - cy
    mu20 = (dx ** 2 * img).sum() / m00
    mu11 = (dx * dy * img).sum() / m00
    mu02 = (dy ** 2 * img).sum() / m00
    theta = 0.5 * np.arctan2(2 * mu11, mu20 - mu02)
    scale = np.sqrt(mu20 + mu02)  # RMS spread
    return {"cx": cx, "cy": cy, "theta": theta, "scale": scale,
            "mu20": mu20, "mu11": mu11, "mu02": mu02}


def rasterize_spots_fast(x, y, shape, sigma=8):
    """Rasterize spots into a density image."""
    h, w = shape
    density = np.zeros((h, w), dtype=float)
    valid = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    ix = np.clip(x[valid].astype(int), 0, w - 1)
    iy = np.clip(y[valid].astype(int), 0, h - 1)
    np.add.at(density, (iy, ix), 1.0)
    density = ndimage.gaussian_filter(density, sigma=sigma)
    return density / density.max() if density.max() > 0 else density


def ncc(a, b):
    a, b = a - a.mean(), b - b.mean()
    d = np.sqrt((a ** 2).sum() * (b ** 2).sum())
    return (a * b).sum() / d if d > 1e-10 else -1.0


def register_sample(sample_id, coords, he_img, pre_name):
    """Register using moments + local Nelder-Mead optimization."""
    img_h, img_w = he_img.shape[:2]
    ds_h, ds_w = img_h // NCC_DS, img_w // NCC_DS

    # Phase 1: Moments-based initialization
    # Downsample tissue mask
    tmask_full = tissue_mask(he_img, sigma=10)
    tmask_ds = ndimage.zoom(tmask_full, (ds_h / img_h, ds_w / img_w), order=1)
    t_mom = image_moments(tmask_ds)

    # Center and pre-transform coordinates
    x = coords[:, 0].astype(float)
    y = coords[:, 1].astype(float)
    cx, cy = x.mean(), y.mean()
    px, py = apply_pre(x - cx, y - cy, pre_name)

    # Rasterize spots at base scale to compute moments
    px_range = px.max() - px.min()
    py_range = py.max() - py.min()
    base_scale = min(img_w * 0.8 / max(px_range, 1), img_h * 0.8 / max(py_range, 1))

    # Rasterize at base scale for moments
    rx = (px - px.mean()) * base_scale + img_w / 2
    ry = (py - py.mean()) * base_scale + img_h / 2
    # Downsample spot coordinates
    rx_ds = rx / NCC_DS
    ry_ds = ry / NCC_DS
    spot_ds = rasterize_spots_fast(rx_ds, ry_ds, (ds_h, ds_w), sigma=3)
    s_mom = image_moments(spot_ds)

    if s_mom is None or t_mom is None:
        return None

    # Derive initial transform from moments
    init_rot = np.degrees(t_mom["theta"] - s_mom["theta"])
    init_scale_mult = t_mom["scale"] / s_mom["scale"]
    init_tx = t_mom["cx"] - s_mom["cx"]
    init_ty = t_mom["cy"] - s_mom["cy"]

    # Convert tx/ty back to full-res fractions
    init_tx_frac = init_tx * NCC_DS / img_w
    init_ty_frac = init_ty * NCC_DS / img_h

    # Clamp to reasonable ranges
    init_rot = np.clip(init_rot, -180, 180)
    init_scale_mult = np.clip(init_scale_mult, 0.3, 3.0)
    init_tx_frac = np.clip(init_tx_frac, 0.05, 0.95)
    init_ty_frac = np.clip(init_ty_frac, 0.05, 0.95)

    print(f"    Moments init: rot={init_rot:.1f}°, scale×={init_scale_mult:.3f},"
          f" tx={init_tx_frac:.4f}, ty={init_ty_frac:.4f}", flush=True)

    # Phase 2: Local optimization using NCC on downsampled grid
    def objective(params):
        rot_deg, scale_mult, tx_frac, ty_frac = params
        scale = base_scale * scale_mult
        tx = tx_frac * img_w
        ty = ty_frac * img_h
        theta = np.radians(rot_deg)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        ix = scale * (cos_t * px - sin_t * py) + tx
        iy = scale * (sin_t * px + cos_t * py) + ty
        # Downsample
        ix_ds = ix / NCC_DS
        iy_ds = iy / NCC_DS
        density = rasterize_spots_fast(ix_ds, iy_ds, (ds_h, ds_w), sigma=3)
        return -ncc(density, tmask_ds)

    # Multi-start: try moments + small perturbations
    best_result = None
    best_ncc = -1

    starts = [
        [init_rot, init_scale_mult, init_tx_frac, init_ty_frac],
        [init_rot + 90, init_scale_mult, init_tx_frac, init_ty_frac],
        [init_rot - 90, init_scale_mult, init_tx_frac, init_ty_frac],
        [init_rot + 180, init_scale_mult, init_tx_frac, init_ty_frac],
        [init_rot, init_scale_mult * 1.5, init_tx_frac, init_ty_frac],
        [init_rot, init_scale_mult * 0.7, init_tx_frac, init_ty_frac],
    ]

    for i, x0 in enumerate(starts):
        try:
            result = minimize(
                objective, x0, method="Nelder-Mead",
                options={"maxiter": 300, "xatol": 0.1, "fatol": 1e-4},
            )
            cur_ncc = -result.fun
            if cur_ncc > best_ncc:
                best_ncc = cur_ncc
                best_result = result
        except Exception:
            pass

    if best_result is None:
        return None

    rot_opt, scale_mult_opt, tx_frac_opt, ty_frac_opt = best_result.x
    scale_opt = base_scale * np.clip(scale_mult_opt, 0.3, 3.0)

    return {
        "pre": pre_name,
        "rot": rot_opt,
        "scale": scale_opt,
        "tx_frac": tx_frac_opt,
        "ty_frac": ty_frac_opt,
        "ncc": best_ncc,
        "base_scale": base_scale,
    }


def plot_registration(sid, coords, he_img, params, output_dir):
    """Generate debug overlay image."""
    img_h, img_w = he_img.shape[:2]
    x = coords[:, 0].astype(float)
    y = coords[:, 1].astype(float)
    cx, cy = x.mean(), y.mean()
    centered = np.column_stack([x - cx, y - cy])

    px, py = apply_pre(centered[:, 0], centered[:, 1], params["pre"])
    theta = np.radians(params["rot"])
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    ix = params["scale"] * (cos_t * px - sin_t * py) + params["tx_frac"] * img_w
    iy = params["scale"] * (sin_t * px + cos_t * py) + params["ty_frac"] * img_h

    tmask = tissue_mask(he_img, sigma=10)

    fig, axes = plt.subplots(1, 3, figsize=(30, 10))

    axes[0].imshow(he_img)
    axes[0].imshow(tmask, cmap="Blues", alpha=0.4)
    axes[0].set_title(f"{sid}: HE + Tissue mask")
    axes[0].axis("off")

    spot_density = rasterize_spots_fast(ix, iy, he_img.shape[:2], sigma=8)
    axes[1].imshow(he_img)
    axes[1].imshow(spot_density, cmap="Reds", alpha=0.5)
    axes[1].set_title(f"{sid}: Spot density (NCC={params['ncc']:.4f})")
    axes[1].axis("off")

    axes[2].imshow(he_img)
    axes[2].scatter(ix, iy, s=1, alpha=0.3, c="red", rasterized=True)
    axes[2].set_title(
        f"{sid}: {params['pre']}+{params['rot']:.1f}° "
        f"s={params['scale']:.5f} tx={params['tx_frac']:.4f} ty={params['ty_frac']:.4f}"
    )
    axes[2].axis("off")

    plt.tight_layout()
    out_path = output_dir / f"auto_reg_{sid}.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}", flush=True)


# =========================================================================
# Register R-samples
# =========================================================================
PRE_TRANSFORMS = ["swap", "flip_y", "flip_both", "flip_x", "swap_fx", "swap_fy"]
R_SAMPLES = ["R001", "R003", "R006"]

debug_dir = RESULTS_DIR / "he_overlays" / "debug_similarity"
debug_dir.mkdir(parents=True, exist_ok=True)

best_results = {}

for sid in R_SAMPLES:
    print(f"\n{'=' * 60}", flush=True)
    print(f"  {sid}: Automated registration", flush=True)
    print(f"{'=' * 60}", flush=True)

    mask_s = st.obs["sample"] == sid
    coords = st.obsm["spatial"][mask_s.values]
    he = load_he(get_he_path(sid))
    print(f"  {mask_s.sum()} spots, HE: {he.shape}", flush=True)

    best_ncc = -1
    best_params = None

    for pre_name in PRE_TRANSFORMS:
        print(f"  Trying pre_transform={pre_name}...", flush=True)
        try:
            params = register_sample(sid, coords, he, pre_name)
            if params:
                print(f"    -> NCC={params['ncc']:.4f}, rot={params['rot']:.1f}°,"
                      f" scale={params['scale']:.5f}", flush=True)
                if params["ncc"] > best_ncc:
                    best_ncc = params["ncc"]
                    best_params = params
        except Exception as e:
            print(f"    Failed: {e}", flush=True)

    if best_params:
        best_results[sid] = best_params
        plot_registration(sid, coords, he, best_params, debug_dir)
    else:
        print(f"  {sid}: All pre-transforms failed!", flush=True)


# =========================================================================
# Print final parameters
# =========================================================================
print("\n\n" + "=" * 60, flush=True)
print("FINAL REGISTRATION PARAMETERS", flush=True)
print("=" * 60, flush=True)
print("_SIMILARITY_PARAMS = {", flush=True)
for sid, p in best_results.items():
    print(f'    "{sid}": ("{p["pre"]}", {p["rot"]:.6f}, {p["scale"]:.6f},'
          f' {p["tx_frac"]:.6f}, {p["ty_frac"]:.6f}),  # NCC={p["ncc"]:.4f}', flush=True)
print("}", flush=True)
