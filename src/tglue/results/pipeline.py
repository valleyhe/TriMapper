"""ResultsPipeline facade for triple-modal VAE result generation (D-04, D-05).

Chain API (D-05):
    pipeline = (
        ResultsPipeline()
        .from_checkpoint(checkpoint_path, spatial_path, scrna_path, bulk_path)
        .export_embeddings()
        .spatial_clustering(resolution=1.0)
        .export_deconvolution(epsilon=0.5, k_neighbors=50, chunk_size=5000)
        .export_mapping(topk=10)
        .he_overlay(data_dir="data/rosacea")
        .generate_report()
    )
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import scanpy as sc
import torch

from ..models.vae import TripleModalVAE
from ..models.vae_triple_efficient import TripleModalVAEEfficient
from ..deconv.label_mapping import get_canonical_cell_types, get_canonical_conditions
from .spatial_clustering import run_leiden
from .deconv_export import run_deconvolution
from .mapping_export import compute_mapping, compute_bulk_comparison
from .visualization import (
    plot_spatial_domains,
    plot_deconvolution_spatial,
    plot_mapping_density,
    plot_bulk_vs_predicted,
    generate_he_overlays,
)

logger = logging.getLogger(__name__)


class ResultsPipeline:
    """Facade for generating all result outputs from a trained checkpoint.

    Usage (D-05 chain pattern)::

        pipeline = (
            ResultsPipeline()
            .from_checkpoint(
                checkpoint_path, spatial_path, scrna_path, bulk_path,
                device="cuda", n_genes=17825, latent_dim=128,
            )
            .export_embeddings()
            .spatial_clustering(resolution=1.0)
            .export_deconvolution(epsilon=0.5, k_neighbors=50, chunk_size=5000)
            .export_mapping(topk=10)
            .he_overlay(data_dir="data/rosacea")
            .generate_report()
        )
    """

    def __init__(self):
        self.vae: Optional[TripleModalVAE | TripleModalVAEEfficient] = None
        self.st_adata: Optional[sc.AnnData] = None
        self.sc_adata: Optional[sc.AnnData] = None
        self.bulk_adata: Optional[sc.AnnData] = None
        self.output_dir: Optional[Path] = None
        self.fig_dir: Optional[Path] = None
        self.device: str = "cuda"
        self.spot_size: float = 1.0
        # Intermediate results
        self._u_sc: Optional[np.ndarray] = None
        self._u_st: Optional[np.ndarray] = None
        self._u_bulk: Optional[np.ndarray] = None
        self._proportions: Optional[np.ndarray] = None
        self._cell_types = None
        self._transport_result = None
        self._mapping_stats: Optional[dict] = None
        self._trans_matrix: Optional[np.ndarray] = None
        self._mapping_counts: Optional[np.ndarray] = None
        self._bulk_matrix: Optional[np.ndarray] = None
        self._pred_matrix: Optional[np.ndarray] = None
        self._report: Optional[dict] = None
        self._ckpt_epoch = None
        self._checkpoint_path = None
        self.st_out: Optional[sc.AnnData] = None

    def from_checkpoint(
        self,
        checkpoint_path: str,
        spatial_path: str,
        scrna_path: str,
        bulk_path: str,
        output_dir: str = "results/rosacea/full_outputs",
        device: str = "cuda",
        n_genes: int = 17_825,
        latent_dim: int = 128,
        spot_size: float = 1.0,
        enc_sc_hidden: int = 256,
        enc_st_hidden: int = 256,
        enc_bulk_hidden: int = 128,
        vae_type: str = "auto",
    ) -> "ResultsPipeline":
        """Load checkpoint and data. First step in chain (D-05).

        Parameters
        ----------
        checkpoint_path : str
            Path to best_model.pt checkpoint.
        spatial_path : str
            Path to spatial .h5ad file.
        scrna_path : str
            Path to scRNA .h5ad file.
        bulk_path : str
            Path to bulk .h5ad file.
        output_dir : str
        device : str
        n_genes : int
        latent_dim : int
        spot_size : float
        enc_sc_hidden, enc_st_hidden, enc_bulk_hidden : int
        vae_type : str
            "auto" (detect from checkpoint), "efficient" (TripleModalVAEEfficient),
            or "full" (TripleModalVAE with graph).
            Output directory (D-03: flat + figures/ subdir).
        device : str
            Torch device.
        n_genes : int
            Number of genes for model reconstruction.
        latent_dim : int
            Latent dimension for model reconstruction.
        spot_size : float
            Default spot size for scatter plots.
        enc_sc_hidden, enc_st_hidden, enc_bulk_hidden : int
            Hidden dimensions for each encoder.

        Returns
        -------
        self
        """
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA unavailable, using CPU")
            device = "cpu"
        self.device = device
        self.spot_size = spot_size

        # Output dirs (D-03: flat + figures/ subdir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fig_dir = self.output_dir / "figures"
        self.fig_dir.mkdir(exist_ok=True)

        # Load checkpoint (T-16-03: weights_only=True)
        ckpt = torch.load(checkpoint_path, weights_only=True)

        # Auto-detect VAE type from checkpoint keys
        if vae_type == "auto":
            has_graph = any("graph_enc" in k for k in ckpt["vae_state_dict"])
            vae_type = "efficient" if not has_graph else "full"
            logger.info(f"Auto-detected VAE type: {vae_type}")

        if vae_type == "efficient":
            # TripleModalVAEEfficient: hidden_dim used for all encoders
            hidden_dim = enc_sc_hidden  # Use enc_sc_hidden as unified hidden_dim
            self.vae = TripleModalVAEEfficient(
                n_genes=n_genes, latent_dim=latent_dim, hidden_dim=hidden_dim
            )
        else:
            # Full TripleModalVAE with graph
            self.vae = TripleModalVAE(
                n_genes=n_genes, latent_dim=latent_dim,
                enc_sc_hidden=enc_sc_hidden, enc_st_hidden=enc_st_hidden,
                enc_bulk_hidden=enc_bulk_hidden,
            )

        self.vae.load_state_dict(ckpt["vae_state_dict"])
        self.vae = self.vae.to(device).eval()
        self._ckpt_epoch = ckpt.get("epoch")
        self._checkpoint_path = checkpoint_path

        # Load data (D-01: np.asarray on spatial)
        self.st_adata = sc.read_h5ad(spatial_path)
        self.st_adata.obsm["spatial"] = np.asarray(self.st_adata.obsm["spatial"])
        self.sc_adata = sc.read_h5ad(scrna_path)
        self.bulk_adata = sc.read_h5ad(bulk_path)

        logger.info(f"Loaded checkpoint epoch {self._ckpt_epoch}")
        logger.info(
            f"ST: {self.st_adata.n_obs}, "
            f"scRNA: {self.sc_adata.n_obs}, "
            f"Bulk: {self.bulk_adata.n_obs}",
        )
        return self

    def export_embeddings(self, batch_size: int = 512) -> "ResultsPipeline":
        """Encode and save latent embeddings for all 3 modalities.

        Parameters
        ----------
        batch_size : int
            Batch size for encoding (T-16-04: prevents OOM).

        Returns
        -------
        self
        """
        self._assert_loaded()
        # Encode
        self._u_sc = self._encode(self.sc_adata.X, "sc", batch_size)
        self._u_st = self._encode(self.st_adata.X, "st", batch_size)
        self._u_bulk = self._encode_bulk(self.bulk_adata.X)

        # Save to h5ad
        sc_out = self.sc_adata.copy()
        sc_out.obsm["X_embedding"] = self._u_sc
        sc_out.write(self.output_dir / "scrna_with_embeddings.h5ad")

        st_out = self.st_adata.copy()
        st_out.obsm["X_embedding"] = self._u_st
        self.st_out = st_out  # Keep for downstream steps

        bulk_out = self.bulk_adata.copy()
        bulk_out.obsm["X_embedding"] = self._u_bulk
        bulk_out.write(self.output_dir / "bulk_with_embeddings.h5ad")

        logger.info(
            f"Saved embeddings: sc {self._u_sc.shape}, "
            f"st {self._u_st.shape}, bulk {self._u_bulk.shape}",
        )
        return self

    def spatial_clustering(
        self, resolution: float = 1.0, use_spatial_neighbors: bool = True
    ) -> "ResultsPipeline":
        """Run Leiden spatial domain segmentation.

        Parameters
        ----------
        resolution : float
            Leiden resolution parameter.
        use_spatial_neighbors : bool
            If True, use spatial coordinates for k-NN graph (much faster for large datasets).
            If False, use latent embeddings.

        Returns
        -------
        self
        """
        self._assert_step("export_embeddings")
        self.st_out = run_leiden(
            self.st_out, resolution=resolution, use_spatial_neighbors=use_spatial_neighbors
        )
        plot_spatial_domains(self.st_out, self.fig_dir, spot_size=self.spot_size)
        return self

    def export_deconvolution(
        self,
        epsilon: float = 0.5,
        k_neighbors: int = 50,
        chunk_size: int = 5000,
    ) -> "ResultsPipeline":
        """Run OT deconvolution and export cell type proportions.

        Parameters
        ----------
        epsilon : float
            OT entropy regularization.
        k_neighbors : int
            k-NN pre-filtering neighbors.
        chunk_size : int
            Chunk size for solve_chunked().

        Returns
        -------
        self
        """
        self._assert_step("export_embeddings")
        proportions, cell_types, transport_result = run_deconvolution(
            self._u_st, self._u_sc, self.sc_adata, self.device,
            epsilon=epsilon, k_neighbors=k_neighbors, chunk_size=chunk_size,
        )
        self._proportions = proportions
        self._cell_types = cell_types
        self._transport_result = transport_result

        # Store in st_out
        self.st_out.obsm["cell_type_proportions"] = proportions
        plot_deconvolution_spatial(
            self.st_out, proportions, cell_types.names, self.fig_dir,
            spot_size=self.spot_size,
        )
        return self

    def export_mapping(self, topk: int = 10) -> "ResultsPipeline":
        """Compute and export SC<->ST mapping matrix.

        Parameters
        ----------
        topk : int
            Number of top mappings per spot.

        Returns
        -------
        self
        """
        self._assert_step("export_deconvolution")
        trans_matrix, stats, mapping_counts = compute_mapping(
            self._transport_result.plan, topk=topk,
        )
        self._trans_matrix = trans_matrix
        self._mapping_stats = stats
        self._mapping_counts = mapping_counts
        self.st_out.obsm["trans_matrix"] = trans_matrix
        plot_mapping_density(mapping_counts, self.fig_dir)
        return self

    def he_overlay(
        self,
        data_dir: str = "data/rosacea",
        he_max_dim: int = 4000,
    ) -> "ResultsPipeline":
        """Generate HE overlay figures for all spatial samples.

        Overlays spatial domains, cell type deconvolution proportions,
        and mapping density onto per-sample HE histopathology images.

        Parameters
        ----------
        data_dir : str
            Directory containing HE TIF files (one per sample).
        he_max_dim : int
            Maximum HE image dimension after downsampling.

        Returns
        -------
        self
        """
        self._assert_step("export_deconvolution")

        # Compute per-spot mapping density from mapping counts
        if self._mapping_counts is not None:
            # mapping_counts is per-cell (scRNA); need per-spot (ST)
            # Use trans_matrix to aggregate: sum across cells for each spot
            if self._trans_matrix is not None:
                mapping_density = np.array(self._trans_matrix.sum(axis=1)).flatten()
            else:
                mapping_density = np.zeros(self.st_out.n_obs)
        else:
            mapping_density = None

        generate_he_overlays(
            self.st_out,
            Path(data_dir),
            self.fig_dir,
            mapping_density=mapping_density,
            he_max_dim=he_max_dim,
        )
        return self

    def generate_report(self) -> "ResultsPipeline":
        """Generate summary report (JSON + Markdown) and save final h5ad.

        Returns
        -------
        self
        """
        self._assert_step("export_mapping")

        # Bulk comparison
        conditions = get_canonical_conditions()
        bulk_matrix, pred_matrix, bulk_ct_names = compute_bulk_comparison(
            self.st_adata, self.bulk_adata, self._proportions, conditions,
        )
        self._bulk_matrix = bulk_matrix
        self._pred_matrix = pred_matrix
        plot_bulk_vs_predicted(
            bulk_matrix, pred_matrix, bulk_ct_names,
            conditions.names, self.fig_dir,
        )

        # Save final st with all results
        self.st_out.write(self.output_dir / "st_full_results.h5ad")
        torch.save(
            self._transport_result.plan.cpu(),
            self.output_dir / "transport_plan_sparse.pt",
        )

        # Build report
        self._report = {
            "timestamp": datetime.now().isoformat(),
            "checkpoint": self._checkpoint_path,
            "checkpoint_epoch": self._ckpt_epoch,
            "data": {
                "n_spots": self.st_adata.n_obs,
                "n_cells": self.sc_adata.n_obs,
                "n_bulk": self.bulk_adata.n_obs,
                "n_genes": self.st_adata.n_vars,
            },
            "spatial_domains": {
                "n_domains": int(self.st_out.obs["domain"].nunique()),
            },
            "deconvolution": {
                "n_cell_types": len(self._cell_types.names),
                "ot_convergence": self._transport_result.convergence_passed,
                "ot_marginal_error": self._transport_result.marginal_error,
            },
            "mapping": self._mapping_stats,
            "output_files": sorted(
                str(p) for p in self.output_dir.rglob("*") if p.is_file()
            ),
        }

        with open(self.output_dir / "report.json", "w") as f:
            json.dump(self._report, f, indent=2, default=str)

        # Markdown report
        md_lines = self._build_markdown_report()
        with open(self.output_dir / "report.md", "w") as f:
            f.write("\n".join(md_lines))

        logger.info(f"Report saved to {self.output_dir / 'report.md'}")
        return self

    # --- Private helpers ---

    def _assert_loaded(self):
        if self.vae is None:
            raise RuntimeError("Call from_checkpoint() first")

    def _assert_step(self, required: str):
        self._assert_loaded()
        checks = {
            "export_embeddings": lambda: self._u_st is not None,
            "export_deconvolution": lambda: self._proportions is not None,
            "export_mapping": lambda: self._trans_matrix is not None,
        }
        if not checks[required]():
            raise RuntimeError(f"Call {required}() before this step")

    def _encode(self, X, encoder_type, batch_size):
        """Encode expression to latent mean (numpy)."""
        self.vae.eval()
        encoder = self.vae.enc_st if encoder_type == "st" else self.vae.enc_sc
        n_obs = X.shape[0]
        u_all = []
        with torch.no_grad():
            for i in range(0, n_obs, batch_size):
                batch = X[i:i + batch_size]
                if hasattr(batch, "toarray"):
                    batch = batch.toarray()
                t = torch.tensor(batch, dtype=torch.float32).to(self.device)
                z, mean, log_var = encoder(t)
                u_all.append(mean.cpu().numpy())
        return np.concatenate(u_all, axis=0)

    def _encode_bulk(self, X):
        """Encode bulk expression via VanillaDataEncoder."""
        self.vae.eval()
        with torch.no_grad():
            if hasattr(X, "toarray"):
                X = X.toarray()
            t = torch.tensor(X, dtype=torch.float32).to(self.device)
            mean = self.vae.enc_bulk(t)
            return mean.cpu().numpy()

    def _build_markdown_report(self):
        r = self._report
        lines = [
            "# Triple-Modal VAE -- Full Results Report",
            "",
            f"**Generated:** {r['timestamp']}",
            f"**Checkpoint:** {r['checkpoint']} (epoch {r.get('checkpoint_epoch', '?')})",
            "",
            "## Data",
            "| Modality | Observations |",
            "|----------|-------------|",
            f"| ST (spatial) | {r['data']['n_spots']:,} |",
            f"| scRNA | {r['data']['n_cells']:,} |",
            f"| Bulk | {r['data']['n_bulk']} |",
            "",
            "## Spatial Domains",
            f"- Domains found: {r['spatial_domains']['n_domains']}",
            "",
            "## Deconvolution",
            f"- Cell types: {r['deconvolution']['n_cell_types']}",
            f"- OT convergence: {r['deconvolution']['ot_convergence']}",
            f"- Marginal error: {r['deconvolution']['ot_marginal_error']:.6f}",
            "",
            "## SC<->ST Mapping",
            f"- Top-k per spot: {r['mapping']['topk']}",
            f"- Mean spots per cell: {r['mapping']['mean_spots_per_cell']:.2f}",
            f"- Cells with no mapping: {r['mapping']['cells_with_zero_mapping']}",
            "",
            "## HE Overlay Figures",
            "HE histopathology overlays (domain, deconvolution, mapping density) "
            "are in figures/ with prefix `he_`. "
            "Generate via `.he_overlay(data_dir='data/rosacea')`.",
            "",
            "## Output Files",
        ]
        for f in r["output_files"]:
            rel = Path(f).relative_to(self.output_dir)
            lines.append(f"- `{rel}`")
        return lines
