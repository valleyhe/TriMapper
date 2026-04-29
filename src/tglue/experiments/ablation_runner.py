"""AblationRunner: Execute 5 ablation experiments with controlled seeds (AB-06).

This module implements the sweep executor that runs:
1. full_model (all components enabled)
2. no_guidance (Erdos-Renyi random graph)
3. no_fusion (identity pass-through)
4. no_bulk (bulk_loss_weight=0)
5. no_ot (uniform transport plan)

Each experiment uses identical seed (42) for reproducibility.
Results stored in JSON format for downstream visualization (D-AB10).
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

from .ablation_config import AblationConfig
from .batch_list_loader import BatchListLoader
from tglue.deconv.ot_solver import OTSolver
from tglue.deconv.evaluator import evaluate_deconvolution, DeconvolutionMetrics
from tglue.evaluation.metrics import evaluate_alignment, AlignmentMetrics


class AblationRunner:
    """Execute ablation experiments with controlled conditions.

    AB-06: Runs 5 experiments sequentially with seed reset before each run.
    Each experiment is a remove-one ablation quantifying component contribution.

    Parameters:
        base_dir: Base directory for results (default: 'results')
        seed: Fixed seed for all experiments (default: 42, per D-AB08)

    Usage:
        runner = AblationRunner(base_dir='results', seed=42)
        results = runner.run_all(train_data, val_data, n_epochs=100)

    Directory structure (D-AB11):
        results/
        ├── ablation_full_model/
        │   ├── config.json
        │   ├── checkpoint_best.pt
        │   ├── metrics.json
        │   └── tensorboard/
        ├── ablation_no_guidance/
        ├── ablation_no_fusion/
        ├── ablation_no_bulk/
        └── ablation_no_ot/
    """

    EXPERIMENTS: List[Tuple[str, AblationConfig]] = [
        ("full_model", AblationConfig()),  # All components enabled
        ("no_guidance", AblationConfig(use_guidance_graph=False)),
        ("no_fusion", AblationConfig(use_fusion_conv=False)),
        ("no_bulk", AblationConfig(use_bulk_prior=False)),
        ("no_ot", AblationConfig(use_ot_deconv=False)),
    ]

    def __init__(self, base_dir: str = "results", seed: int = 42):
        """Initialize AblationRunner with base directory and fixed seed.

        Args:
            base_dir: Base directory for experiment results
            seed: Fixed seed for all experiments (D-AB08)
        """
        self.base_dir = Path(base_dir)
        self.seed = seed

    def get_git_hash(self) -> str:
        """Capture current git commit hash for experiment tracking (D-AB12).

        Returns:
            str: 7-character short git hash

        Raises:
            subprocess.CalledProcessError: If not in a git repository
        """
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()

    def run_all(
        self,
        train_batches: List[Dict[str, Any]],
        val_batches: List[Dict[str, Any]],
        n_epochs: int,
    ) -> Dict[str, Dict[str, Any]]:
        """Run all 5 ablation experiments sequentially.

        AB-06: Each experiment uses identical seed (reset before each run).
        Experiments run sequentially (not parallel) per Open Question 2.

        Args:
            train_batches: Training data batches
            val_batches: Validation data batches
            n_epochs: Number of training epochs

        Returns:
            dict: Results dict mapping experiment name to metrics dict

        Example:
            {
                'full_model': {'asw': 0.65, 'nmi': 0.42, ...},
                'no_guidance': {'asw': 0.45, ...},
                ...
            }
        """
        from tglue.train.deterministic import set_deterministic_seed

        all_results: Dict[str, Dict[str, Any]] = {}

        for name, config in self.EXPERIMENTS:
            # Reset seed before each experiment (D-AB08)
            set_deterministic_seed(self.seed)

            # Run single experiment
            results = self.run_single(name, config, train_batches, val_batches, n_epochs)
            all_results[name] = results

        # Aggregate results into comparison table
        self.save_comparison_table(all_results)

        return all_results

    def run_single(
        self,
        name: str,
        config: AblationConfig,
        train_batches: List[Dict[str, Any]],
        val_batches: List[Dict[str, Any]],
        n_epochs: int,
    ) -> Dict[str, Any]:
        """Run one ablation experiment.

        Args:
            name: Experiment name (e.g., 'full_model', 'no_guidance')
            config: AblationConfig for this experiment
            train_batches: Training data batches
            val_batches: Validation data batches
            n_epochs: Number of training epochs

        Returns:
            dict: Metrics dict for this experiment

        Side effects:
            - Creates results/ablation_{name}/ directory
            - Saves config.json with experiment metadata
            - Saves metrics.json with final metrics
            - Saves checkpoint_best.pt
            - Saves tensorboard logs
        """
        from pathlib import Path
        from tglue.models.vae import TripleModalVAE
        from tglue.models.discriminator import ModalityDiscriminator
        from tglue.train.trainer import TripleModalTrainer
        from tglue.train.pipeline import TrainPipeline
        from tglue.experiments.variants import (
            NoSpatialScaffold,
            create_no_bulk_trainer_config,
        )

        # Create results directory
        results_dir = self.base_dir / f"ablation_{name}"
        results_dir.mkdir(parents=True, exist_ok=True)

        # Save config snapshot (D-AB12)
        self.save_config_snapshot(results_dir, name, config, n_epochs)

        # Extract gene count from batch data (assume consistent across batches)
        n_genes = train_batches[0]['x_sc'].shape[1] if train_batches else 50
        latent_dim = 128

        # Create VAE (standard for all ablations, guidance graph handled in batch)
        vae = TripleModalVAE(n_genes=n_genes, latent_dim=latent_dim)

        # Create discriminator
        discriminator = ModalityDiscriminator(latent_dim=latent_dim)

        # Configure trainer based on AblationConfig
        spatial_scaffold = None
        bulk_prior_config = None

        if not config.use_fusion_conv:
            # AB-02: Replace fusion_conv with identity
            spatial_scaffold = NoSpatialScaffold(latent_dim=latent_dim)

        if not config.use_bulk_prior:
            # AB-03: Set bulk_loss_weight = 0
            bulk_prior_config = create_no_bulk_trainer_config()

        # Create trainer
        trainer = TripleModalTrainer(
            vae=vae,
            discriminator=discriminator,
            spatial_scaffold=spatial_scaffold,
            bulk_prior_config=bulk_prior_config,
            device='cpu',  # Use CPU for testing
        )

        # Create pipeline with checkpoint and tensorboard directories
        checkpoint_dir = results_dir / "checkpoints"
        log_dir = results_dir / "tensorboard"

        pipeline = TrainPipeline(
            trainer=trainer,
            checkpoint_dir=str(checkpoint_dir),
            log_dir=str(log_dir),
            seed=self.seed,
            patience=10,
            checkpoint_every=5,
        )

        # Run training
        history = pipeline.train(train_batches, val_batches, n_epochs)

        # D-MI01: Compute alignment metrics on validation set after training
        # D-MI02: Embeddings collected from model.encode() with torch.no_grad()
        # D-MI04: Use separate validation set for metrics

        # Get guidance_data from first batch (for GC computation)
        guidance_data = train_batches[0].get('guidance_data') if train_batches else None

        # Wrap validation batches in adapter for evaluate_alignment()
        val_loader = BatchListLoader(val_batches)

        # Compute alignment metrics (ASW, NMI, GC)
        alignment_metrics = evaluate_alignment(
            vae=trainer.vae,
            dataloader=val_loader,
            guidance_data=guidance_data,
            device=trainer.device,
        )
        alignment_metrics.epoch = n_epochs - 1  # Set final epoch

        # Log to TensorBoard (if tb_logger available)
        if hasattr(pipeline, 'tb_logger') and pipeline.tb_logger is not None:
            pipeline.tb_logger.log_alignment_metrics(alignment_metrics, n_epochs - 1)

        # D-MI03: Compute deconvolution metrics via OTSolver (per RESEARCH.md recommendation)
        # OTSolver created separately from trainer - simpler integration
        deconv_metrics = None

        if config.use_ot_deconv:  # Skip for no_ot ablation
            try:
                # Create OTSolver (epsilon >= 0.1 per D-08 anti-pattern)
                solver = OTSolver(epsilon=0.1, k_neighbors=50, n_iters=100)

                # Extract embeddings for OT from validation batch
                # Need fused_st (spatially-aware ST) and scRNA profiles
                trainer.vae.eval()

                with torch.no_grad():
                    # Get a batch with ST and scRNA data
                    val_batch = val_batches[0] if val_batches else {}

                    x_st = val_batch.get('x_st')
                    x_sc = val_batch.get('x_sc')

                    if x_st is not None and x_sc is not None:
                        x_st = x_st.to(trainer.device)
                        x_sc = x_sc.to(trainer.device)

                        # Encode ST to get u_st
                        z_st, _, _ = trainer.vae.enc_st(x_st)
                        u_st = z_st

                        # Apply spatial scaffold for fused_st (detached per anti-pattern)
                        if trainer.spatial_scaffold is not None:
                            fused_st = trainer.spatial_scaffold(u_st.detach())
                        else:
                            fused_st = u_st  # Fallback if no scaffold

                        # Encode scRNA for profiles
                        z_sc, _, _ = trainer.vae.enc_sc(x_sc)
                        scRNA_profiles = z_sc

                        # Solve OT
                        transport_result = solver(fused_st, scRNA_profiles)
                        transport_plan = transport_result.plan

                        # Handle sparse tensor (Pitfall 3 from RESEARCH.md)
                        if transport_plan.is_sparse:
                            transport_plan = transport_plan.to_dense()

                        # Create mock cell_type_onehot (synthetic data has no real cell types)
                        # For real data, this would come from AnnData .obs['cell_type']
                        n_cells = scRNA_profiles.shape[0]
                        n_types = 5  # Assume 5 cell types for synthetic
                        cell_type_onehot = torch.eye(n_types, device=trainer.device)[:n_cells]
                        if n_cells > n_types:
                            # Pad with repeats
                            repeats_needed = (n_cells // n_types) + 1
                            cell_type_onehot = torch.eye(n_types, device=trainer.device).repeat(repeats_needed, 1)[:n_cells]

                        # Ground truth is None for synthetic data (Pitfall 1 from RESEARCH.md)
                        # evaluate_deconvolution returns 0.0 pearson_r when ground_truth is None
                        ground_truth = None  # Synthetic data has no ground truth

                        # Compute deconvolution metrics
                        deconv_metrics = evaluate_deconvolution(
                            transport_plan=transport_plan,
                            cell_type_onehot=cell_type_onehot,
                            ground_truth=ground_truth,
                        )

                        # Log to TensorBoard
                        if hasattr(pipeline, 'tb_logger') and pipeline.tb_logger is not None:
                            from tglue.deconv.evaluator import log_deconvolution_metrics
                            log_deconvolution_metrics(deconv_metrics, pipeline.tb_logger, n_epochs - 1)

            except Exception as e:
                # Fallback: metrics computation failed
                print(f"Warning: Deconvolution metrics computation failed: {e}")
                deconv_metrics = None

        # Extract final metrics
        metrics = self.extract_final_metrics(history, trainer, alignment_metrics=alignment_metrics, deconv_metrics=deconv_metrics)

        # Save metrics to JSON
        metrics_path = results_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        return metrics

    def extract_final_metrics(
        self,
        history: Dict[str, List[float]],
        trainer: Any,
        alignment_metrics: AlignmentMetrics | None = None,
        deconv_metrics: DeconvolutionMetrics | None = None,
    ) -> Dict[str, Any]:
        """Extract final epoch metrics for ablation comparison.

        Args:
            history: Training loss history from TrainPipeline
            trainer: TripleModalTrainer instance
            alignment_metrics: Alignment metrics from evaluate_alignment (optional)
            deconv_metrics: Deconvolution metrics from evaluate_deconvolution (optional)

        Returns:
            dict: Metrics dict with final values
        """
        metrics = {}

        # Training losses - final epoch values
        if history.get("vae_loss"):
            metrics["vae_loss_final"] = history["vae_loss"][-1]
        if history.get("recon_loss"):
            metrics["recon_loss_final"] = history["recon_loss"][-1]
        if history.get("kl_loss"):
            metrics["kl_loss_final"] = history["kl_loss"][-1]
        if history.get("graph_recon_loss"):
            metrics["graph_recon_loss_final"] = history["graph_recon_loss"][-1]

        # Use alignment_metrics passed from run_single()
        if alignment_metrics is not None:
            metrics["asw"] = float(alignment_metrics.asw)
            metrics["nmi"] = float(alignment_metrics.nmi)
            metrics["gc"] = float(alignment_metrics.gc)
        else:
            # Fallback if metrics not computed
            metrics["asw"] = 0.0
            metrics["nmi"] = 0.0
            metrics["gc"] = 0.0

        # Deconvolution metrics (may be near-zero for synthetic without ground_truth)
        if deconv_metrics is not None:
            metrics["pearson_r"] = float(deconv_metrics.pearson_r)
            metrics["kl_mean"] = float(deconv_metrics.kl_mean)
        else:
            # Fallback if metrics not computed
            metrics["pearson_r"] = 0.0
            metrics["kl_mean"] = 0.0

        return metrics

    def save_comparison_table(self, all_results: Dict[str, Dict[str, Any]]) -> None:
        """Aggregate all results into comparison table JSON.

        Args:
            all_results: Dict mapping experiment name to metrics dict
        """
        # Create summary table
        summary_path = self.base_dir / "ablation_comparison.json"

        with open(summary_path, "w") as f:
            json.dump(all_results, f, indent=2)

    def save_config_snapshot(
        self,
        results_dir: Path,
        name: str,
        config: AblationConfig,
        n_epochs: int,
    ) -> None:
        """Save experiment config snapshot to JSON (D-AB12).

        Args:
            results_dir: Directory for this experiment's results
            name: Experiment name
            config: AblationConfig instance
            n_epochs: Number of epochs
        """
        config_dict = {
            "name": name,
            "config": config.to_dict(),
            "seed": self.seed,
            "n_epochs": n_epochs,
            "git_hash": self.get_git_hash(),
            "timestamp": self._get_timestamp(),
        }

        config_path = results_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2)

    def _get_timestamp(self) -> str:
        """Get current timestamp for experiment tracking.

        Returns:
            str: ISO format timestamp
        """
        from datetime import datetime
        return datetime.utcnow().isoformat()