#!/usr/bin/env python3
"""Phase 12: TripleModalVAE training on Rosacea real data.

Executes triple-modal VAE training with:
- TripleModalVAE: scRNA (76K cells), ST (100K spots), Bulk (58 samples)
- Guidance graph: 346K edges (267K genomic + 79K co-expression)
- Checkpointing: every 5 epochs + best model
- Early stopping: patience=10 on validation VAE loss
- TensorBoard logging: loss curves and alignment metrics
- Deterministic seed: 42

Chunk 4: Condition-level bulk prior integration:
- Uses preprocess_bulk_with_metadata() for condition-level proportions
- Sets bulk_condition_proportions and bulk_condition_names on trainer
- Sets metadata_for_condition_prior (scRNA cell types, ST conditions)
- Computes epoch-level condition_prior_state via OT deconvolution

Usage:
    # Init only (Task 1):
    python scripts/train_rosacea_phase12.py --mode=init

    # Full training (Task 2):
    python scripts/train_rosacea_phase12.py --mode=train

    # Resume from checkpoint:
    python scripts/train_rosacea_phase12.py --mode=train --resume checkpoints/rosacea/best_model.pt
"""

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Ensure src is in path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tglue.data.rosacea_loader import load_rosacea_dataset
from tglue.deconv.bulk_prior import BulkPriorConfig
from tglue.models.discriminator import ModalityDiscriminator
from tglue.models.vae import TripleModalVAE
from tglue.preprocessing.ssgsea_bulk import preprocess_bulk_ssgsea
from tglue.scaffold.spatial_scaffold import SpatialScaffold
from tglue.train.pipeline import TrainPipeline
from tglue.train.trainer import TripleModalTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ============================================================================
# Configuration (D-02, D-03, D-08, D-11 locked decisions)
# ============================================================================
N_GENES = 17_825
LATENT_DIM = 128
PRE_WARM_EPOCHS = 20
OT_EPSILON = 0.1
LR = 1e-3
SEED = 42
N_EPOCHS = 200
PATIENCE = 10
CHECKPOINT_EVERY = 5
BATCH_SIZE = 128

# Directories
CHECKPOINT_DIR = Path("checkpoints/rosacea")
LOG_DIR = Path("runs/rosacea")
MARKERS_PATH = Path("src/tglue/data/markers/skin_markers.gmt")


def set_seed(seed: int = SEED) -> None:
    """Set deterministic seed for reproducibility (TR-04)."""
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)
    logger.info(f"Deterministic seed set to {seed}")


def prebuild_guidance_graph() -> "GuidanceGraph":
    """Pre-build guidance graph for Rosacea data.

    Loads only the data needed for graph construction (scRNA expression + GTF).
    The graph is then reused for both train and validation datasets.

    Returns
    -------
    GuidanceGraph
        Pre-built guidance graph ready for VAE consumption
    """
    import scanpy as sc
    from tglue.graph.guidance_graph import build_guidance_graph, GuidanceGraph

    # Check for cached graph
    cache_path = Path("data/rosacea/guidance_graph.pkl")
    if cache_path.exists():
        logger.info(f"Loading cached guidance graph from {cache_path}")
        graph = GuidanceGraph.load(str(cache_path))
        logger.info(f"Loaded cached graph: {graph}")
        return graph

    logger.info("Pre-building guidance graph...")
    scRNA_adata = sc.read_h5ad("data/rosacea/sc_reference.h5ad")
    st_adata = sc.read_h5ad("data/rosacea/spatial_100k.h5ad")
    bulk_adata = sc.read_h5ad("data/rosacea/array_test.h5ad")
    logger.info(
        f"Loaded data for graph: scRNA={scRNA_adata.n_obs} cells, "
        f"ST={st_adata.n_obs} spots, Bulk={bulk_adata.n_obs} samples"
    )

    graph = build_guidance_graph(
        scRNA_adata=scRNA_adata,
        st_adata=st_adata,
        bulk_adata=bulk_adata,
        gtf_path="data/rosacea/mock_gtf.txt",
        genomic_window_bp=150_000,
        coexpr_threshold=0.3,
    )
    logger.info(f"Guidance graph built: {graph}")

    # Save to cache
    logger.info(f"Caching guidance graph to {cache_path}")
    graph.save(str(cache_path))
    return graph


def create_models(device: str) -> tuple:
    """Initialize VAE, discriminator, spatial scaffold, bulk config, trainer.

    Parameters
    ----------
    device : str
        Device for model placement ('cuda' or 'cpu')

    Returns
    -------
    tuple
        (vae, discriminator, trainer, train_dataset, val_dataset)
    """
    # TripleModalVAE (D-02: latent_dim=128)
    logger.info(f"Initializing TripleModalVAE: n_genes={N_GENES}, latent_dim={LATENT_DIM}")
    vae = TripleModalVAE(
        n_genes=N_GENES,
        latent_dim=LATENT_DIM,
        enc_sc_hidden=256,
        enc_st_hidden=256,
        enc_bulk_hidden=128,
    )

    # Modality Discriminator (D-03: adversarial post-warm)
    discriminator = ModalityDiscriminator(latent_dim=LATENT_DIM, hidden_dim=256)

    # Spatial Scaffold (D-05: fusion_conv, 2-layer MLP)
    spatial_scaffold = SpatialScaffold(latent_dim=LATENT_DIM, n_neighbors=6)

    # Bulk Prior Config (D-11: lambda warm-up 0.01 -> 0.1 at epoch 20-40)
    bulk_config = BulkPriorConfig(
        lambda_start=0.01,
        lambda_max=0.1,
        warmup_start=20,
        warmup_end=40,
    )

    # TripleModalTrainer (D-03: pre_warm_epochs=20, D-08: ot_epsilon>=0.1)
    trainer = TripleModalTrainer(
        vae=vae,
        discriminator=discriminator,
        lr=LR,
        pre_warm_epochs=PRE_WARM_EPOCHS,
        r1_weight=1.0,
        ot_epsilon=OT_EPSILON,
        spatial_scaffold=spatial_scaffold,
        bulk_prior_config=bulk_config,
        device=device,
    )
    logger.info(f"TripleModalTrainer initialized: device={device}, pre_warm_epochs={PRE_WARM_EPOCHS}, ot_epsilon={OT_EPSILON}")

    # Pre-build guidance graph (loaded from cache if available)
    guidance_graph = prebuild_guidance_graph()

    # Load datasets (TripleModalDataset handles guidance graph + spatial adjacency)
    # Reuse guidance graph for both datasets (avoids double computation)
    logger.info("Loading Rosacea datasets...")

    train_dataset = load_rosacea_dataset(
        is_validation=False,
        preprocessed=True,
        device=device,
        use_lazy_loading=True,
        validation_quadrant=2,
        guidance_graph=guidance_graph,
    )
    val_dataset = load_rosacea_dataset(
        is_validation=True,
        preprocessed=True,
        device=device,
        use_lazy_loading=True,
        validation_quadrant=2,
        guidance_graph=guidance_graph,
    )
    logger.info(f"Train dataset: {train_dataset.adata_sc.n_obs} cells, {len(train_dataset.st_indices)} ST spots")
    logger.info(f"Val dataset: {val_dataset.adata_sc.n_obs} cells, {len(val_dataset.st_indices)} ST spots")

    # Set spatial graph from dataset (SP-01: squidpy k-NN graph)
    trainer.set_spatial_graph(train_dataset.spatial_adj)
    logger.info(f"Spatial graph set: {train_dataset.spatial_adj.nnz} edges")

    # Chunk 4: Compute condition-level bulk proportions via validated ssGSEA
    logger.info("Computing bulk ssGSEA proportions with condition aggregation...")
    try:
        import scanpy as sc
        from gseapy.parser import read_gmt
        from tglue.deconv.bulk_prior import preprocess_bulk_with_metadata

        bulk_adata = sc.read_h5ad("data/rosacea/array_test.h5ad")
        markers = read_gmt(str(MARKERS_PATH))

        # Use validated ssGSEA with condition aggregation
        bulk_meta = preprocess_bulk_with_metadata(
            bulk_adata, markers, condition_col="condition", sample_id_col="sample"
        )

        # Set condition-level proportions (NEW API)
        trainer.set_bulk_condition_proportions(
            bulk_meta.condition_proportions,
            bulk_meta.condition_names,
        )
        logger.info(f"Bulk condition proportions set: conditions={bulk_meta.condition_names}")
        logger.info(f"  sample_proportions unique rows: {bulk_meta.sample_proportions.unique(dim=0).shape[0]}")

        # Chunk 4: Set metadata for condition aggregation
        # Note: Zarr lazy loading may not preserve all metadata
        # Use original h5ad files for metadata extraction
        try:
            import scanpy as sc

            # Load metadata from original h5ad (not Zarr)
            scrna_h5ad = sc.read_h5ad("data/rosacea/sc_reference.h5ad")
            st_h5ad = sc.read_h5ad("data/rosacea/spatial_100k.h5ad")

            # Get scRNA cell type labels
            if 'cell_type' in scrna_h5ad.obs.columns:
                scrna_cell_type_labels = scrna_h5ad.obs['cell_type'].values
                logger.info(f"scRNA cell type labels: {len(scrna_cell_type_labels)} cells")
            else:
                logger.warning("scRNA missing 'cell_type' column - condition prior may be limited")
                scrna_cell_type_labels = None

            # Get ST condition labels (for training split indices)
            if 'condition' in st_h5ad.obs.columns:
                # Map to canonical conditions (HV -> Normal)
                from tglue.deconv.label_mapping import get_canonical_conditions
                conditions = get_canonical_conditions()
                st_conditions_raw = st_h5ad.obs['condition'].values[train_dataset.st_indices]
                st_condition_labels = np.array([conditions.normalize(str(c)) for c in st_conditions_raw])
                logger.info(f"ST condition labels: {len(st_condition_labels)} spots")
            else:
                logger.warning("ST missing 'condition' column - condition prior may be limited")
                st_condition_labels = None

            # Set metadata for condition prior computation
            if scrna_cell_type_labels is not None and st_condition_labels is not None:
                trainer.set_metadata_for_condition_prior(
                    scrna_cell_type_labels=scrna_cell_type_labels,
                    st_condition_labels=st_condition_labels,
                )
                logger.info("Metadata for condition prior set successfully")
            else:
                logger.warning("Could not set condition prior metadata - missing required columns")

        except Exception as e:
            logger.warning(f"Could not load metadata from h5ad: {e}")
            scrna_cell_type_labels = None
            st_condition_labels = None

    except Exception as e:
        logger.warning(f"Could not compute bulk ssGSEA proportions: {e}. Setting uniform proportions.")
        import traceback
        traceback.print_exc()
        # Fallback: uniform proportions for 2 conditions x n_cell_types
        from tglue.deconv.label_mapping import get_canonical_conditions, get_canonical_cell_types
        conditions = get_canonical_conditions()
        cell_types = get_canonical_cell_types()
        condition_proportions = torch.ones(len(conditions.names), cell_types.n_types) / cell_types.n_types
        trainer.set_bulk_condition_proportions(condition_proportions, conditions.names)

    return vae, discriminator, trainer, train_dataset, val_dataset


def collect_batches(dataset, max_batches_per_epoch: int | None = None):
    """Collect all batches from an IterableDataset into a list.

    Parameters
    ----------
    dataset : TripleModalDataset
        Dataset to iterate over
    max_batches_per_epoch : int, optional
        Maximum number of batches per epoch (for memory management)

    Returns
    -------
    list[dict]
        List of batch dicts
    """
    batches = []
    for batch_idx, batch in enumerate(dataset):
        batches.append(batch)
        if max_batches_per_epoch is not None and batch_idx >= max_batches_per_epoch - 1:
            break
        if batch_idx % 50 == 0 and batch_idx > 0:
            logger.info(f"  Collected {batch_idx + 1} batches...")
    logger.info(f"Collected {len(batches)} batches from dataset")
    return batches


def save_training_metrics(
    checkpoint_dir: Path,
    history: dict,
    final_epoch: int,
    early_stopped: bool,
    best_epoch: int,
) -> None:
    """Save training metrics to JSON.

    Parameters
    ----------
    checkpoint_dir : Path
        Directory for checkpoint files
    history : dict
        Training history from trainer
    final_epoch : int
        Last epoch completed
    early_stopped : bool
        Whether early stopping triggered
    best_epoch : int
        Epoch with best validation loss
    """
    # Compute final loss values
    final_vae_loss = history["vae_loss"][-1] if history["vae_loss"] else None
    final_disc_loss = history["disc_loss"][-1] if history["disc_loss"] else None

    # Compute convergence status from last 10 epochs
    if len(history["vae_loss"]) >= 10:
        last_10_losses = history["vae_loss"][-10:]
        loss_std = float(np.std(last_10_losses))
        convergence_status = "stable" if loss_std < 0.01 else "unstable"
    else:
        loss_std = None
        convergence_status = "insufficient_data"

    metrics = {
        "final_epoch": final_epoch,
        "final_vae_loss": float(final_vae_loss) if final_vae_loss else None,
        "final_disc_loss": float(final_disc_loss) if final_disc_loss else None,
        "early_stopped": early_stopped,
        "best_epoch": best_epoch,
        "convergence_status": convergence_status,
        "loss_std_last_10": loss_std,
        "total_epochs": len(history["vae_loss"]) if history["vae_loss"] else 0,
        "history_keys": list(history.keys()),
    }

    metrics_path = checkpoint_dir / "training_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Training metrics saved to {metrics_path}")


def train_with_pipeline(
    trainer,
    train_dataset,
    val_dataset,
    checkpoint_dir: Path,
    log_dir: Path,
    n_epochs: int = N_EPOCHS,
    patience: int = PATIENCE,
    checkpoint_every: int = CHECKPOINT_EVERY,
    device: str = "cuda",
    resume_from: str | None = None,
) -> dict:
    """Run training using TrainPipeline with streaming batch iteration.

    Uses on-the-fly batch loading from Zarr to avoid pre-collection overhead.
    Uses itertools.cycle to allow multiple training epochs from IterableDataset.

    Parameters
    ----------
    trainer : TripleModalTrainer
        Initialized trainer
    train_dataset : TripleModalDataset
        Training dataset
    val_dataset : TripleModalDataset
        Validation dataset
    checkpoint_dir : Path
        Checkpoint directory
    log_dir : Path
        TensorBoard log directory
    n_epochs : int
        Maximum epochs
    patience : int
        Early stopping patience
    checkpoint_every : int
        Checkpoint interval
    device : str
        Device for training
    resume_from : str, optional
        Checkpoint path to resume from

    Returns
    -------
    dict
        Training history
    """
    import itertools
    from tglue.train.checkpoint import CheckpointManager
    from tglue.train.early_stopping import EarlyStopping
    from tglue.train.tensorboard_logger import TensorBoardLogger
    from tglue.train.deterministic import set_deterministic_seed

    # Initialize components
    set_deterministic_seed(SEED)
    checkpoint_manager = CheckpointManager(str(checkpoint_dir))
    early_stopping = EarlyStopping(patience=patience)
    tb_logger = TensorBoardLogger(str(log_dir))

    # Compute batch counts from dataset sizes (avoid slow iteration)
    n_sc = train_dataset.adata_sc.n_obs
    n_st = len(train_dataset.st_indices)
    train_batch_count = max(
        (n_sc + train_dataset.batch_size_sc - 1) // train_dataset.batch_size_sc,
        (n_st + train_dataset.batch_size_st - 1) // train_dataset.batch_size_st
    )
    val_batch_count = max(
        (val_dataset.adata_sc.n_obs + val_dataset.batch_size_sc - 1) // val_dataset.batch_size_sc,
        (len(val_dataset.st_indices) + val_dataset.batch_size_st - 1) // val_dataset.batch_size_st
    )
    logger.info(f"Streaming training: {train_batch_count} train batches, {val_batch_count} val batches")
    logger.info(f"  patience={patience}, checkpoint_every={checkpoint_every}, device={device}")

    # Create cycling iterator for training (allows multiple epochs)
    train_iterator = itertools.cycle(iter(train_dataset))

    # Resume from checkpoint if provided
    start_epoch = 0
    if resume_from:
        start_epoch = checkpoint_manager.load(trainer, resume_from)
        logger.info(f"Resumed from epoch {start_epoch}")
        early_stopping.reset()

    for epoch in range(start_epoch, n_epochs):
        epoch_start = time.time()
        logger.info(f"Epoch {epoch}/{n_epochs} starting...")

        # Chunk 4: Epoch-start condition prior refresh (using shared helper)
        from tglue.train.condition_prior import refresh_condition_prior_for_epoch
        
        refresh_condition_prior_for_epoch(
            trainer=trainer,
            dataset=train_dataset,
            epoch=epoch,
            device=device,
            ot_prior_start_epoch=20,
            chunk_size=512,
        )

        # Training phase - stream batches from cycling iterator
        trainer.vae.train()
        trainer.discriminator.train()
        epoch_losses: list[dict] = []

        nan_count = 0
        for batch_idx in range(train_batch_count):
            batch = next(train_iterator)  # Get next batch from cycling iterator
            losses = trainer.train_step(batch, epoch)
            epoch_losses.append(losses)

            # BF-05: Track NaN occurrences
            if losses['vae_loss'] != losses['vae_loss']:  # NaN check
                nan_count += 1
                if nan_count <= 5:  # Log first 5 NaN occurrences
                    # Get data info for debugging
                    x_sc_info = f"shape={batch['x_sc'].shape}, range=[{batch['x_sc'].min():.2f}, {batch['x_sc'].max():.2f}]"
                    x_st_info = f"shape={batch['x_st'].shape}, range=[{batch['x_st'].min():.2f}, {batch['x_st'].max():.2f}]"
                    logger.warning(
                        f"  NaN batch {batch_idx + 1}: x_sc={x_sc_info}, x_st={x_st_info}"
                    )

            if (batch_idx + 1) % 100 == 0:
                tb_logger.log_batch_losses(losses, batch_idx + 1, epoch)
                loss_str = f"{losses['vae_loss']:.4f}" if not (losses['vae_loss'] != losses['vae_loss']) else "NaN"
                nan_str = f" ({nan_count} NaN so far)" if nan_count > 0 else ""
                logger.info(f"  Epoch {epoch}: batch {batch_idx + 1}/{train_batch_count}, vae_loss={loss_str}{nan_str}")

        # Compute epoch average losses (skip NaN values)
        avg_losses = {}
        if epoch_losses:
            keys = set()
            for l in epoch_losses:
                keys.update(l.keys())
            for key in keys:
                values = [l[key] for l in epoch_losses if key in l and isinstance(l[key], (int, float))]
                # BF-05: Filter out NaN values
                valid_values = [v for v in values if v == v]  # v != v is True only for NaN
                if valid_values:
                    avg_losses[key] = sum(valid_values) / len(valid_values)
                    if len(valid_values) < len(values):
                        logger.warning(
                            f"  Skipped {len(values) - len(valid_values)} NaN values for {key} in epoch {epoch}"
                        )
            tb_logger.log_epoch_losses(avg_losses, epoch)

        # Validation phase - stream from fresh iterator
        trainer.vae.eval()
        trainer.discriminator.eval()
        val_losses: list[float] = []
        with torch.no_grad():
            for batch in val_dataset:
                x_sc = batch['x_sc'].to(device)
                x_st = batch['x_st'].to(device)
                x_bulk = batch['x_bulk'].to(device)
                guidance_data = batch['guidance_data']
                if hasattr(guidance_data, 'to'):
                    guidance_data = guidance_data.to(device)
                vae_output = trainer.vae(x_sc, x_st, x_bulk, guidance_data)
                recon_loss = vae_output.get('recon_loss', torch.tensor(0.0, device=device))
                kl_loss = vae_output.get('kl_loss', torch.tensor(0.0, device=device))
                graph_loss = vae_output.get('graph_recon_loss', torch.tensor(0.0, device=device))
                val_loss = recon_loss + kl_loss + graph_loss
                if isinstance(val_loss, torch.Tensor):
                    val_loss = val_loss.item()
                val_losses.append(val_loss)

        avg_val_loss = sum(val_losses) / len(val_losses) if val_losses else 0.0
        tb_logger.log_validation_loss(avg_val_loss, epoch)

        epoch_time = time.time() - epoch_start
        logger.info(
            f"Epoch {epoch}: vae_loss={avg_losses.get('vae_loss', 0):.4f}, "
            f"val_loss={avg_val_loss:.4f}, disc_loss={avg_losses.get('disc_loss', 0):.4f}, "
            f"time={epoch_time:.1f}s"
        )

        # Early stopping check
        improved = early_stopping.step(avg_val_loss, epoch)
        if improved:
            checkpoint_manager.save(trainer, epoch, avg_val_loss, is_best=True)
            logger.info(f"  -> New best model (val_loss={avg_val_loss:.4f})")

        # Periodic checkpoint
        if epoch % checkpoint_every == 0:
            checkpoint_manager.save(trainer, epoch, avg_val_loss)
            logger.info(f"  -> Checkpoint saved at epoch {epoch}")

        if early_stopping.should_stop:
            logger.info(f"Early stopping at epoch {epoch}")
            logger.info(f"Best epoch: {early_stopping.best_epoch}")
            best_path = checkpoint_manager.get_best_checkpoint_path()
            if best_path and Path(best_path).exists():
                checkpoint_manager.load(trainer, best_path)
            break

    tb_logger.close()
    return trainer.history


def main():
    parser = argparse.ArgumentParser(description="Phase 12: TripleModalVAE Training on Rosacea Data")
    parser.add_argument(
        "--mode",
        choices=["init", "train"],
        default="train",
        help="Mode: 'init' for initialization only, 'train' for full training",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Checkpoint path to resume from",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=N_EPOCHS,
        help=f"Maximum training epochs (default: {N_EPOCHS})",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for training",
    )
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("Phase 12: TripleModalVAE Training on Rosacea Data")
    logger.info("=" * 70)

    # Set deterministic seed
    set_seed(SEED)

    # Determine device
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = "cpu"
    logger.info(f"Using device: {device}")

    # Create model directories
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize models
    vae, discriminator, trainer, train_dataset, val_dataset = create_models(device)

    if args.mode == "init":
        logger.info("=" * 70)
        logger.info("INIT_COMPLETE: Models and datasets initialized successfully")
        logger.info("=" * 70)
        logger.info(f"VAE parameters: {sum(p.numel() for p in vae.parameters()):,}")
        logger.info(f"Discriminator parameters: {sum(p.numel() for p in discriminator.parameters()):,}")
        logger.info(f"Checkpoint directory: {CHECKPOINT_DIR}")
        logger.info(f"Log directory: {LOG_DIR}")
        return

    # Full training
    logger.info("=" * 70)
    logger.info("Starting training...")
    logger.info(f"  n_epochs={args.epochs}, patience={PATIENCE}, checkpoint_every={CHECKPOINT_EVERY}")
    logger.info("=" * 70)

    history = train_with_pipeline(
        trainer=trainer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        checkpoint_dir=CHECKPOINT_DIR,
        log_dir=LOG_DIR,
        n_epochs=args.epochs,
        patience=PATIENCE,
        checkpoint_every=CHECKPOINT_EVERY,
        device=device,
        resume_from=args.resume,
    )

    # Determine final state
    final_epoch = len(history["vae_loss"]) - 1
    early_stopped = False  # Determined from history length vs n_epochs
    best_epoch = final_epoch  # Approximate

    # Save final metrics
    save_training_metrics(
        checkpoint_dir=CHECKPOINT_DIR,
        history=history,
        final_epoch=final_epoch,
        early_stopped=early_stopped,
        best_epoch=best_epoch,
    )

    # Log final state
    logger.info("=" * 70)
    logger.info("TRAINING_COMPLETE")
    logger.info(f"  Final epoch: {final_epoch}")
    logger.info(f"  Final VAE loss: {history['vae_loss'][-1]:.4f}")
    logger.info(f"  Final disc loss: {history['disc_loss'][-1]:.4f}")
    logger.info("=" * 70)

    # List checkpoint files
    checkpoint_files = sorted(CHECKPOINT_DIR.glob("*.pt"))
    logger.info(f"Checkpoint files ({len(checkpoint_files)}):")
    for cf in checkpoint_files:
        logger.info(f"  {cf}")

    # Verify TensorBoard logs
    tb_files = sorted(LOG_DIR.glob("events.out.tfevents.*"))
    logger.info(f"TensorBoard log files ({len(tb_files)}):")
    for tf in tb_files:
        logger.info(f"  {tf}")

    logger.info("=" * 70)
    logger.info("View TensorBoard: tensorboard --logdir runs/rosacea")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
