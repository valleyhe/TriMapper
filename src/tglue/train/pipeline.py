"""TrainPipeline: Training orchestrator for TripleModalTrainer (TR-01).

Coordinates train + validate loop with:
- Checkpointing for crash recovery (TR-02)
- Early stopping for convergence monitoring (TR-03)
- Deterministic seed control for reproducibility (TR-04)
- TensorBoard logging for visualization (TR-05)

Wraps TripleModalTrainer without modifying it.
"""

from __future__ import annotations

from typing import Any
import torch
from pathlib import Path

from .trainer import TripleModalTrainer
from .checkpoint import CheckpointManager
from .early_stopping import EarlyStopping
from .deterministic import set_deterministic_seed
from .tensorboard_logger import TensorBoardLogger


class TrainPipeline:
    """Training orchestrator for TripleModalTrainer (TR-01).

    Coordinates:
    - Train + validate loop
    - Checkpointing (TR-02)
    - Early stopping (TR-03)
    - Deterministic training (TR-04)
    - TensorBoard logging (TR-05)

    Wraps TripleModalTrainer without modifying it.

    Parameters:
        trainer: TripleModalTrainer instance
        checkpoint_dir: Directory for checkpoints
        log_dir: Directory for TensorBoard logs
        patience: Early stopping patience (epochs)
        checkpoint_every: Save checkpoint every N epochs
        seed: RNG seed for reproducibility
        device: Training device ('cuda' or 'cpu')

    Usage:
        pipeline = TrainPipeline(trainer, checkpoint_dir='checkpoints/')
        history = pipeline.train(train_dataset, val_dataset, n_epochs=100)
    """

    def __init__(
        self,
        trainer: TripleModalTrainer,
        checkpoint_dir: str = 'checkpoints',
        log_dir: str = 'runs',
        patience: int = 10,
        checkpoint_every: int = 5,
        seed: int = 42,
        device: str | None = None,
    ):
        self.trainer = trainer
        # Use trainer's device if not explicitly specified
        self.device = device if device is not None else trainer.device
        self.seed = seed

        # Initialize components from Wave 1 plans
        self.checkpoint_manager = CheckpointManager(checkpoint_dir)
        self.early_stopping = EarlyStopping(patience=patience)
        self.tb_logger = TensorBoardLogger(log_dir)

        self.checkpoint_every = checkpoint_every
        self.log_dir = Path(log_dir)

    def train(
        self,
        train_batches: list[dict[str, Any]],
        val_batches: list[dict[str, Any]],
        n_epochs: int,
        resume_from: str | None = None,
    ) -> dict[str, list[float]]:
        """Full training loop with validation, checkpointing, early stopping.

        Parameters:
            train_batches: List of batch dicts from TripleModalDataset
            val_batches: List of validation batch dicts
            n_epochs: Maximum epochs to train
            resume_from: Optional checkpoint path to resume from

        Returns:
            history: Training loss history dict
        """
        # TR-04: Set deterministic seed
        set_deterministic_seed(self.seed)

        # Resume from checkpoint if provided
        start_epoch = 0
        if resume_from:
            start_epoch = self.checkpoint_manager.load(self.trainer, resume_from)
            print(f"Resumed from epoch {start_epoch}")
            # Reset early stopping state after resume
            self.early_stopping.reset()

        for epoch in range(start_epoch, n_epochs):
            # Training phase
            epoch_losses: list[dict[str, float]] = []
            self.trainer.vae.train()
            self.trainer.discriminator.train()

            for batch_idx, batch in enumerate(train_batches):
                losses = self.trainer.train_step(batch, epoch)
                epoch_losses.append(losses)

                # TR-05: Log per-batch losses
                self.tb_logger.log_batch_losses(losses, batch_idx, epoch)

            # Compute epoch average losses
            avg_losses = {}
            if epoch_losses:
                # Get all keys from first loss dict
                keys = set()
                for l in epoch_losses:
                    keys.update(l.keys())

                for key in keys:
                    values = [l[key] for l in epoch_losses if key in l and isinstance(l[key], (int, float))]
                    if values:
                        avg_losses[key] = sum(values) / len(values)

                self.tb_logger.log_epoch_losses(avg_losses, epoch)

            # Validation phase
            val_loss = self.validate(val_batches, epoch)
            self.tb_logger.log_validation_loss(val_loss, epoch)

            # TR-03: Early stopping check
            improved = self.early_stopping.step(val_loss, epoch)
            if improved:
                # Save best model checkpoint
                self.checkpoint_manager.save(
                    self.trainer, epoch, val_loss, is_best=True
                )
                print(f"Epoch {epoch}: New best model (val_loss={val_loss:.4f})")

            # TR-02: Periodic checkpoint
            if epoch % self.checkpoint_every == 0:
                self.checkpoint_manager.save(self.trainer, epoch, val_loss)
                print(f"Epoch {epoch}: Checkpoint saved")

            # Check early stopping
            if self.early_stopping.should_stop:
                print(f"Early stopping at epoch {epoch}")
                print(f"Best epoch: {self.early_stopping.best_epoch}")
                # Restore best model
                best_path = self.checkpoint_manager.get_best_checkpoint_path()
                if best_path and Path(best_path).exists():
                    self.checkpoint_manager.load(self.trainer, best_path)
                break

        self.tb_logger.close()
        return self.trainer.history

    def validate(
        self,
        val_batches: list[dict[str, Any]],
        epoch: int,
    ) -> float:
        """Validation loop. Returns average validation loss.

        Args:
            val_batches: List of validation batch dicts
            epoch: Current epoch number

        Returns:
            float: Average VAE loss over validation batches
        """
        self.trainer.vae.eval()
        self.trainer.discriminator.eval()

        val_losses: list[float] = []
        with torch.no_grad():
            for batch in val_batches:
                # Move batch to device
                x_sc = batch['x_sc'].to(self.device)
                x_st = batch['x_st'].to(self.device)
                x_bulk = batch['x_bulk'].to(self.device)
                guidance_data = batch['guidance_data']
                if hasattr(guidance_data, 'to'):
                    guidance_data = guidance_data.to(self.device)

                # VAE forward pass (no backward in validation)
                vae_output = self.trainer.vae(x_sc, x_st, x_bulk, guidance_data)

                # Compute validation loss (recon + KL + graph, no adversarial)
                recon_loss = vae_output.get('recon_loss', torch.tensor(0.0))
                kl_loss = vae_output.get('kl_loss', torch.tensor(0.0))
                graph_loss = vae_output.get('graph_recon_loss', torch.tensor(0.0))

                val_loss = recon_loss + kl_loss + graph_loss
                if isinstance(val_loss, torch.Tensor):
                    val_loss = val_loss.item()
                val_losses.append(val_loss)

        avg_val_loss = sum(val_losses) / len(val_losses) if val_losses else 0.0
        return avg_val_loss

    def __repr__(self) -> str:
        return (
            f"TrainPipeline(checkpoint_dir='{self.checkpoint_manager.checkpoint_dir}', "
            f"log_dir='{self.log_dir}', patience={self.early_stopping.patience})"
        )