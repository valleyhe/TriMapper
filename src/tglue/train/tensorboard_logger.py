"""TensorBoard logging for TripleModalTrainer (TR-05).

Provides structured logging of training losses and alignment metrics
to TensorBoard for real-time visualization.

Usage:
    logger = TensorBoardLogger(log_dir='runs/experiment_01')
    for epoch in range(100):
        for batch_idx, batch in enumerate(dataloader):
            losses = trainer.train_step(batch, epoch)
            logger.log_batch_losses(losses, batch_idx, epoch)

        val_loss = validate()
        logger.log_validation_loss(val_loss, epoch)

        metrics = evaluate_alignment()
        logger.log_alignment_metrics(metrics, epoch)

    logger.close()

    # View: tensorboard --logdir runs/
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from torch.utils.tensorboard import SummaryWriter

if TYPE_CHECKING:
    from ..evaluation.metrics import AlignmentMetrics


class TensorBoardLogger:
    """TensorBoard logging for TripleModalTrainer (TR-05).

    Logs training losses per batch and per epoch, plus alignment metrics.
    Uses separate prefixes for organized visualization:
    - batch/{key}: Per-batch losses (detailed view)
    - epoch/{key}: Per-epoch average losses (summary view)
    - alignment/{metric}: Alignment metrics (ASW, NMI, GC)
    - validation/{loss}: Validation loss

    Parameters:
        log_dir: Directory for TensorBoard logs
        comment: Optional comment appended to log_dir name
    """

    def __init__(self, log_dir: str, comment: str = ''):
        self.log_dir = Path(log_dir)
        self.writer = SummaryWriter(log_dir=str(self.log_dir), comment=comment)

    def log_batch_losses(
        self,
        losses: dict[str, float],
        batch_idx: int,
        epoch: int,
    ) -> None:
        """Log per-batch training losses.

        Args:
            losses: Dict from trainer.train_step() with loss values
            batch_idx: Batch index within epoch
            epoch: Current epoch number
        """
        # Use epoch-based global step for consistent epoch boundaries
        # Scale: epoch * 10000 + batch_idx (assumes <10000 batches per epoch)
        global_step = epoch * 10000 + batch_idx

        for key, value in losses.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                # Skip non-scalar values like modality_weights list
                self.writer.add_scalar(f'batch/{key}', float(value), global_step)

    def log_epoch_losses(
        self,
        avg_losses: dict[str, float],
        epoch: int,
    ) -> None:
        """Log per-epoch average losses.

        Args:
            avg_losses: Dict of average loss values for the epoch
            epoch: Current epoch number
        """
        for key, value in avg_losses.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                self.writer.add_scalar(f'epoch/{key}', float(value), epoch)

    def log_validation_loss(self, val_loss: float, epoch: int) -> None:
        """Log validation loss.

        Args:
            val_loss: Average validation loss for the epoch
            epoch: Current epoch number
        """
        self.writer.add_scalar('validation/loss', float(val_loss), epoch)

    def log_alignment_metrics(
        self,
        metrics: "AlignmentMetrics",
        epoch: int,
    ) -> None:
        """Log alignment metrics (ASW, NMI, GC).

        Args:
            metrics: AlignmentMetrics object from evaluate_alignment()
            epoch: Current epoch number
        """
        self.writer.add_scalar('alignment/asw', float(metrics.asw), epoch)
        self.writer.add_scalar('alignment/nmi', float(metrics.nmi), epoch)
        self.writer.add_scalar('alignment/gc', float(metrics.gc), epoch)

    def log_hyperparameters(
        self,
        hparams: dict[str, float | int | str],
        metrics: dict[str, float],
    ) -> None:
        """Log hyperparameters with final metrics.

        Args:
            hparams: Dict of hyperparameter values
            metrics: Dict of final metric values
        """
        # Convert all values to compatible types
        hparams_clean = {}
        for k, v in hparams.items():
            if isinstance(v, (int, float, str, bool)):
                hparams_clean[k] = v

        metrics_clean = {}
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                metrics_clean[k] = float(v)

        self.writer.add_hparams(hparams_clean, metrics_clean)

    def close(self) -> None:
        """Close SummaryWriter and flush pending logs."""
        self.writer.close()

    def __repr__(self) -> str:
        return f"TensorBoardLogger(log_dir='{self.log_dir}')"