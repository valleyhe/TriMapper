"""Checkpoint save/restore for TripleModalTrainer (TR-02).

Saves complete training state for crash recovery:
- Model state dicts (vae, discriminator, spatial_scaffold)
- Optimizer state dicts (essential for identical resume - Adam momentum)
- RNG states (torch, cuda, numpy, python)
- Training history dict
- Epoch and validation loss

Usage:
    manager = CheckpointManager('checkpoints/', max_checkpoints=5)
    manager.save(trainer, epoch=10, val_loss=0.5, is_best=True)
    start_epoch = manager.load(trainer, 'checkpoints/best_model.pt')
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from .trainer import TripleModalTrainer


class CheckpointManager:
    """Checkpoint save/restore for TripleModalTrainer (TR-02).

    Saves complete training state for crash recovery:
    - Model state dicts (vae, discriminator, spatial_scaffold)
    - Optimizer state dicts (essential for identical resume - Adam momentum)
    - RNG states (torch, cuda, numpy, python)
    - Training history dict
    - Epoch and validation loss

    Usage:
        manager = CheckpointManager('checkpoints/', max_checkpoints=5)
        manager.save(trainer, epoch=10, val_loss=0.5, is_best=True)
        start_epoch = manager.load(trainer, 'checkpoints/best_model.pt')
    """

    def __init__(self, checkpoint_dir: str, max_checkpoints: int = 5):
        """Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to store checkpoints
            max_checkpoints: Maximum number of epoch checkpoints to keep (best_model.pt excluded)
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.max_checkpoints = max_checkpoints
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        trainer: TripleModalTrainer,
        epoch: int,
        val_loss: float,
        is_best: bool = False,
    ) -> str:
        """Save checkpoint with complete training state.

        Args:
            trainer: TripleModalTrainer instance
            epoch: Current epoch number
            val_loss: Validation loss for this epoch
            is_best: If True, also save as best_model.pt

        Returns:
            str: Path to saved checkpoint file
        """
        checkpoint = {
            'epoch': epoch,
            'val_loss': val_loss,
            'vae_state_dict': trainer.vae.state_dict(),
            'disc_state_dict': trainer.discriminator.state_dict(),
            'opt_vae_state_dict': trainer.opt_vae.state_dict(),
            'opt_disc_state_dict': trainer.opt_disc.state_dict(),
            'torch_rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            'numpy_rng_state': np.random.get_state(),
            'python_rng_state': random.getstate(),
            'history': trainer.history,
        }

        # Save spatial_scaffold if present
        if hasattr(trainer, 'spatial_scaffold') and trainer.spatial_scaffold is not None:
            checkpoint['spatial_scaffold_state_dict'] = trainer.spatial_scaffold.state_dict()

        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)

        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)

        self._cleanup_old_checkpoints()

        # Human-readable metrics
        metrics_path = self.checkpoint_dir / f'metrics_epoch_{epoch}.json'
        with open(metrics_path, 'w') as f:
            json.dump({
                'epoch': epoch,
                'val_loss': val_loss,
                'history_last': {
                    k: v[-1] if v else None
                    for k, v in trainer.history.items()
                }
            }, f, indent=2)

        return str(checkpoint_path)

    def load(self, trainer: TripleModalTrainer, checkpoint_path: str) -> int:
        """Load checkpoint and restore all states.

        Args:
            trainer: TripleModalTrainer instance to restore
            checkpoint_path: Path to checkpoint file

        Returns:
            int: Epoch number from checkpoint
        """
        checkpoint = torch.load(checkpoint_path, weights_only=False)

        trainer.vae.load_state_dict(checkpoint['vae_state_dict'])
        trainer.discriminator.load_state_dict(checkpoint['disc_state_dict'])

        # Load optimizer states if present (older checkpoints may not have them)
        if 'opt_vae_state_dict' in checkpoint:
            trainer.opt_vae.load_state_dict(checkpoint['opt_vae_state_dict'])
        if 'opt_disc_state_dict' in checkpoint:
            trainer.opt_disc.load_state_dict(checkpoint['opt_disc_state_dict'])

        # Restore spatial_scaffold if saved
        if 'spatial_scaffold_state_dict' in checkpoint and trainer.spatial_scaffold is not None:
            trainer.spatial_scaffold.load_state_dict(checkpoint['spatial_scaffold_state_dict'])

        # Restore RNG states if saved (critical for TR-04 identical resume)
        if 'torch_rng_state' in checkpoint:
            torch.set_rng_state(checkpoint['torch_rng_state'])
        if 'cuda_rng_state' in checkpoint and checkpoint['cuda_rng_state'] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state(checkpoint['cuda_rng_state'])
        if 'numpy_rng_state' in checkpoint:
            np.random.set_state(checkpoint['numpy_rng_state'])
        if 'python_rng_state' in checkpoint:
            random.setstate(checkpoint['python_rng_state'])

        trainer.history = checkpoint['history']

        return checkpoint['epoch']

    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints beyond max_checkpoints limit."""
        checkpoints = sorted(self.checkpoint_dir.glob('checkpoint_epoch_*.pt'))
        while len(checkpoints) > self.max_checkpoints:
            old_checkpoint = checkpoints.pop(0)
            old_checkpoint.unlink()
            # Also remove associated metrics file
            epoch_num = old_checkpoint.stem.split('_')[-1]
            metrics_file = self.checkpoint_dir / f'metrics_epoch_{epoch_num}.json'
            if metrics_file.exists():
                metrics_file.unlink()

    def get_best_checkpoint_path(self) -> str | None:
        """Get path to best model checkpoint if exists.

        Returns:
            str | None: Path to best_model.pt or None if not exists
        """
        best_path = self.checkpoint_dir / 'best_model.pt'
        return str(best_path) if best_path.exists() else None

    def get_latest_checkpoint_path(self) -> str | None:
        """Get path to latest epoch checkpoint if exists.

        Returns:
            str | None: Path to latest checkpoint or None if not exists
        """
        checkpoints = sorted(self.checkpoint_dir.glob('checkpoint_epoch_*.pt'))
        return str(checkpoints[-1]) if checkpoints else None