"""Early stopping with patience and best model tracking (TR-03).

Monitors validation loss (or metric) and signals when training should stop.
After patience epochs without improvement, should_stop becomes True.
"""

from __future__ import annotations


class EarlyStopping:
    """Early stopping with patience and best model tracking (TR-03).

    Monitors validation loss (or metric) and signals when training should stop.
    After patience epochs without improvement, should_stop becomes True.

    Usage:
        es = EarlyStopping(patience=10, min_delta=0.001)
        for epoch in range(100):
            val_loss = validate()
            improved = es.step(val_loss, epoch)
            if improved:
                checkpoint_manager.save(trainer, epoch, val_loss, is_best=True)
            if es.should_stop:
                checkpoint_manager.load(trainer, 'best_model.pt')
                break

    Parameters:
        patience: Number of epochs to wait for improvement before stopping
        min_delta: Minimum change to qualify as improvement
        mode: 'min' for loss (lower is better), 'max' for metrics (higher is better)
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.001,
        mode: str = 'min',
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode

        self.counter = 0
        self.best_loss = float('inf') if mode == 'min' else float('-inf')
        self.should_stop = False
        self.best_epoch = 0

    def step(self, val_loss: float, epoch: int) -> bool:
        """Check if validation loss improved. Returns True if improved.

        Args:
            val_loss: Current validation loss (or metric value)
            epoch: Current epoch number

        Returns:
            bool: True if this is a new best value, False otherwise
        """
        if self.mode == 'min':
            improved = val_loss < self.best_loss - self.min_delta
        else:
            improved = val_loss > self.best_loss + self.min_delta

        if improved:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
            return False

    def reset(self):
        """Reset state for a new training run."""
        self.counter = 0
        self.best_loss = float('inf') if self.mode == 'min' else float('-inf')
        self.should_stop = False
        self.best_epoch = 0

    def __repr__(self) -> str:
        return (
            f"EarlyStopping(patience={self.patience}, min_delta={self.min_delta}, "
            f"mode='{self.mode}', counter={self.counter}, best_epoch={self.best_epoch})"
        )