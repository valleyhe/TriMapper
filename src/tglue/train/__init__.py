"""Training infrastructure modules.

Includes:
- TripleModalTrainer: Main trainer with phased adversarial training
- TrainPipeline: Training orchestrator integrating all components (TR-01)
- CheckpointManager: Checkpoint save/restore for crash recovery (TR-02)
- EarlyStopping: Patience-based early stopping (TR-03)
- TensorBoardLogger: TensorBoard logging for losses and metrics (TR-05)
- set_deterministic_seed: Reproducible training seed control (TR-04)
"""

from __future__ import annotations

from .trainer import TripleModalTrainer
from .pipeline import TrainPipeline
from .checkpoint import CheckpointManager
from .early_stopping import EarlyStopping
from .tensorboard_logger import TensorBoardLogger
from .deterministic import set_deterministic_seed

__all__ = [
    "TripleModalTrainer",
    "TrainPipeline",
    "CheckpointManager",
    "EarlyStopping",
    "TensorBoardLogger",
    "set_deterministic_seed",
]