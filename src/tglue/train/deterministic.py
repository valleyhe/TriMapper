"""Deterministic seed control for reproducible training (TR-04).

Sets all RNG sources and enables deterministic algorithms for reproducibility.
Must be called BEFORE model creation, data loading, and training.

Note: CUBLAS_WORKSPACE_CONFIG must be set before torch import for CUDA determinism.
This module sets it at import time to ensure proper configuration.
"""

from __future__ import annotations

import os
import random
import numpy as np
import torch

# Set CUBLAS workspace config before any torch operations
# This must be set BEFORE torch import for CUDA determinism
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


def set_deterministic_seed(seed: int = 42) -> None:
    """Set all RNG seeds for reproducible training (TR-04).

    Configures:
    - Python random module
    - NumPy RNG
    - PyTorch CPU RNG
    - PyTorch CUDA RNG (all devices)
    - cuDNN deterministic mode
    - PyTorch deterministic algorithms

    Args:
        seed: Integer seed for all RNG sources

    Note:
        This function must be called before:
        - Model creation (parameter initialization)
        - Data loading (shuffle order)
        - Training loop (dropout, batch order)

        For full reproducibility across runs:
        - Use DataLoader with num_workers=0 (workers have separate RNG)
        - Avoid operations without deterministic implementations
    """
    # Python RNG
    random.seed(seed)

    # NumPy RNG
    np.random.seed(seed)

    # PyTorch CPU RNG
    torch.manual_seed(seed)

    # PyTorch CUDA RNG (current device)
    torch.cuda.manual_seed(seed)

    # PyTorch CUDA RNG (all devices)
    torch.cuda.manual_seed_all(seed)

    # cuDNN deterministic mode (slower but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # PyTorch deterministic algorithms
    # warn_only=True allows operations without deterministic implementation
    # to run with a warning instead of raising an error
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        # PyTorch < 1.11 doesn't have warn_only parameter
        torch.use_deterministic_algorithms(True)


def get_rng_states() -> dict:
    """Capture current RNG states for checkpointing.

    Returns dict with RNG states that can be restored via set_rng_states().
    Used by CheckpointManager to save RNG state for identical resume.

    Returns:
        dict with 'torch', 'cuda', 'numpy', 'python' RNG states
    """
    return {
        'torch': torch.get_rng_state(),
        'cuda': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        'numpy': np.random.get_state(),
        'python': random.getstate(),
    }


def set_rng_states(states: dict) -> None:
    """Restore RNG states from captured dict.

    Args:
        states: Dict from get_rng_states() containing RNG state tensors/tuples
    """
    torch.set_rng_state(states['torch'])
    if states['cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(states['cuda'])
    np.random.set_state(states['numpy'])
    random.setstate(states['python'])