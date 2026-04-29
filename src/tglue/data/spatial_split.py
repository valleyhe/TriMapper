"""Spatial contiguous validation split for ST data.

Implements quadrant-based split to prevent information leakage through spatial neighbors.
Following D-05: ST validation uses geographically separated blocks.
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)


def spatial_quadrant_split(
    coords: np.ndarray,
    validation_fraction: float = 0.2,
    validation_quadrant: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split ST spots into geographically separated train/validation sets.

    Uses quadrant-based split: validation set is one quadrant of the tissue.
    This ensures spatial separation and prevents information leakage through
    neighboring spots (D-05: spatial contiguous blocks).

    Parameters
    ----------
    coords : np.ndarray
        Spatial coordinates array (n_spots, 2) from st_adata.obsm['spatial'].
        First column is x-coordinate, second is y-coordinate.
    validation_fraction : float, default 0.2
        Target fraction of spots for validation (approximately 20%).
    validation_quadrant : int, default 2
        Which quadrant to use for validation:
        - 0: Northeast (NE) - x >= center_x, y >= center_y
        - 1: Northwest (NW) - x < center_x, y >= center_y
        - 2: Southwest (SW) - x < center_x, y < center_y (default)
        - 3: Southeast (SE) - x >= center_x, y < center_y

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        - train_indices: Array of indices for training spots
        - val_indices: Array of indices for validation spots

    Notes
    -----
    Geographic separation guarantee:
    - Train and validation spots are in different quadrants
    - No validation spot is adjacent to a training spot across quadrant boundary
    - This prevents information leakage through spatial k-NN graph

    Fraction adjustment:
    - If validation quadrant has >1.5x target fraction, subsample with seed 42
    - This ensures reproducibility while maintaining geographic separation
    """
    # Step 1: Compute tissue center
    center_x = coords[:, 0].mean()
    center_y = coords[:, 1].mean()

    # Step 2: Assign each spot to quadrant
    # Quadrant assignment based on position relative to center
    quadrant = np.zeros(coords.shape[0], dtype=int)

    # NE: x >= center_x, y >= center_y (quadrant 0)
    quadrant[(coords[:, 0] >= center_x) & (coords[:, 1] >= center_y)] = 0

    # NW: x < center_x, y >= center_y (quadrant 1)
    quadrant[(coords[:, 0] < center_x) & (coords[:, 1] >= center_y)] = 1

    # SW: x < center_x, y < center_y (quadrant 2)
    quadrant[(coords[:, 0] < center_x) & (coords[:, 1] < center_y)] = 2

    # SE: x >= center_x, y < center_y (quadrant 3)
    quadrant[(coords[:, 0] >= center_x) & (coords[:, 1] < center_y)] = 3

    # Step 3: Choose validation quadrant
    val_indices = np.where(quadrant == validation_quadrant)[0]
    train_indices = np.where(quadrant != validation_quadrant)[0]

    # Step 4: Adjust to target fraction if quadrant is too large
    val_fraction_actual = len(val_indices) / coords.shape[0]

    if val_fraction_actual > validation_fraction * 1.5:
        # Subsample validation set to match target fraction
        n_val_target = int(coords.shape[0] * validation_fraction)
        np.random.seed(42)  # Reproducibility
        val_indices = np.random.choice(val_indices, n_val_target, replace=False)
        # Recompute train indices (all spots not in subsampled val set)
        train_indices = np.setdiff1d(np.arange(coords.shape[0]), val_indices)

    # Step 5: Log split statistics
    train_fraction = len(train_indices) / coords.shape[0]
    val_fraction_final = len(val_indices) / coords.shape[0]

    logger.info(f"[Spatial Split] Train: {len(train_indices)} spots ({train_fraction:.1%})")
    logger.info(f"[Spatial Split] Val: {len(val_indices)} spots ({val_fraction_final:.1%})")
    logger.info(f"[Spatial Split] Val quadrant: {validation_quadrant}")

    return train_indices, val_indices