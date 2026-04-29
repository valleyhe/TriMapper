"""BulkPrior disable helper for AB-03 (D-AB03).

Disables Bulk prior constraint by setting lambda=0.
Bulk encoder remains, but KL term contribution is zero.

Alternative: trainer.bulk_proportions = None to skip entirely.
"""

from __future__ import annotations

from tglue.deconv.bulk_prior import BulkPriorConfig


def create_no_bulk_trainer_config() -> BulkPriorConfig:
    """Create BulkPriorConfig with lambda=0 (AB-03 baseline).

    Bulk prior disabled via lambda_start=0, lambda_max=0.
    This ensures bulk_kl_loss = 0 throughout training.

    Returns
    -------
    BulkPriorConfig
        Config with lambda=0, disabling Bulk prior constraint
    """
    return BulkPriorConfig(
        lambda_start=0.0,
        lambda_max=0.0,
        warmup_start=0,
        warmup_end=0,
    )