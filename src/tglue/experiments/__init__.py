"""Ablation experiment framework for Triple-Modal GLUE.

This package provides:
- AblationConfig: Dataclass for component-level control
- AblationRunner: Sweep executor for running 5 ablation experiments
- variants: Baseline implementations for ablation studies
"""

from .ablation_config import AblationConfig
from .ablation_runner import AblationRunner

__all__ = ["AblationConfig", "AblationRunner"]