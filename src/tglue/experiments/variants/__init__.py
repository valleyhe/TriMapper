"""Ablation variant implementations for component replacement.

Each variant replaces a component with a baseline:
- build_erdos_renyi_baseline: Random graph for AB-01
- NoSpatialScaffold: Identity pass-through for AB-02
- create_no_bulk_trainer_config: Lambda=0 helper for AB-03
- UniformTransport: Uniform coupling for AB-04
"""

from .random_graph import build_erdos_renyi_baseline
from .no_spatial import NoSpatialScaffold
from .no_bulk import create_no_bulk_trainer_config
from .no_ot import UniformTransport

__all__ = [
    "build_erdos_renyi_baseline",
    "NoSpatialScaffold",
    "create_no_bulk_trainer_config",
    "UniformTransport",
]