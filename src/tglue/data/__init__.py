"""Data loading utilities for Triple-Modal Integration.

Provides TripleModalDataset for loading scRNA, ST, and Bulk AnnData files
and producing batch dicts compatible with TripleModalTrainer.
"""

from __future__ import annotations

from .dataset import TripleModalDataset

__all__ = ["TripleModalDataset"]