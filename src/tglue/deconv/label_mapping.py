"""Label mapping utilities for triple-modal integration.

Provides:
1. Canonical cell type names and ordering
2. Canonical condition names with alias mapping
3. Label normalization functions

This module ensures consistent label spaces across scRNA, ST, and Bulk data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class CanonicalCellTypes:
    """Canonical cell type definitions for Rosacea dataset.

    Attributes:
        names: List of canonical cell type names (standardized)
        original_to_canonical: Mapping from original names to canonical names
        n_types: Number of cell types
    """

    names: List[str] = field(default_factory=lambda: [
        "Vascular_endothelial",
        "Pericytes",
        "Fibroblasts",
        "CD4_T_cells",
        "Lymphatic_endothelial",
        "Cytotoxic_T_cells",
        "Macrophages",
        "Melanocytes",
        "Keratinocytes",
        "Mast_cells",
        "Adipocytes",
        "B_cells",
        "Vascular_smooth_muscle",
        "Schwann_cells",
        "Sweat_gland_secretory",
        "Neutrophils",
    ])

    original_to_canonical: Dict[str, str] = field(default_factory=lambda: {
        # Original -> Canonical mapping
        "Vascular endothelial cells": "Vascular_endothelial",
        "Pericytes": "Pericytes",
        "Fibroblasts": "Fibroblasts",
        "CD4⁺ T cells": "CD4_T_cells",
        "CD4+ T cells": "CD4_T_cells",
        "Lymphatic endothelial cells": "Lymphatic_endothelial",
        "Cytotoxic T cells": "Cytotoxic_T_cells",
        "Macrophages": "Macrophages",
        "Melanocytes": "Melanocytes",
        "Keratinocytes": "Keratinocytes",
        "Mast cells": "Mast_cells",
        "Adipocytes": "Adipocytes",
        "B cells": "B_cells",
        "Vascular smooth muscle cells": "Vascular_smooth_muscle",
        "Schwann cells": "Schwann_cells",
        "Sweat gland secretory cells": "Sweat_gland_secretory",
        "Neutrophils": "Neutrophils",
        # GMT marker aliases (underscores to underscores)
        "Vascular_endothelial": "Vascular_endothelial",
        "CD4_T_cells": "CD4_T_cells",
        "Lymphatic_endothelial": "Lymphatic_endothelial",
        "Cytotoxic_T_cells": "Cytotoxic_T_cells",
        "Vascular_smooth_muscle": "Vascular_smooth_muscle",
        "Sweat_gland_secretory": "Sweat_gland_secretory",
    })

    @property
    def n_types(self) -> int:
        return len(self.names)

    def get_index(self, canonical_name: str) -> int:
        """Get index of canonical cell type name."""
        return self.names.index(canonical_name)

    def normalize(self, original_name: str) -> str:
        """Normalize original cell type name to canonical."""
        return self.original_to_canonical.get(original_name, original_name)

    def to_onehot(self, labels: np.ndarray) -> np.ndarray:
        """Convert original labels to one-hot matrix using canonical ordering.

        Parameters
        ----------
        labels : np.ndarray
            Array of original cell type labels

        Returns
        -------
        np.ndarray
            One-hot matrix of shape (n_samples, n_types)
        """
        n_samples = len(labels)
        onehot = np.zeros((n_samples, self.n_types), dtype=np.float32)

        for i, label in enumerate(labels):
            canonical = self.normalize(label)
            if canonical in self.names:
                onehot[i, self.get_index(canonical)] = 1.0

        return onehot


@dataclass
class CanonicalConditions:
    """Canonical condition definitions for Rosacea dataset.

    Attributes:
        names: List of canonical condition names
        aliases: Mapping from original condition names to canonical names
        n_conditions: Number of conditions
    """

    names: List[str] = field(default_factory=lambda: ["Normal", "Rosacea"])

    aliases: Dict[str, str] = field(default_factory=lambda: {
        # Bulk condition aliases
        "HV": "Normal",
        "Healthy Volunteer": "Normal",
        "Healthy": "Normal",
        "Control": "Normal",
        # scRNA/ST already use canonical names
        "Normal": "Normal",
        "Rosacea": "Rosacea",
    })

    @property
    def n_conditions(self) -> int:
        return len(self.names)

    def get_index(self, canonical_name: str) -> int:
        """Get index of canonical condition name."""
        return self.names.index(canonical_name)

    def normalize(self, original_name: str) -> str:
        """Normalize original condition name to canonical."""
        return self.aliases.get(original_name, original_name)

    def normalize_array(self, labels: np.ndarray) -> np.ndarray:
        """Normalize array of condition labels to canonical.

        Parameters
        ----------
        labels : np.ndarray
            Array of original condition labels

        Returns
        -------
        np.ndarray
            Array of canonical condition labels
        """
        return np.array([self.normalize(str(label)) for label in labels])


def get_canonical_cell_types() -> CanonicalCellTypes:
    """Get canonical cell type definitions for Rosacea."""
    return CanonicalCellTypes()


def get_canonical_conditions() -> CanonicalConditions:
    """Get canonical condition definitions for Rosacea."""
    return CanonicalConditions()


def validate_label_consistency(
    scrna_obs: pd.DataFrame,
    st_obs: pd.DataFrame,
    bulk_obs: pd.DataFrame,
    cell_type_col: str = "cell_type",
    condition_col: str = "condition",
) -> Dict[str, any]:
    """Validate label consistency across modalities.

    Parameters
    ----------
    scrna_obs : pd.DataFrame
        scRNA observation metadata
    st_obs : pd.DataFrame
        ST observation metadata
    bulk_obs : pd.DataFrame
        Bulk observation metadata
    cell_type_col : str
        Column name for cell type in scRNA
    condition_col : str
        Column name for condition

    Returns
    -------
    Dict
        Validation results with consistency status
    """
    cell_types = get_canonical_cell_types()
    conditions = get_canonical_conditions()

    results = {
        "valid": True,
        "issues": [],
        "scrna_cell_types": [],
        "scrna_conditions": [],
        "st_conditions": [],
        "bulk_conditions": [],
    }

    # Check scRNA cell types
    if cell_type_col in scrna_obs.columns:
        original_types = scrna_obs[cell_type_col].unique().tolist()
        for t in original_types:
            canonical = cell_types.normalize(t)
            if canonical not in cell_types.names:
                results["issues"].append(f"scRNA cell type '{t}' cannot be mapped to canonical")
                results["valid"] = False
            else:
                results["scrna_cell_types"].append(canonical)

    # Check conditions in all modalities
    if condition_col in scrna_obs.columns:
        for c in scrna_obs[condition_col].unique():
            canonical = conditions.normalize(str(c))
            if canonical not in conditions.names:
                results["issues"].append(f"scRNA condition '{c}' cannot be mapped")
                results["valid"] = False
            results["scrna_conditions"].append(canonical)

    if condition_col in st_obs.columns:
        for c in st_obs[condition_col].unique():
            canonical = conditions.normalize(str(c))
            if canonical not in conditions.names:
                results["issues"].append(f"ST condition '{c}' cannot be mapped")
                results["valid"] = False
            results["st_conditions"].append(canonical)

    if condition_col in bulk_obs.columns:
        for c in bulk_obs[condition_col].unique():
            canonical = conditions.normalize(str(c))
            if canonical not in conditions.names:
                results["issues"].append(f"Bulk condition '{c}' cannot be mapped")
                results["valid"] = False
            results["bulk_conditions"].append(canonical)

    # Check condition intersection
    scrna_cond_set = set(results["scrna_conditions"])
    st_cond_set = set(results["st_conditions"])
    bulk_cond_set = set(results["bulk_conditions"])

    common_conditions = scrna_cond_set & st_cond_set & bulk_cond_set
    if len(common_conditions) == 0:
        results["issues"].append("No common conditions across modalities")
        results["valid"] = False

    results["common_conditions"] = list(common_conditions)

    return results