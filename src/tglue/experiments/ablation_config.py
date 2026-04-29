"""AblationConfig dataclass for component-level control (D-AB07).

GLUE pattern: Constructor-time conditional instantiation, not runtime branching.
Each flag controls whether a component is instantiated in the model/trainer.

Usage:
    config = AblationConfig(use_guidance_graph=False)  # AB-01
    config = AblationConfig(use_fusion_conv=False)     # AB-02
    config = AblationConfig(use_bulk_prior=False)      # AB-03
    config = AblationConfig(use_ot_deconv=False)       # AB-04
    config = AblationConfig()                          # Full model (all True)
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Any


@dataclass(frozen=True)
class AblationConfig:
    """Configuration for ablation experiments.

    Each boolean flag controls component instantiation:
    - use_guidance_graph: If False, replace with Erdos-Renyi random graph (AB-01)
    - use_fusion_conv: If False, replace with identity pass-through (AB-02)
    - use_bulk_prior: If False, set bulk_loss_weight=0 (AB-03)
    - use_ot_deconv: If False, replace Sinkhorn with uniform coupling (AB-04)

    All flags default to True (full model baseline).

    frozen=True ensures immutability after creation.
    """

    use_guidance_graph: bool = True
    use_fusion_conv: bool = True
    use_bulk_prior: bool = True
    use_ot_deconv: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Serialize config to JSON-compatible dict.

        For experiment tracking and reproducibility (D-AB10).
        """
        return asdict(self)

    def ablation_name(self) -> str:
        """Return descriptive name for this ablation configuration.

        Returns:
            str: 'full' if all True, otherwise describes disabled components
        """
        if all([self.use_guidance_graph, self.use_fusion_conv,
                self.use_bulk_prior, self.use_ot_deconv]):
            return "full"

        disabled = []
        if not self.use_guidance_graph:
            disabled.append("no_guidance")
        if not self.use_fusion_conv:
            disabled.append("no_spatial")
        if not self.use_bulk_prior:
            disabled.append("no_bulk")
        if not self.use_ot_deconv:
            disabled.append("no_ot")

        return "_".join(disabled) if disabled else "full"