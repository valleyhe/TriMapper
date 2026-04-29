"""Results output pipeline for Triple-Modal VAE (D-04 facade pattern)."""

from .pipeline import ResultsPipeline
from .pipeline_dual import ResultsPipelineDual
from .pipeline_single import ResultsPipelineSingle

__all__ = ["ResultsPipeline", "ResultsPipelineDual", "ResultsPipelineSingle"]
