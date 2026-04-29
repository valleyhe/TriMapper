"""BatchListLoader: Adapter to make batch list iterable like DataLoader.

Used by AblationRunner to pass List[Dict] batches to evaluate_alignment()
which expects a DataLoader-like iterable.
"""

from typing import Any, Dict, Iterator, List


class BatchListLoader:
    """Adapter to make batch list iterable like DataLoader.

    Parameters:
        batches: List of batch dicts from TripleModalDataset

    Usage:
        val_loader = BatchListLoader(val_batches)
        alignment_metrics = evaluate_alignment(vae, val_loader, guidance_data)
    """

    def __init__(self, batches: List[Dict[str, Any]]):
        self.batches = batches

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        return iter(self.batches)

    def __len__(self) -> int:
        return len(self.batches)