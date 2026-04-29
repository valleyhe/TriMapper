"""Zarr-backed lazy loading for large-scale gene expression data.

Implements LD-01: On-demand slicing without full RAM load.
"""

from __future__ import annotations

import logging

import numpy as np
import scipy.sparse as sp
import zarr

logger = logging.getLogger(__name__)


class ZarrLazyLoader:
    """Lazy expression matrix loader using Zarr (D-10, D-16).

    Stores path and metadata only. Data is read on-demand via slicing,
    keeping peak RAM proportional to batch size, not dataset size.

    Parameters
    ----------
    zarr_path : str
        Path to Zarr array directory (lazy proxy created here).
    gene_indices : list[int] | np.ndarray
        Indices of harmonized genes to subset.

    Attributes
    ----------
    z : zarr.Array
        Lazy array proxy (no data read until slicing).
    gene_indices : np.ndarray
        Gene indices for harmonization.
    n_cells : int
        Total cells in dataset.
    n_genes : int
        Number of harmonized genes.

    Examples
    --------
    >>> loader = ZarrLazyLoader("data/scrna.zarr", gene_indices=[0, 1, 2])
    >>> batch = loader.get_batch(np.array([0, 10, 100]))  # Reads only 3 cells
    >>> batch.shape
    (3, 3)
    """

    def __init__(self, zarr_path: str, gene_indices: list[int] | np.ndarray):
        """Initialize lazy loader with path and gene indices (D-16).

        Creates lazy Zarr proxy - no data is loaded at this point.
        Only metadata (shape, chunks, dtype) is accessible.

        Parameters
        ----------
        zarr_path : str
            Path to Zarr array directory (can be plain array or X inside AnnData.zarr).
        gene_indices : list[int] | np.ndarray
            Indices of genes to subset (for harmonization).
        """
        # Handle both plain Zarr arrays and AnnData.zarr/X arrays
        # zarr_path might be "/path/to/data.zarr/X" (array inside group)
        # or "/path/to/data.zarr" (plain array)
        self.z = None
        self._csr_group = None
        self._csr_data = None
        self._csr_indices = None
        self._csr_indptr = None

        try:
            # Direct Zarr array: dense matrix or explicit X array path.
            self.z = zarr.open_array(zarr_path, mode='r')
        except zarr.errors.ContainsGroupError:
            group = zarr.open_group(zarr_path, mode='r')
            self._configure_from_group(group, zarr_path)

        # Store gene indices for harmonized subset
        self.gene_indices = np.asarray(gene_indices, dtype=np.int64)
        self._gene_lookup = {gene_idx: i for i, gene_idx in enumerate(self.gene_indices.tolist())}

        # Metadata from Zarr array
        if self.z is not None:
            shape = self.z.shape
            chunks = self.z.chunks
        else:
            shape = tuple(self._csr_group.attrs["shape"])
            chunks = None

        self.n_cells = shape[0]
        self.n_genes = len(self.gene_indices)

        logger.debug(
            f"ZarrLazyLoader initialized: path={zarr_path}, "
            f"n_cells={self.n_cells}, n_genes={self.n_genes}, "
            f"chunks={chunks}"
        )

    def _configure_from_group(self, group, zarr_path: str) -> None:
        """Resolve a Zarr group into either dense X storage or CSR sparse storage."""
        encoding_type = group.attrs.get("encoding-type")

        if encoding_type == "csr_matrix":
            self._csr_group = group
            self._csr_data = group["data"]
            self._csr_indices = group["indices"]
            self._csr_indptr = group["indptr"]
            return

        if "X" in group:
            x = group["X"]
            if isinstance(x, zarr.Array):
                self.z = x
                return

            x_encoding = x.attrs.get("encoding-type")
            if x_encoding == "csr_matrix":
                self._csr_group = x
                self._csr_data = x["data"]
                self._csr_indices = x["indices"]
                self._csr_indptr = x["indptr"]
                return

        raise ValueError(f"Unsupported Zarr layout at {zarr_path}")

    def get_batch(self, cell_indices: np.ndarray) -> np.ndarray:
        """Lazy slice: reads only requested cells and genes (LD-01).

        Zarr slicing reads only chunks containing the requested indices.
        Peak RAM proportional to batch size, not dataset size.

        Parameters
        ----------
        cell_indices : np.ndarray
            Array of cell indices to read (1D int array).

        Returns
        -------
        np.ndarray
            Expression matrix (len(cell_indices), n_genes) as float32.

        Raises
        ------
        IndexError
            If cell_indices are out of bounds (WR-05 FIX).
        """
        # Ensure cell_indices is 1D array
        cell_indices = np.asarray(cell_indices, dtype=np.int64).ravel()

        # WR-05 FIX: Add bounds checking before array access
        if len(cell_indices) > 0:
            if (cell_indices >= self.n_cells).any() or (cell_indices < 0).any():
                raise IndexError(
                    f"cell_indices out of bounds: valid range [0, {self.n_cells-1}], "
                    f"got range [{cell_indices.min()}, {cell_indices.max()}]"
                )
        
        if self.z is not None:
            # Lazy slice: reads only requested chunks from disk.
            X_batch = self.z[cell_indices][:, self.gene_indices]
            return np.asarray(X_batch, dtype=np.float32)

        # Sparse CSR fallback for AnnData Zarr stores where X is encoded as a group.
        # Batch sizes are small, so reconstructing selected rows as a CSR matrix is
        # acceptable and avoids assuming that X is always a dense Zarr array.
        indptr = np.asarray(self._csr_indptr[:], dtype=np.int64)
        batch_rows = []
        batch_cols = []
        batch_data = []

        for out_row, src_row in enumerate(cell_indices.tolist()):
            start = indptr[src_row]
            end = indptr[src_row + 1]
            src_cols = np.asarray(self._csr_indices[start:end], dtype=np.int64)
            src_vals = np.asarray(self._csr_data[start:end])

            for col, value in zip(src_cols.tolist(), src_vals.tolist()):
                mapped_col = self._gene_lookup.get(col)
                if mapped_col is None:
                    continue
                batch_rows.append(out_row)
                batch_cols.append(mapped_col)
                batch_data.append(value)

        batch_shape = (len(cell_indices), self.n_genes)
        X_batch = sp.csr_matrix(
            (batch_data, (batch_rows, batch_cols)),
            shape=batch_shape,
        )
        return np.asarray(X_batch.toarray(), dtype=np.float32)

    @property
    def shape(self) -> tuple[int, int]:
        """Return (n_cells, n_genes) shape of harmonized matrix."""
        return (self.n_cells, self.n_genes)

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"ZarrLazyLoader(n_cells={self.n_cells}, n_genes={self.n_genes}, "
            f"chunks={self.z.chunks if self.z is not None else None})"
        )
