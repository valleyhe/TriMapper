"""TripleModalDataset: IterableDataset for triple-modal data loading.

Loads three AnnData files (scRNA, ST, Bulk) and yields batch dicts matching
the TripleModalTrainer.train_step() contract (x_sc, x_st, x_bulk, guidance_data).

Gene harmonization follows D-04: scRNA gene list is canonical.
Guidance graph built from GTF + scRNA co-expression (D-01).

Lazy loading (LD-04, D-16): .zarr paths use ZarrLazyLoader for on-demand slicing.
H5AD fallback (D-10): .h5ad paths auto-convert to cached .zarr.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import scanpy as sc
import scipy.sparse as sp
import torch
from torch.utils.data import IterableDataset
import torch_geometric.data as gd
import zarr

from ..graph.genes import harmonize_genes, load_gtf_annotations
from ..graph.guidance_graph import build_guidance_graph, GuidanceGraph
from ..scaffold.spatial_scaffold import build_spatial_knn
from .preprocessing import preprocess_scrna, preprocess_st, preprocess_bulk, convert_h5ad_to_zarr
from .spatial_split import spatial_quadrant_split
from .zarr_loader import ZarrLazyLoader

logger = logging.getLogger(__name__)


class TripleModalDataset(IterableDataset):
    """IterableDataset for triple-modal (scRNA, ST, Bulk) data loading.

    Loads three AnnData files, harmonizes genes, builds guidance graph,
    and yields batch dicts compatible with TripleModalTrainer.

    Batch dict contract (matching trainer.py line 483-567):
        - x_sc: torch.Tensor (batch_size_sc, n_genes) — float type
        - x_st: torch.Tensor (batch_size_st, n_genes) — float type
        - x_bulk: torch.Tensor (n_bulk_samples, n_genes) — float type
        - guidance_data: PyG Data object (same for all batches)
        - recon_loss, kl_loss, graph_recon_loss: placeholder torch.tensor(0.0)
          (trainer computes these)

    Parameters
    ----------
    scRNA_path : str
        Path to scRNA AnnData .h5ad file.
    st_path : str
        Path to ST AnnData .h5ad file.
    bulk_path : str
        Path to Bulk AnnData .h5ad file.
    gtf_path : str
        Path to GTF annotation file for genomic edge construction.
    batch_size_sc : int, default 128
        Batch size for scRNA samples.
    batch_size_st : int, default 128
        Batch size for ST spots.
    is_validation : bool, default False
        Whether this dataset is for validation (affects batch sampling).
    preprocessed : bool, default False
        If True, skip QC/normalization (assume data is already processed).
    device : str, default "cpu"
        Device for tensor placement.

    Attributes
    ----------
    adata_sc : sc.AnnData
        Loaded scRNA AnnData (harmonized genes).
    adata_st : sc.AnnData
        Loaded ST AnnData (harmonized genes).
    adata_bulk : sc.AnnData
        Loaded Bulk AnnData (harmonized genes).
    guidance_data : gd.Data
        PyG Data object from GuidanceGraph.to_data().
    spatial_adj : scipy.sparse.csr_matrix
        Spatial k-NN adjacency matrix from squidpy.
    coords : np.ndarray
        Spatial coordinates (n_spots, 2).
    n_genes : int
        Number of shared genes after harmonization.
    """

    def __init__(
        self,
        scRNA_path: str,
        st_path: str,
        bulk_path: str,
        gtf_path: str,
        batch_size_sc: int = 128,
        batch_size_st: int = 128,
        is_validation: bool = False,
        preprocessed: bool = False,
        device: str = "cpu",
        validation_quadrant: int = 2,
        use_lazy_loading: bool = True,  # D-16: default True for large-scale
        guidance_graph: "GuidanceGraph | None" = None,  # Pre-built graph to reuse
    ) -> None:
        """Initialize TripleModalDataset with paths and configuration.

        Parameters
        ----------
        scRNA_path : str
            Path to scRNA AnnData .h5ad or .zarr file.
        st_path : str
            Path to ST AnnData .h5ad or .zarr file.
        bulk_path : str
            Path to Bulk AnnData .h5ad or .zarr file.
        gtf_path : str
            Path to GTF annotation file.
        batch_size_sc : int, default 128
            Batch size for scRNA samples.
        batch_size_st : int, default 128
            Batch size for ST spots.
        is_validation : bool, default False
            Validation mode flag.
        preprocessed : bool, default False
            Skip QC if True.
        device : str, default "cpu"
            Device for tensors.
        validation_quadrant : int, default 2
            Quadrant for validation split (D-05: spatial contiguous blocks).
            Options: 0=NE, 1=NW, 2=SW (default), 3=SE.
        use_lazy_loading : bool, default True
            Enable Zarr lazy loading for large-scale data (D-16, LD-04).
        guidance_graph : GuidanceGraph, optional
            Pre-built guidance graph to reuse (avoids recomputation when
            creating multiple datasets with the same data).
        """
        self.batch_size_sc = batch_size_sc
        self.batch_size_st = batch_size_st
        self.is_validation = is_validation
        self.preprocessed = preprocessed
        self.device = device
        self.gtf_path = gtf_path
        self.validation_quadrant = validation_quadrant
        self.use_lazy_loading = use_lazy_loading

        # D-10: Zarr priority + H5AD fallback
        # Store original paths for reference
        self._scrna_path_orig = scRNA_path
        self._st_path_orig = st_path
        self._bulk_path_orig = bulk_path

        # Convert .h5ad to .zarr if needed (D-10, D-11)
        if use_lazy_loading:
            # Check if path is already Zarr (ends with .zarr or .zarr/X or contains zarr)
            def is_zarr_path(path: str) -> bool:
                """Check if path is a Zarr array or group."""
                return '.zarr' in path or Path(path).suffix == '.zarr'

            if not is_zarr_path(scRNA_path):
                logger.info(f"Converting {scRNA_path} to Zarr (D-10: H5AD fallback)")
                scRNA_path = convert_h5ad_to_zarr(scRNA_path)
            if not is_zarr_path(st_path):
                logger.info(f"Converting {st_path} to Zarr (D-10: H5AD fallback)")
                st_path = convert_h5ad_to_zarr(st_path)
            if not is_zarr_path(bulk_path):
                logger.info(f"Converting {bulk_path} to Zarr (D-10: H5AD fallback)")
                bulk_path = convert_h5ad_to_zarr(bulk_path)

        # Store converted Zarr paths
        self._scrna_path = scRNA_path
        self._st_path = st_path
        self._bulk_path = bulk_path

        # Step 1: Load metadata from Zarr or AnnData
        if use_lazy_loading:
            # D-16: Load metadata only (not expression data)
            # For .zarr, read var_names and obs_names from Zarr metadata
            self._load_metadata_zarr(scRNA_path, st_path, bulk_path)
        else:
            # Original full-load mode
            logger.info(f"Loading scRNA from {scRNA_path}")
            self.adata_sc = sc.read_h5ad(scRNA_path)
            logger.info(f"Loading ST from {st_path}")
            self.adata_st = sc.read_h5ad(st_path)
            logger.info(f"Loading Bulk from {bulk_path}")
            self.adata_bulk = sc.read_h5ad(bulk_path)

        # Step 2: Preprocessing placeholder (implemented in Plan 03)
        if not preprocessed:
            self._preprocess()

        # Step 3: Gene harmonization (D-04: scRNA canonical)
        shared_genes = self._harmonize_genes_all_three()

        # Step 4: Subset AnnData to shared genes (preserve scRNA order)
        gene_idx_sc = [self.adata_sc.var_names.get_loc(g) for g in shared_genes]
        gene_idx_st = [self.adata_st.var_names.get_loc(g) for g in shared_genes]
        gene_idx_bulk = [self.adata_bulk.var_names.get_loc(g) for g in shared_genes]

        # Create subset views (AnnData does not support direct slicing by gene list)
        # We store the indices for later use in __iter__
        self._gene_idx_sc = gene_idx_sc
        self._gene_idx_st = gene_idx_st
        self._gene_idx_bulk = gene_idx_bulk

        self.n_genes = len(shared_genes)
        self._shared_genes = shared_genes

        # Step 4.1: Create lazy loaders if enabled (D-16, LD-04)
        if use_lazy_loading:
            self._lazy_loaders = {
                "scrna": ZarrLazyLoader(self._scrna_path, gene_idx_sc),
                "st": ZarrLazyLoader(self._st_path, gene_idx_st),
                "bulk": ZarrLazyLoader(self._bulk_path, gene_idx_bulk),
            }
            logger.info(
                f"TripleModalDataset initialized (lazy mode): "
                f"scRNA={self._lazy_loaders['scrna'].shape[0]} cells (lazy), "
                f"ST={self._lazy_loaders['st'].shape[0]} spots (lazy), "
                f"Bulk={self._lazy_loaders['bulk'].shape[0]} samples (lazy), "
                f"genes={self.n_genes}"
            )
        else:
            self._lazy_loaders = None
            logger.info(
                f"TripleModalDataset initialized (full-load mode): "
                f"scRNA={self.adata_sc.n_obs} cells, "
                f"ST={self.adata_st.n_obs} spots, "
                f"Bulk={self.adata_bulk.n_obs} samples, "
                f"genes={self.n_genes}"
            )

        # Guidance graph construction needs scRNA expression for co-expression edges.
        # In lazy mode, materialize only a SAMPLED subset (10K cells) to avoid
        # slow full-materialization of 76K x 17825 sparse matrix.
        # Skip if guidance_graph is provided (pre-built graph already has co-expr edges).
        if self.use_lazy_loading and self.adata_sc.X is None and guidance_graph is None:
            n_cells = self._lazy_loaders["scrna"].shape[0]
            n_sample = min(10_000, n_cells)  # 10K sample is sufficient for stable correlations
            sample_indices = np.random.choice(n_cells, n_sample, replace=False)
            logger.info(
                f"Materializing {n_sample} sampled cells for guidance graph "
                f"construction (full {n_cells} cells skipped for performance)."
            )
            X_sample = self._lazy_loaders["scrna"].get_batch(sample_indices)
            # Create a new AnnData with only the sampled cells (avoids shape mismatch)
            self.adata_sc = sc.AnnData(
                X=sp.csr_matrix(X_sample),
                obs=self.adata_sc.obs.iloc[sample_indices],
                var=self.adata_sc.var,
            )

        # Step 5: Build guidance graph (D-01: genomic + coexpr edges)
        # Use pre-built graph if provided (avoids recomputation for train/val datasets)
        if guidance_graph is not None:
            logger.info("Using pre-built guidance graph...")
            graph = guidance_graph
        else:
            # Parameters from D-08/D-09: genomic_window_bp=150_000, coexpr_threshold=0.3
            logger.info("Building guidance graph...")
            graph = build_guidance_graph(
                self.adata_sc,
                self.adata_st,
                self.adata_bulk,
                gtf_path,
                genomic_window_bp=150_000,
                coexpr_threshold=0.3,
            )
        logger.info(f"[GuidanceGraph] Genes: {len(graph.gene_list)}, "
                    f"Genomic edges: {sum(1 for t in graph.edge_type if t == 'genomic')}, "
                    f"Coexpr edges: {sum(1 for t in graph.edge_type if t == 'coexpr')}")

        # Step 5.1: Degree distribution validation (D-10)
        edge_index = graph.edge_index.numpy()
        # Count occurrences of each node in edge_index (undirected graph)
        degrees = np.bincount(edge_index.flatten(), minlength=len(graph.gene_list))
        min_degree = int(degrees.min())
        max_degree = int(degrees.max())
        mean_degree = float(degrees.mean())
        median_degree = float(np.median(degrees))
        zero_degree_count = int((degrees == 0).sum())

        logger.info(
            f"[GuidanceGraph] Degrees: min={min_degree}, max={max_degree}, "
            f"mean={mean_degree:.1f}, median={median_degree:.1f}"
        )
        logger.info(f"[GuidanceGraph] Zero-degree nodes: {zero_degree_count}")

        # Warning if genes have no edges (D-10: no disconnected nodes)
        if zero_degree_count > 0:
            logger.warning(
                f"[GuidanceGraph] Warning: {zero_degree_count} genes have no edges "
                f"({100.0 * zero_degree_count / len(graph.gene_list):.1f}% of genes)"
            )

        # Step 5.2: Edge type counts (explicit logging for D-08/D-09)
        n_genomic = sum(1 for t in graph.edge_type if t == "genomic")
        n_coexpr = sum(1 for t in graph.edge_type if t == "coexpr")
        logger.info(
            f"[GuidanceGraph] Genomic edges: {n_genomic}, Coexpr edges: {n_coexpr}"
        )

        # Step 5.3: Validate edge types per D-08/D-09 (explicit user visibility)
        has_genomic = graph.has_edge_type("genomic")
        has_coexpr = graph.has_edge_type("coexpr")

        if not has_genomic:
            raise ValueError(
                "GuidanceGraph missing genomic edges (D-08). "
                "Check GTF file and genomic_window_bp parameter."
            )
        if not has_coexpr:
            raise ValueError(
                "GuidanceGraph missing co-expression edges (D-09). "
                "Check scRNA data and coexpr_threshold parameter."
            )

        logger.info(
            "[GuidanceGraph] Validation passed: both edge types present "
            "(genomic and coexpr)"
        )

        # Step 6: Convert to PyG Data
        self.guidance_graph = graph  # Store original graph for reuse
        self.guidance_data = graph.to_data()

        # Step 7: Build spatial k-NN graph (SP-01)
        logger.info("Building spatial k-NN graph...")
        self.spatial_adj, self.coords = build_spatial_knn(self.adata_st)
        logger.info(f"[Spatial k-NN] Spots: {self.adata_st.n_obs}, "
                    f"Edges: {self.spatial_adj.nnz}")

        # Step 8: Spatial contiguous validation split (D-05)
        logger.info("Applying spatial quadrant split...")
        self.train_st_indices, self.val_st_indices = spatial_quadrant_split(
            self.coords,
            validation_fraction=0.2,
            validation_quadrant=validation_quadrant,
        )
        # Set active indices based on is_validation flag
        self.st_indices = (
            self.val_st_indices if is_validation else self.train_st_indices
        )
        logger.info(
            f"[Spatial Split] Active ST indices: {len(self.st_indices)} "
            f"({'validation' if is_validation else 'training'})"
        )

        logger.info(f"TripleModalDataset initialized: "
                    f"scRNA={self.adata_sc.n_obs} cells, "
                    f"ST={self.adata_st.n_obs} spots, "
                    f"Bulk={self.adata_bulk.n_obs} samples, "
                    f"genes={self.n_genes}")

    def _load_metadata_zarr(self, scrna_path: str, st_path: str, bulk_path: str) -> None:
        """Load metadata from Zarr arrays without loading expression data (D-16).

        Creates lightweight AnnData wrappers with obs/var metadata only,
        enabling gene harmonization and guidance graph construction without RAM spike.

        Parameters
        ----------
        scrna_path : str
            Path to scRNA Zarr array.
        st_path : str
            Path to ST Zarr array.
        bulk_path : str
            Path to Bulk Zarr array.
        """
        # For Zarr arrays created by convert_h5ad_to_zarr, we need to read metadata
        # The Zarr path points to the 'X' array inside the AnnData.zarr group
        # For metadata, we need the parent group
        logger.info(f"Loading metadata from Zarr (lazy mode)")

        # Open Zarr groups to read metadata
        # convert_h5ad_to_zarr returns path to X array (e.g., "data/scrna.zarr/X")
        # We need the parent group for obs/var (e.g., "data/scrna.zarr")
        def get_zarr_group_path(x_path: str) -> str:
            """Get parent Zarr group path from X array path."""
            if '/X' in x_path:
                return x_path.replace('/X', '')
            return x_path

        # Read metadata from Zarr groups
        scrna_group_path = get_zarr_group_path(scrna_path)
        st_group_path = get_zarr_group_path(st_path)
        bulk_group_path = get_zarr_group_path(bulk_path)

        # Create lightweight AnnData wrappers
        # Note: For synthetic test Zarr files (no AnnData.zarr structure),
        # we need to handle the case where obs/var don't exist

        def load_zarr_metadata(zarr_group_path: str, x_path: str, load_spatial: bool = False) -> sc.AnnData:
            """Load metadata from Zarr group, create AnnData wrapper.

            Parameters
            ----------
            zarr_group_path : str
                Path to Zarr group directory.
            x_path : str
                Path to X array inside the group.
            load_spatial : bool
                If True, also load obsm/spatial coordinates (needed for ST data).

            Returns
            -------
            sc.AnnData
                Lightweight AnnData with obs, var, and optionally obsm/spatial.
            """
            # Try opening as group first (AnnData.zarr format)
            try:
                z_root = zarr.open_group(zarr_group_path, mode='r')

                # Check if this is AnnData.zarr format (has obs, var)
                if 'obs' in z_root and 'var' in z_root:
                    # AnnData.zarr format: use AnnData to read backed mode
                    # This avoids manual Zarr group parsing complexity
                    try:
                        adata_backed = sc.read_h5ad(zarr_group_path, backed='r')
                        # Create lightweight AnnData with metadata only (no X data)
                        adata = sc.AnnData(
                            X=None,
                            obs=adata_backed.obs.copy(),
                            var=adata_backed.var.copy(),
                        )
                        # Load obsm/spatial if requested
                        if load_spatial and 'spatial' in adata_backed.obsm:
                            adata.obsm['spatial'] = adata_backed.obsm['spatial'].copy()
                        adata_backed.file.close()
                        return adata
                    except Exception:
                        # Fallback to manual parsing if backed mode fails
                        pass

                    # Manual parsing fallback
                    var_index = z_root.get('var/_index')
                    obs_index = z_root.get('obs/_index')
                    if var_index is not None and obs_index is not None:
                        n_obs = len(obs_index)
                        n_vars = len(var_index)
                        var_names = var_index[:]
                        obs_names = obs_index[:]
                    else:
                        # Get dimensions from X if available
                        x_arr = z_root.get('X')
                        if x_arr is not None:
                            n_obs, n_vars = x_arr.shape
                        else:
                            n_obs, n_vars = 0, 0
                        var_names = None
                        obs_names = None

                    adata = sc.AnnData(X=None, shape=(n_obs, n_vars))
                    if var_names is not None:
                        adata.var_names = var_names
                    else:
                        adata.var_names = [f"GENE_{i:03d}" for i in range(1, n_vars + 1)]
                    if obs_names is not None:
                        adata.obs_names = obs_names
                    else:
                        adata.obs_names = [f"cell_{i:03d}" for i in range(1, n_obs + 1)]

                    # Load obsm/spatial if requested
                    if load_spatial and 'obsm' in z_root and 'spatial' in z_root['obsm']:
                        obsm_spatial = z_root['obsm']['spatial']
                        # AnnData.zarr stores spatial as 2D array (n_obs, 2), not as group
                        if isinstance(obsm_spatial, zarr.Array):
                            spatial = np.asarray(obsm_spatial[:])
                            adata.obsm['spatial'] = spatial
                        else:
                            # Legacy format with x/y fields (if present)
                            x_coords = obsm_spatial['x'][:]
                            y_coords = obsm_spatial['y'][:]
                            spatial = np.column_stack([x_coords, y_coords])
                            adata.obsm['spatial'] = spatial

                    return adata
            except zarr.errors.ContainsArrayError:
                # Plain Zarr array (no group structure)
                pass

            # Plain Zarr array (synthetic test fixture or direct X array)
            # Open as array and generate default metadata
            z_x = zarr.open_array(x_path, mode='r')
            n_obs, n_vars = z_x.shape
            adata = sc.AnnData(X=None, shape=(n_obs, n_vars))
            adata.var_names = [f"GENE_{i:03d}" for i in range(1, n_vars + 1)]
            adata.obs_names = [f"cell_{i:03d}" for i in range(1, n_obs + 1)]

            # Generate synthetic spatial coordinates for ST data if needed
            # (plain Zarr arrays don't have obsm structure)
            if load_spatial:
                logger.info(
                    f"Generating synthetic spatial coordinates for {n_obs} spots "
                    f"(plain Zarr array has no obsm structure)"
                )
                # Generate grid-like coordinates for spatial split
                np.random.seed(42)  # Deterministic for testing
                adata.obsm['spatial'] = np.random.randint(0, 1001, size=(n_obs, 2), dtype=np.int32)

            return adata

        self.adata_sc = load_zarr_metadata(scrna_group_path, scrna_path)
        logger.info(f"Loaded scRNA metadata from Zarr: {self.adata_sc.n_obs} cells, {self.adata_sc.n_vars} genes")

        self.adata_st = load_zarr_metadata(st_group_path, st_path, load_spatial=True)
        logger.info(f"Loaded ST metadata from Zarr: {self.adata_st.n_obs} spots, {self.adata_st.n_vars} genes")

        self.adata_bulk = load_zarr_metadata(bulk_group_path, bulk_path)
        logger.info(f"Loaded Bulk metadata from Zarr: {self.adata_bulk.n_obs} samples, {self.adata_bulk.n_vars} genes")

        # Initialize lazy loaders (will be populated after gene harmonization)
        self._lazy_loaders: Dict[str, ZarrLazyLoader] = {}

    def _preprocess(self) -> None:
        """Apply QC and normalization to each modality.

        Following D-02 (QC defaults), D-03 (normalization), D-04 (logging).
        """
        logger.info("[Preprocessing] Starting QC and normalization...")

        self.adata_sc = preprocess_scrna(self.adata_sc)
        self.adata_st = preprocess_st(self.adata_st)
        self.adata_bulk = preprocess_bulk(self.adata_bulk)

        logger.info("[Preprocessing] Completed QC and normalization")

    def _harmonize_genes_all_three(self) -> List[str]:
        """Harmonize genes across all three modalities with detailed logging.

        D-06: scRNA gene list is canonical.
        D-07: min_shared >= 2000 genes threshold.
        Returns intersection of scRNA, ST, and Bulk genes.
        Raises ValueError if shared < 2000 with explanatory message.

        Returns
        -------
        List[str]
            Shared gene list in scRNA canonical order.
        """
        sc_genes = list(self.adata_sc.var_names)
        st_genes = list(self.adata_st.var_names)
        bulk_genes = list(self.adata_bulk.var_names)

        # Log input counts per modality (D-04)
        logger.info(
            "[Gene Harmonization] Input counts: scRNA=%d, ST=%d, Bulk=%d",
            len(sc_genes),
            len(st_genes),
            len(bulk_genes),
        )

        # First intersection: scRNA <-> ST
        common_sc_st, st_filtered = harmonize_genes(sc_genes, st_genes, min_shared=2000)
        logger.info(
            "[Gene Harmonization] scRNA-ST intersection: %d genes",
            len(common_sc_st),
        )

        # Second intersection: (scRNA-ST) <-> Bulk
        sc_st_set = set(common_sc_st)
        bulk_set = set(bulk_genes)
        shared_genes = [g for g in common_sc_st if g in bulk_set]

        # Log final shared count
        logger.info(
            "[Gene Harmonization] Final shared genes across all three: %d",
            len(shared_genes),
        )

        # Check threshold with explanatory message (D-07)
        if len(shared_genes) < 2000:
            raise ValueError(
                f"Only {len(shared_genes)} shared genes after harmonizing all "
                f"three modalities; minimum 2000 required (D-07). "
                f"Smallest intersection: scRNA-ST={len(common_sc_st)}, "
                f"scRNA-Bulk={len(sc_st_set & bulk_set)}"
            )

        return shared_genes

    def __iter__(self):
        """Iterate over batches, yielding batch dicts matching trainer contract.

        DS-03 FIX: Thread-safe iteration with worker partitioning for multi-worker
        DataLoader. Each worker gets a distinct subset of indices.

        Yields
        ------
        dict[str, Any]
            Batch dict with keys:
            - x_sc: Tensor (batch_size_sc, n_genes)
            - x_st: Tensor (batch_size_st, n_genes)
            - x_bulk: Tensor (n_bulk_samples, n_genes)
            - guidance_data: PyG Data
            - recon_loss, kl_loss, graph_recon_loss: placeholder tensors
        """
        # DS-03 FIX: Worker info check for thread safety
        worker_info = torch.utils.data.get_worker_info()

        if self.use_lazy_loading and hasattr(self, '_lazy_loaders') and self._lazy_loaders:
            # Lazy mode: use ZarrLazyLoader for batch slicing (LD-04, D-16)
            n_sc = self._lazy_loaders["scrna"].shape[0]
            n_bulk = self._lazy_loaders["bulk"].shape[0]

            # DS-03 FIX: Partition indices across workers for thread safety
            if worker_info is not None:
                # Worker process: use worker-specific subset
                worker_id = worker_info.id
                num_workers = worker_info.num_workers

                # Shuffle with worker-specific seed for reproducibility
                if not self.is_validation:
                    np.random.seed(worker_id + 42)
                    sc_indices = np.random.permutation(n_sc)
                    st_indices = np.random.permutation(self.st_indices)
                else:
                    sc_indices = np.arange(n_sc)
                    st_indices = self.st_indices.copy()

                # Partition: each worker gets a slice
                n_st = len(st_indices)
                per_worker = max(1, n_st // num_workers)
                worker_start = worker_id * per_worker
                worker_end = min(worker_start + per_worker, n_st) if worker_id < num_workers - 1 else n_st
                st_indices = st_indices[worker_start:worker_end]

                # Similar partitioning for sc_indices
                n_sc_actual = len(sc_indices)
                per_worker_sc = max(1, n_sc_actual // num_workers)
                worker_start_sc = worker_id * per_worker_sc
                worker_end_sc = min(worker_start_sc + per_worker_sc, n_sc_actual) if worker_id < num_workers - 1 else n_sc_actual
                sc_indices = sc_indices[worker_start_sc:worker_end_sc]
            else:
                # Single-process: use full indices
                if self.is_validation:
                    sc_indices = np.arange(n_sc)
                    st_indices = self.st_indices
                else:
                    sc_indices = np.random.permutation(n_sc)
                    st_indices = np.random.permutation(self.st_indices)

            # n_st must match the actual st_indices length (not full Zarr shape),
            # otherwise batch counts are wrong and indices wrap out-of-bounds.
            n_st = len(st_indices)

            # Compute number of batches
            n_batches_sc = max(1, (n_sc + self.batch_size_sc - 1) // self.batch_size_sc)
            n_batches_st = max(1, (n_st + self.batch_size_st - 1) // self.batch_size_st)
            max_batches = max(n_batches_sc, n_batches_st)

            # Iterate through batches with modulo wrapping
            for batch_idx in range(max_batches):
                # scRNA batch (with modulo wrapping)
                sc_batch_start = (batch_idx % n_batches_sc) * self.batch_size_sc
                sc_batch_end = min(sc_batch_start + self.batch_size_sc, n_sc)
                sc_batch_indices = sc_indices[sc_batch_start:sc_batch_end]

                # ST batch (with modulo wrapping) — indices map to training split
                st_batch_start = (batch_idx % n_batches_st) * self.batch_size_st
                st_batch_end = min(st_batch_start + self.batch_size_st, n_st)
                st_batch_indices = st_indices[st_batch_start:st_batch_end]

                # Lazy batch read from Zarr (on-demand slicing)
                x_sc_batch = self._lazy_loaders["scrna"].get_batch(sc_batch_indices)
                x_st_batch = self._lazy_loaders["st"].get_batch(st_batch_indices)
                x_bulk_batch = self._lazy_loaders["bulk"].get_batch(np.arange(n_bulk))

                # Convert to tensors
                x_sc_tensor = torch.tensor(x_sc_batch, dtype=torch.float32, device=self.device)
                x_st_tensor = torch.tensor(x_st_batch, dtype=torch.float32, device=self.device)
                x_bulk_tensor = torch.tensor(x_bulk_batch, dtype=torch.float32, device=self.device)

                # Build batch dict (same contract as original)
                batch = {
                    "x_sc": x_sc_tensor,
                    "x_st": x_st_tensor,
                    "x_bulk": x_bulk_tensor,
                    "guidance_data": self.guidance_data,
                    "recon_loss": torch.tensor(0.0, dtype=torch.float32, device=self.device),
                    "kl_loss": torch.tensor(0.0, dtype=torch.float32, device=self.device),
                    "graph_recon_loss": torch.tensor(0.0, dtype=torch.float32, device=self.device),
                    "st_indices": torch.tensor(st_batch_indices, dtype=torch.long, device=self.device),
                    "sc_indices": torch.tensor(sc_batch_indices, dtype=torch.long, device=self.device),
                }

                yield batch
        else:
            # Original full-load mode (backward compatibility)
            # Get expression matrices (harmonized gene subset)
            X_sc = self._get_expression_matrix(self.adata_sc, self._gene_idx_sc)
            X_st = self._get_expression_matrix(self.adata_st, self._gene_idx_st)
            X_bulk = self._get_expression_matrix(self.adata_bulk, self._gene_idx_bulk)

            # Convert to tensors
            x_bulk_tensor = torch.tensor(X_bulk, dtype=torch.float32, device=self.device)

            # Shuffle indices for training (no shuffle for validation)
            n_sc = X_sc.shape[0]

            # DS-03 FIX: Worker partitioning for thread safety in full-load mode
            if worker_info is not None:
                worker_id = worker_info.id
                num_workers = worker_info.num_workers

                if not self.is_validation:
                    np.random.seed(worker_id + 42)
                    sc_indices = np.random.permutation(n_sc)
                    st_indices = np.random.permutation(self.st_indices)
                else:
                    sc_indices = np.arange(n_sc)
                    st_indices = self.st_indices.copy()

                # Partition indices across workers
                n_st_total = len(st_indices)
                per_worker = max(1, n_st_total // num_workers)
                worker_start = worker_id * per_worker
                worker_end = min(worker_start + per_worker, n_st_total) if worker_id < num_workers - 1 else n_st_total
                st_indices = st_indices[worker_start:worker_end]

                n_sc_total = len(sc_indices)
                per_worker_sc = max(1, n_sc_total // num_workers)
                worker_start_sc = worker_id * per_worker_sc
                worker_end_sc = min(worker_start_sc + per_worker_sc, n_sc_total) if worker_id < num_workers - 1 else n_sc_total
                sc_indices = sc_indices[worker_start_sc:worker_end_sc]
            else:
                # Single-process: use full indices
                if self.is_validation:
                    sc_indices = np.arange(n_sc)
                    st_indices = self.st_indices  # Use spatial split indices
                else:
                    sc_indices = np.random.permutation(n_sc)
                    st_indices = np.random.permutation(self.st_indices)  # Shuffle spatial split indices

            n_st = len(st_indices)

            # Compute number of batches
            n_batches_sc = max(1, (n_sc + self.batch_size_sc - 1) // self.batch_size_sc)
            n_batches_st = max(1, (n_st + self.batch_size_st - 1) // self.batch_size_st)
            max_batches = max(n_batches_sc, n_batches_st)

            # Iterate through batches with modulo wrapping for smaller modality
            for batch_idx in range(max_batches):
                # scRNA batch (with modulo wrapping)
                sc_batch_start = (batch_idx % n_batches_sc) * self.batch_size_sc
                sc_batch_end = min(sc_batch_start + self.batch_size_sc, n_sc)
                sc_batch_indices = sc_indices[sc_batch_start:sc_batch_end]

                # ST batch (with modulo wrapping)
                st_batch_start = (batch_idx % n_batches_st) * self.batch_size_st
                st_batch_end = min(st_batch_start + self.batch_size_st, n_st)
                st_batch_indices = st_indices[st_batch_start:st_batch_end]

                # Slice expression matrices
                x_sc_batch = X_sc[sc_batch_indices]
                x_st_batch = X_st[st_batch_indices]

                # Convert to tensors
                x_sc_tensor = torch.tensor(x_sc_batch, dtype=torch.float32, device=self.device)
                x_st_tensor = torch.tensor(x_st_batch, dtype=torch.float32, device=self.device)

                # Build batch dict matching trainer contract
                batch = {
                    "x_sc": x_sc_tensor,
                    "x_st": x_st_tensor,
                    "x_bulk": x_bulk_tensor,  # All bulk samples, not batched
                    "guidance_data": self.guidance_data,
                    "recon_loss": torch.tensor(0.0, dtype=torch.float32, device=self.device),
                    "kl_loss": torch.tensor(0.0, dtype=torch.float32, device=self.device),
                    "graph_recon_loss": torch.tensor(0.0, dtype=torch.float32, device=self.device),
                    # BF-02: Global indices of ST spots in this batch (for SpatialAwareLoss)
                    "st_indices": torch.tensor(st_batch_indices, dtype=torch.long, device=self.device),
                    "sc_indices": torch.tensor(sc_batch_indices, dtype=torch.long, device=self.device),
                }

                yield batch

    def _get_expression_matrix(self, adata, gene_indices: List[int]) -> np.ndarray:
        """Extract expression matrix for harmonized genes.

        Parameters
        ----------
        adata : sc.AnnData
            AnnData object with expression data.
        gene_indices : List[int]
            Indices of harmonized genes.

        Returns
        -------
        np.ndarray
            Expression matrix (n_obs, n_genes) as dense array.
        """
        X = adata.X[:, gene_indices]
        if hasattr(X, "toarray"):
            X = X.toarray()
        return np.asarray(X, dtype=np.float32)

    def get_expression_matrix(
        self,
        modality: str,
        obs_indices: np.ndarray | List[int] | None = None,
        as_tensor: bool = False,
        device: str | None = None,
    ) -> np.ndarray | torch.Tensor:
        """Materialize expression matrix for a given modality (D-16, Chunk 1).

        This is the public API for epoch-level condition prior refresh.
        Handles both lazy-loading (Zarr) and in-memory modes.

        Parameters
        ----------
        modality : str
            One of {"scrna", "st", "bulk"}.
        obs_indices : np.ndarray | List[int] | None, default None
            Row indices to subset. If None, materializes all rows.
        as_tensor : bool, default False
            If True, return torch.Tensor instead of np.ndarray.
        device : str | None, default None
            Device for tensor placement (only used if as_tensor=True).
            Defaults to self.device if None.

        Returns
        -------
        np.ndarray | torch.Tensor
            Expression matrix (n_obs or len(obs_indices), n_genes).

        Raises
        ------
        ValueError
            If modality is not one of {"scrna", "st", "bulk"}.
        RuntimeError
            If expression data cannot be materialized (e.g., missing lazy loader).
        """
        # Validate modality
        valid_modalities = {"scrna", "st", "bulk"}
        if modality not in valid_modalities:
            raise ValueError(
                f"Invalid modality '{modality}'. Must be one of {valid_modalities}."
            )

        # Default device
        if device is None:
            device = self.device

        # Handle indices
        if obs_indices is None:
            # Materialize all rows
            indices = None
        else:
            indices = np.asarray(obs_indices, dtype=np.int64).ravel()

        # Lazy mode: use ZarrLazyLoader
        if self.use_lazy_loading and self._lazy_loaders is not None:
            loader = self._lazy_loaders[modality]
            if loader is None:
                raise RuntimeError(
                    f"No lazy loader for modality '{modality}'. "
                    f"Check dataset initialization."
                )

            if indices is None:
                # Materialize all rows
                all_indices = np.arange(loader.shape[0])
                X = loader.get_batch(all_indices)
            else:
                X = loader.get_batch(indices)

        # Non-lazy mode: use in-memory AnnData.X
        else:
            adata_map = {
                "scrna": self.adata_sc,
                "st": self.adata_st,
                "bulk": self.adata_bulk,
            }
            adata = adata_map[modality]

            if adata.X is None:
                raise RuntimeError(
                    f"AnnData.X is None for modality '{modality}'. "
                    f"In non-lazy mode, expression data should be loaded."
                )

            gene_idx_map = {
                "scrna": self._gene_idx_sc,
                "st": self._gene_idx_st,
                "bulk": self._gene_idx_bulk,
            }
            gene_indices = gene_idx_map[modality]

            if indices is None:
                X = self._get_expression_matrix(adata, gene_indices)
            else:
                # Subset rows then genes
                X_subset = adata.X[indices][:, gene_indices]
                if hasattr(X_subset, "toarray"):
                    X_subset = X_subset.toarray()
                X = np.asarray(X_subset, dtype=np.float32)

        # Check for finite values
        if not np.all(np.isfinite(X)):
            logger.warning(
                f"Expression matrix for '{modality}' contains non-finite values "
                f"(inf/nan). This may cause issues in downstream computation."
            )

        # Convert to tensor if requested
        if as_tensor:
            return torch.tensor(X, dtype=torch.float32, device=device)

        return X

    def iter_expression_chunks(
        self,
        modality: str,
        obs_indices: np.ndarray | list[int] | None = None,
        chunk_size: int = 512,
        as_tensor: bool = False,
        device: str | None = None,
    ):
        """Yield expression chunks preserving row order.

        Parameters
        ----------
        modality : str
            One of {"scrna", "st", "bulk"}.
        obs_indices : np.ndarray | list[int] | None, default None
            Row indices to iterate over. If None, all rows.
        chunk_size : int, default 512
            Rows per yielded chunk.
        as_tensor : bool, default False
            If True, yield torch.Tensor.
        device : str | None, default None
            Device for tensor placement when as_tensor=True.
        """
        valid_modalities = {"scrna", "st", "bulk"}
        if modality not in valid_modalities:
            raise ValueError(f"Invalid modality '{modality}'.")
        loader_map = {
            "scrna": self._lazy_loaders["scrna"] if hasattr(self, "_lazy_loaders") else None,
            "st": self._lazy_loaders["st"] if hasattr(self, "_lazy_loaders") else None,
            "bulk": self._lazy_loaders["bulk"] if hasattr(self, "_lazy_loaders") else None,
        }
        # For MapStyleTripleModalDataset use its own loaders
        if hasattr(self, "scrna_loader"):
            loader_map = {"scrna": self.scrna_loader, "st": self.st_loader, "bulk": self.bulk_loader}
        loader = loader_map[modality]
        if loader is None:
            raise RuntimeError(f"No loader for modality '{modality}'.")
        if obs_indices is None:
            n_rows = loader.shape[0]
            if modality == "st" and hasattr(self, "st_indices"):
                n_rows = len(self.st_indices)
            indices = np.arange(n_rows) if modality != "st" else np.asarray(self.st_indices, dtype=np.int64)
        else:
            indices = np.asarray(obs_indices, dtype=np.int64).ravel()
        for i in range(0, len(indices), chunk_size):
            chunk_idx = indices[i:i + chunk_size]
            X = loader.get_batch(chunk_idx)
            if as_tensor:
                dev = device or getattr(self, "device", "cpu")
                yield torch.tensor(X, dtype=torch.float32, device=dev)
            else:
                yield X
