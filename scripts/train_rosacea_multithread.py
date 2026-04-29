#!/usr/bin/env python3
"""Phase 12: TripleModalVAE training with multi-threaded CPU data loading.

Key improvements over train_rosacea_phase12.py:
1. Map-style Dataset (supports DataLoader num_workers)
2. Multi-threaded data prefetching (4 workers by default)
3. Pin_memory for faster CPU->GPU transfer
4. Optional CPU-only mode (bypass unstable GPU driver)
5. Gradient accumulation for larger effective batch size

Usage:
    # Multi-threaded GPU training:
    python scripts/train_rosacea_multithread.py --mode=train --num_workers=4

    # CPU-only training (slower but stable):
    python scripts/train_rosacea_multithread.py --mode=train --device=cpu

    # Resume from checkpoint:
    python scripts/train_rosacea_multithread.py --mode=train --resume checkpoints/rosacea/checkpoint_epoch_9.pt
"""

import argparse
import hashlib
import hmac
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Ensure src is in path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tglue.data.dataset import TripleModalDataset
from tglue.data.zarr_loader import ZarrLazyLoader
from tglue.data.rosacea_loader import load_rosacea_dataset
from tglue.data.spatial_split import spatial_quadrant_split
from tglue.data.preprocessing import preprocess_scrna, preprocess_st, preprocess_bulk
from tglue.deconv.bulk_prior import BulkPriorConfig
from tglue.deconv.label_mapping import get_canonical_cell_types, get_canonical_conditions
from tglue.models.vae import TripleModalVAE
from tglue.models.discriminator import ModalityDiscriminator
from tglue.scaffold.spatial_scaffold import SpatialScaffold, build_spatial_knn
from tglue.train.trainer import TripleModalTrainer
from tglue.graph.guidance_graph import GuidanceGraph, build_guidance_graph
from tglue.graph.genes import load_gtf_annotations, harmonize_genes

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================
N_GENES = 17_825
LATENT_DIM = 128
PRE_WARM_EPOCHS = 20
OT_EPSILON = 0.1
LR = 1e-3
SEED = 42
N_EPOCHS = 200
PATIENCE = 10
CHECKPOINT_EVERY = 5
BATCH_SIZE = 128
NUM_WORKERS = 4  # Multi-threaded data loading
ACCUMULATION_STEPS = 1  # Gradient accumulation

# Directories
CHECKPOINT_DIR = Path("checkpoints/rosacea")
LOG_DIR = Path("runs/rosacea")
MARKERS_PATH = Path("src/tglue/data/markers/skin_markers.gmt")


def set_seed(seed: int = SEED) -> None:
    """Set deterministic seed for reproducibility."""
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Deterministic seed set to {seed}")


class MapStyleTripleModalDataset(Dataset):
    """Map-style Dataset supporting DataLoader with num_workers.

    Unlike TripleModalDataset (IterableDataset), this supports:
    - __len__ and __getitem__ for DataLoader
    - Multi-threaded data loading via num_workers
    - Pin_memory for faster CPU->GPU transfer

    Each __getitem__ returns a batch dict with:
        - x_sc: Tensor (batch_size_sc, n_genes)
        - x_st: Tensor (batch_size_st, n_genes)
        - x_bulk: Tensor (n_bulk, n_genes)
        - st_indices: Tensor (batch_size_st,) for spatial loss
        - guidance_data: PyG Data (shared across all batches)
        - cell_type_onehot: Optional Tensor subsetting per batch from canonical sc cell types
        - st_condition_idx: Optional Tensor of integer condition indices matching ST row order
    """

    def __init__(
        self,
        scrna_zarr_path: str,
        st_zarr_path: str,
        bulk_zarr_path: str,
        st_indices: np.ndarray,
        gene_idx_sc: np.ndarray,
        gene_idx_st: np.ndarray,
        gene_idx_bulk: np.ndarray,
        guidance_data,
        batch_size_sc: int = 128,
        batch_size_st: int = 128,
        n_bulk: int = 58,
        shuffle: bool = True,
        seed: int = 42,
        cell_type_onehot: np.ndarray | torch.Tensor | None = None,
        st_condition_idx: np.ndarray | torch.Tensor | None = None,
    ):
        """Initialize map-style dataset.

        Parameters
        ----------
        scrna_zarr_path : str
            Path to scRNA Zarr array
        st_zarr_path : str
            Path to ST Zarr array
        bulk_zarr_path : str
            Path to Bulk Zarr array
        st_indices : np.ndarray
            Indices of ST spots to use (train or validation split)
        gene_idx_sc/st/bulk : np.ndarray
            Gene harmonization indices
        guidance_data : PyG Data
            Shared guidance graph data
        batch_size_sc/st : int
            Batch sizes per modality
        n_bulk : int
            Number of bulk samples
        shuffle : bool
            Whether to shuffle indices
        seed : int
            Random seed for shuffling
        """
        from tglue.data.zarr_loader import ZarrLazyLoader

        self.scrna_loader = ZarrLazyLoader(scrna_zarr_path, gene_idx_sc)
        self.st_loader = ZarrLazyLoader(st_zarr_path, gene_idx_st)
        self.bulk_loader = ZarrLazyLoader(bulk_zarr_path, gene_idx_bulk)

        self.st_indices = st_indices
        self.n_bulk = n_bulk
        self.guidance_data = guidance_data
        self.batch_size_sc = batch_size_sc
        self.batch_size_st = batch_size_st

        # Compute total batches
        self.n_sc = self.scrna_loader.shape[0]
        self.n_st = len(st_indices)
        self.n_batches = max(
            (self.n_sc + batch_size_sc - 1) // batch_size_sc,
            (self.n_st + batch_size_st - 1) // batch_size_st
        )

        # Generate shuffled indices per batch (deterministic)
        self.seed = seed
        self.shuffle = shuffle
        self._generate_batch_indices()

        if cell_type_onehot is not None:
            arr = np.asarray(cell_type_onehot, dtype=np.float32)
            if arr.shape[0] != self.n_sc or arr.ndim != 2:
                raise ValueError(
                    "cell_type_onehot must have shape (n_sc, n_cell_types)"
                )
            self.cell_type_onehot = arr
        else:
            self.cell_type_onehot = None

        if st_condition_idx is not None:
            idx_arr = np.asarray(st_condition_idx, dtype=np.int64).ravel()
            if idx_arr.shape[0] != self.st_loader.shape[0]:
                raise ValueError(
                    "st_condition_idx must have length equal to total ST spots"
                )
            self.st_condition_idx = idx_arr
        else:
            self.st_condition_idx = None

        logger.info(
            f"MapStyleTripleModalDataset: {self.n_batches} batches, "
            f"scRNA={self.n_sc}, ST={self.n_st}, Bulk={n_bulk}"
        )

    def _generate_batch_indices(self):
        """Generate batch indices (shuffled if requested)."""
        np.random.seed(self.seed)

        if self.shuffle:
            self.sc_indices = np.random.permutation(self.n_sc)
            self.st_shuffled = np.random.permutation(self.st_indices)
        else:
            self.sc_indices = np.arange(self.n_sc)
            self.st_shuffled = self.st_indices.copy()

    def __len__(self):
        """Return number of batches."""
        return self.n_batches

    def __getitem__(self, batch_idx: int) -> dict:
        """Get a single batch by index.

        Parameters
        ----------
        batch_idx : int
            Batch index (0 to n_batches-1)

        Returns
        -------
        dict
            Batch dict with tensors and metadata
        """
        # scRNA batch indices (with modulo wrapping for multi-epoch)
        n_sc_batches = max(1, (self.n_sc + self.batch_size_sc - 1) // self.batch_size_sc)
        sc_start = (batch_idx % n_sc_batches) * self.batch_size_sc
        sc_end = min(sc_start + self.batch_size_sc, self.n_sc)
        sc_batch_indices = self.sc_indices[sc_start:sc_end]

        # ST batch indices
        n_st_batches = max(1, (self.n_st + self.batch_size_st - 1) // self.batch_size_st)
        st_start = (batch_idx % n_st_batches) * self.batch_size_st
        st_end = min(st_start + self.batch_size_st, self.n_st)
        st_batch_indices = self.st_shuffled[st_start:st_end]

        # Load data from Zarr (CPU operation)
        x_sc = self.scrna_loader.get_batch(sc_batch_indices)
        x_st = self.st_loader.get_batch(st_batch_indices)
        x_bulk = self.bulk_loader.get_batch(np.arange(self.n_bulk))

        # Convert to tensors (on CPU, will be moved to GPU by DataLoader)
        batch = {
            "x_sc": torch.tensor(x_sc, dtype=torch.float32),
            "x_st": torch.tensor(x_st, dtype=torch.float32),
            "x_bulk": torch.tensor(x_bulk, dtype=torch.float32),
            "guidance_data": self.guidance_data,  # Shared PyG Data
            "st_indices": torch.tensor(st_batch_indices, dtype=torch.long),
            "batch_idx": batch_idx,
        }

        if self.cell_type_onehot is not None:
            batch["cell_type_onehot"] = torch.tensor(
                self.cell_type_onehot[sc_batch_indices],
                dtype=torch.float32,
            )

        if self.st_condition_idx is not None:
            batch["st_condition_idx"] = torch.tensor(
                self.st_condition_idx[st_batch_indices],
                dtype=torch.long,
            )

        return batch

    def reshuffle(self, seed: int):
        """Reshuffle indices for new epoch."""
        self.seed = seed
        self._generate_batch_indices()

    def get_expression_matrix(
        self,
        modality: str,
        obs_indices: np.ndarray | None = None,
        as_tensor: bool = False,
        device: str | None = None,
    ) -> np.ndarray | torch.Tensor:
        """Materialize expression matrix for a given modality.

        Task 6a: Added for compatibility with condition prior epoch refresh.

        Parameters
        ----------
        modality : str
            One of {"scrna", "st", "bulk"}.
        obs_indices : np.ndarray | None, default None
            Row indices to subset. If None, materializes all rows.
        as_tensor : bool, default False
            If True, return torch.Tensor instead of np.ndarray.
        device : str | None, default None
            Device for tensor placement (only used if as_tensor=True).

        Returns
        -------
        np.ndarray | torch.Tensor
            Expression matrix (n_obs or len(obs_indices), n_genes).

        Raises
        ------
        ValueError
            If modality is not one of {"scrna", "st", "bulk"}.
        """
        # Validate modality
        valid_modalities = {"scrna", "st", "bulk"}
        if modality not in valid_modalities:
            raise ValueError(
                f"Invalid modality '{modality}'. Must be one of {valid_modalities}."
            )

        # Get loader for modality
        loader_map = {
            "scrna": self.scrna_loader,
            "st": self.st_loader,
            "bulk": self.bulk_loader,
        }
        loader = loader_map[modality]

        # Determine indices
        if obs_indices is None:
            if modality == "scrna":
                indices = np.arange(loader.shape[0])
            elif modality == "st":
                indices = self.st_indices
            else:
                indices = np.arange(self.n_bulk)
        else:
            indices = np.asarray(obs_indices, dtype=np.int64).ravel()

        # Load from Zarr
        X = loader.get_batch(indices)

        # Check for finite values
        if not np.all(np.isfinite(X)):
            import logging
            logging.getLogger(__name__).warning(
                f"Expression matrix for '{modality}' contains non-finite values "
                f"(inf/nan). This may cause issues in downstream computation."
            )

        # Convert to tensor if requested
        if as_tensor:
            if device is None:
                device = "cpu"
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
            "scrna": self.scrna_loader,
            "st": self.st_loader,
            "bulk": self.bulk_loader,
        }
        loader = loader_map[modality]
        if loader is None:
            raise RuntimeError(f"No loader for modality '{modality}'.")
        if obs_indices is None:
            n_rows = loader.shape[0]
            if modality == "st":
                n_rows = len(self.st_indices)
            indices = np.arange(n_rows) if modality != "st" else np.asarray(self.st_indices, dtype=np.int64)
        else:
            indices = np.asarray(obs_indices, dtype=np.int64).ravel()
        for i in range(0, len(indices), chunk_size):
            chunk_idx = indices[i:i + chunk_size]
            X = loader.get_batch(chunk_idx)
            if as_tensor:
                dev = device or "cpu"
                yield torch.tensor(X, dtype=torch.float32, device=dev)
            else:
                yield X


def collate_fn(batch_list: list) -> dict:
    """Custom collate function for TripleModalDataset.

    Handles:
    - Tensors: stack them
    - guidance_data: return first (shared across all batches)
    - st_indices: concatenate
    """
    if len(batch_list) == 1:
        return batch_list[0]

    # Stack tensors
    result = {
        "x_sc": torch.cat([b["x_sc"] for b in batch_list], dim=0),
        "x_st": torch.cat([b["x_st"] for b in batch_list], dim=0),
        "x_bulk": batch_list[0]["x_bulk"],  # Bulk is same for all batches
        "guidance_data": batch_list[0]["guidance_data"],  # Shared
        "st_indices": torch.cat([b["st_indices"] for b in batch_list], dim=0),
    }

    def _cat_optional(key: str, dim: int = 0):
        values = [b[key] for b in batch_list if key in b and b[key] is not None]
        return torch.cat(values, dim=dim) if values else None

    cell_type_batch = _cat_optional("cell_type_onehot", dim=0)
    if cell_type_batch is not None:
        result["cell_type_onehot"] = cell_type_batch

    st_condition_batch = _cat_optional("st_condition_idx", dim=0)
    if st_condition_batch is not None:
        result["st_condition_idx"] = st_condition_batch

    return result


def _guidance_graph_hash_path(graph_path: Path) -> Path:
    """Return sidecar path that stores trusted SHA-256 for cached graph."""
    return graph_path.with_suffix(f"{graph_path.suffix}.sha256")


def _compute_file_sha256(file_path: Path) -> str:
    """Compute SHA-256 digest for a file."""
    digest = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_sha256_sidecar(hash_path: Path) -> str | None:
    """Read sidecar hash if present and valid hex digest."""
    if not hash_path.exists():
        return None
    token = hash_path.read_text(encoding="utf-8").strip().split(maxsplit=1)[0]
    if len(token) != 64:
        return None
    if any(c not in "0123456789abcdefABCDEF" for c in token):
        return None
    return token.lower()


def _load_cached_guidance_graph(graph_path: Path):
    """Load pickled graph only when sidecar hash validates trusted bytes."""
    hash_path = _guidance_graph_hash_path(graph_path)
    expected_hash = _read_sha256_sidecar(hash_path)
    if expected_hash is None:
        logger.warning(f"Guidance graph cache hash missing/invalid: {hash_path}")
        return None

    actual_hash = _compute_file_sha256(graph_path)
    if not hmac.compare_digest(actual_hash, expected_hash):
        logger.warning(
            f"Guidance graph cache hash mismatch for {graph_path}: "
            f"expected {expected_hash}, got {actual_hash}"
        )
        return None

    import pickle

    with open(graph_path, "rb") as f:
        return pickle.load(f)


def _write_guidance_graph_cache(graph_path: Path, graph) -> None:
    """Write pickled graph cache and its SHA-256 sidecar."""
    import pickle

    graph_path.parent.mkdir(parents=True, exist_ok=True)
    with open(graph_path, "wb") as f:
        pickle.dump(graph, f)

    hash_path = _guidance_graph_hash_path(graph_path)
    hash_path.write_text(f"{_compute_file_sha256(graph_path)}\n", encoding="utf-8")


def prebuild_guidance_graph() -> "GuidanceGraph":
    """Pre-build guidance graph for Rosacea data."""
    from tglue.graph.guidance_graph import GuidanceGraph

    graph_path = Path("data/rosacea/guidance_graph.pkl")
    if graph_path.exists():
        graph = _load_cached_guidance_graph(graph_path)
        if graph is not None:
            logger.info(
                f"Loaded cached guidance graph: {len(graph.gene_list)} genes, "
                f"{graph.edge_index.shape[1]} edges"
            )
            return graph
        logger.warning("Cached guidance graph failed verification; rebuilding cache.")

    # Build fresh if not cached
    logger.info("Building guidance graph...")
    from tglue.graph.genes import load_gtf_annotations
    gtf_path = "data/rosacea/mock_gtf.txt"
    scrna_path = "data/rosacea/sc_reference.zarr/X"

    import zarr
    scrna_expr = zarr.open_array(scrna_path, mode='r')

    gtf_data = load_gtf_annotations(gtf_path)
    graph = GuidanceGraph(
        gene_names=list(scrna_expr.attrs.get("var_names", [])),
        gtf_data=gtf_data,
        scrna_expr=scrna_expr[:],
        genomic_edge_weight=1.0,
        coexpr_threshold=0.3,
    )

    # Cache for reuse
    _write_guidance_graph_cache(graph_path, graph)

    return graph


def initialize_training(device: str = "cuda", online_prior_temperature: float = 0.1):
    """Initialize VAE, discriminator, trainer, and datasets."""

    # Model initialization
    vae = TripleModalVAE(
        n_genes=N_GENES,
        latent_dim=LATENT_DIM,
        enc_sc_hidden=256,
        enc_st_hidden=256,
        enc_bulk_hidden=128,
    )
    discriminator = ModalityDiscriminator(
        latent_dim=LATENT_DIM,
        hidden_dim=256,
    )

    # Spatial scaffold (CPU-safe: no CUDA ops)
    spatial_scaffold = SpatialScaffold(
        latent_dim=LATENT_DIM,
        n_neighbors=6,
        fusion_hidden=128,
    )

    # Bulk prior config
    bulk_config = BulkPriorConfig(
        lambda_start=0.01,
        lambda_max=0.1,
        warmup_start=20,
        warmup_end=40,
        online_prior_temperature=online_prior_temperature,
    )

    trainer = TripleModalTrainer(
        vae=vae,
        discriminator=discriminator,
        lr=LR,
        pre_warm_epochs=PRE_WARM_EPOCHS,
        r1_weight=1.0,
        ot_epsilon=OT_EPSILON,
        spatial_scaffold=spatial_scaffold,
        bulk_prior_config=bulk_config,
        device=device,
    )
    logger.info(f"TripleModalTrainer initialized: device={device}")

    return vae, discriminator, trainer


def setup_map_style_datasets(
    guidance_graph,
    device: str = "cuda",
    validation_quadrant: int = 2,
):
    """Setup map-style datasets for DataLoader."""

    import zarr
    from tglue.data.spatial_split import spatial_quadrant_split
    import torch_geometric.data as gd

    # Zarr paths (AnnData format: X is CSR sparse matrix)
    scrna_path = "data/rosacea/sc_reference.zarr/X"
    st_path = "data/rosacea/spatial_100k.zarr/X"
    bulk_path = "data/rosacea/array_test.zarr/X"

    # Open groups to get metadata
    scrna_group = zarr.open_group("data/rosacea/sc_reference.zarr", mode='r')
    st_group = zarr.open_group("data/rosacea/spatial_100k.zarr", mode='r')
    bulk_group = zarr.open_group("data/rosacea/array_test.zarr", mode='r')

    # Optional metadata for P2 bulk prior
    cell_type_group = scrna_group["obs"]["cell_type"]
    cell_type_codes = cell_type_group["codes"][:]
    cell_type_categories = [str(c) for c in cell_type_group["categories"][:]]
    cell_type_labels = np.array(
        [cell_type_categories[c] for c in cell_type_codes], dtype=str
    )
    canonical_cell_types = get_canonical_cell_types()
    cell_type_onehot = canonical_cell_types.to_onehot(cell_type_labels)

    st_condition_group = st_group["obs"]["condition"]
    st_condition_codes = st_condition_group["codes"][:]
    st_condition_categories = [str(c) for c in st_condition_group["categories"][:]]
    st_raw_conditions = np.array(
        [st_condition_categories[c] for c in st_condition_codes], dtype=str
    )
    canonical_conditions = get_canonical_conditions()
    st_condition_names = np.array(
        [canonical_conditions.normalize(str(cond)) for cond in st_raw_conditions],
        dtype=str,
    )
    condition_to_idx = {
        name: idx for idx, name in enumerate(canonical_conditions.names)
    }
    missing_conditions = sorted(set(st_condition_names) - set(condition_to_idx))
    if missing_conditions:
        logger.warning(
            "ST conditions outside canonical space: %s",
            ", ".join(missing_conditions),
        )
    st_condition_idx = np.array(
        [condition_to_idx.get(name, 0) for name in st_condition_names],
        dtype=np.int64,
    )

    # Get gene names from var/_index array (AnnData Zarr stores gene names there)
    scrna_genes = list(scrna_group['var']['_index'][:])
    st_genes = list(st_group['var']['_index'][:])
    bulk_genes = list(bulk_group['var']['_index'][:])

    # Harmonize genes
    from tglue.graph.genes import harmonize_genes_three_modalities
    shared_genes, gene_idx_sc, gene_idx_st, gene_idx_bulk = harmonize_genes_three_modalities(
        scrna_genes, st_genes, bulk_genes
    )

    logger.info(f"Shared genes: {len(shared_genes)}")

    # Spatial split
    coords_path = "data/rosacea/spatial_100k.zarr"
    coords_zarr = zarr.open_group(coords_path, mode='r')
    spatial_group = coords_zarr['obsm']['spatial']
    x_coords = spatial_group['x'][:]
    y_coords = spatial_group['y'][:]
    coords = np.column_stack([x_coords, y_coords])

    train_indices, val_indices = spatial_quadrant_split(
        coords, validation_fraction=0.25, validation_quadrant=validation_quadrant
    )

    logger.info(f"Train ST spots: {len(train_indices)}, Val ST spots: {len(val_indices)}")

    # Build guidance_data as PyG Data
    guidance_data = gd.Data(
        x=torch.zeros(len(shared_genes), LATENT_DIM),  # Placeholder
        edge_index=torch.tensor(guidance_graph.edge_index, dtype=torch.long),
        edge_attr=torch.tensor(guidance_graph.edge_weight, dtype=torch.float32),
    )

    # Create map-style datasets
    train_dataset = MapStyleTripleModalDataset(
        scrna_zarr_path=scrna_path,
        st_zarr_path=st_path,
        bulk_zarr_path=bulk_path,
        st_indices=train_indices,
        gene_idx_sc=gene_idx_sc,
        gene_idx_st=gene_idx_st,
        gene_idx_bulk=gene_idx_bulk,
        guidance_data=guidance_data,
        cell_type_onehot=cell_type_onehot,
        st_condition_idx=st_condition_idx,
        batch_size_sc=BATCH_SIZE,
        batch_size_st=BATCH_SIZE,
        n_bulk=bulk_group['X'].shape[0],
        shuffle=True,
        seed=SEED,
    )

    val_dataset = MapStyleTripleModalDataset(
        scrna_zarr_path=scrna_path,
        st_zarr_path=st_path,
        bulk_zarr_path=bulk_path,
        st_indices=val_indices,
        gene_idx_sc=gene_idx_sc,
        gene_idx_st=gene_idx_st,
        gene_idx_bulk=gene_idx_bulk,
        guidance_data=guidance_data,
        cell_type_onehot=cell_type_onehot,
        st_condition_idx=st_condition_idx,
        batch_size_sc=BATCH_SIZE,
        batch_size_st=BATCH_SIZE,
        n_bulk=bulk_group['X'].shape[0],
        shuffle=False,  # No shuffle for validation
        seed=SEED,
    )

    return train_dataset, val_dataset, train_indices, scrna_group, st_group, bulk_group


def train_epoch(
    trainer: TripleModalTrainer,
    dataloader: DataLoader,
    epoch: int,
    device: str,
    accumulation_steps: int = 1,
) -> dict:
    """Train one epoch with gradient accumulation."""

    trainer.vae.train()
    trainer.discriminator.train()

    epoch_losses = []
    nan_count = 0
    consecutive_nan_count = 0

    optimizer_zeroed = False

    for batch_idx, batch in enumerate(dataloader):
        # Move batch to device
        x_sc = batch["x_sc"].to(device)
        x_st = batch["x_st"].to(device)
        x_bulk = batch["x_bulk"].to(device)
        guidance_data = batch["guidance_data"]
        if hasattr(guidance_data, "to"):
            guidance_data = guidance_data.to(device)

        # Train step
        try:
            losses = trainer.train_step(batch, epoch)
        except RuntimeError as e:
            if "Non-finite" in str(e):
                nan_count += 1
                consecutive_nan_count += 1
                if nan_count <= 5:
                    logger.warning(f"NaN at batch {batch_idx} (consecutive={consecutive_nan_count}, total={nan_count})")
                if consecutive_nan_count >= 5 or nan_count >= 20:
                    logger.error(f"Aborting epoch {epoch}: too many non-finite batches (consecutive={consecutive_nan_count}, total={nan_count})")
                    break
                continue
            else:
                raise

        # Reset consecutive counter on success
        consecutive_nan_count = 0

        epoch_losses.append(losses)

        # Gradient accumulation
        if (batch_idx + 1) % accumulation_steps == 0:
            # Already stepped in train_step, no additional step needed
            pass

        # Progress logging
        if (batch_idx + 1) % 100 == 0:
            avg_loss = sum(l["vae_loss"] for l in epoch_losses[-100:]) / min(100, len(epoch_losses))
            logger.info(f"  Epoch {epoch}: batch {batch_idx + 1}/{len(dataloader)}, vae_loss={avg_loss:.4f}")

    # Compute epoch average
    avg_losses = {}
    if epoch_losses:
        keys = set()
        for l in epoch_losses:
            keys.update(l.keys())
        for key in keys:
            values = [l[key] for l in epoch_losses if key in l and isinstance(l[key], (int, float))]
            valid_values = [v for v in values if v == v]  # Filter NaN
            if valid_values:
                avg_losses[key] = sum(valid_values) / len(valid_values)

    return avg_losses


def check_parameter_contamination(trainer: TripleModalTrainer) -> tuple[int, int]:
    """Scan VAE and Discriminator parameters for non-finite values."""
    vae_bad = 0
    disc_bad = 0
    for p in trainer.vae.parameters():
        if p.is_floating_point():
            vae_bad += (~torch.isfinite(p)).sum().item()
    for p in trainer.discriminator.parameters():
        if p.is_floating_point():
            disc_bad += (~torch.isfinite(p)).sum().item()
    return vae_bad, disc_bad


def validate_epoch(
    trainer: TripleModalTrainer,
    dataloader: DataLoader,
    device: str,
) -> float:
    """Run validation epoch."""

    trainer.vae.eval()
    trainer.discriminator.eval()

    val_losses = []

    with torch.no_grad():
        for batch in dataloader:
            x_sc = batch["x_sc"].to(device)
            x_st = batch["x_st"].to(device)
            x_bulk = batch["x_bulk"].to(device)
            guidance_data = batch["guidance_data"]
            if hasattr(guidance_data, "to"):
                guidance_data = guidance_data.to(device)

            vae_output = trainer.vae(x_sc, x_st, x_bulk, guidance_data)
            recon_loss = vae_output.get("recon_loss", torch.tensor(0.0, device=device))
            kl_loss = vae_output.get("kl_loss", torch.tensor(0.0, device=device))
            val_loss = (recon_loss + kl_loss).item()

            if val_loss == val_loss:  # Not NaN
                val_losses.append(val_loss)

    if not val_losses:
        return float("nan")
    return sum(val_losses) / len(val_losses)


def save_checkpoint(
    trainer: TripleModalTrainer,
    epoch: int,
    val_loss: float,
    is_best: bool = False,
):
    """Save checkpoint."""
    import pickle

    checkpoint = {
        "epoch": epoch,
        "val_loss": val_loss,
        "vae_state_dict": trainer.vae.state_dict(),
        "disc_state_dict": trainer.discriminator.state_dict(),
        "opt_vae_state_dict": trainer.opt_vae.state_dict(),
        "opt_disc_state_dict": trainer.opt_disc.state_dict(),
        "history": trainer.history,
    }

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # Save epoch checkpoint
    epoch_path = CHECKPOINT_DIR / f"checkpoint_epoch_{epoch}.pt"
    torch.save(checkpoint, epoch_path)
    logger.info(f"Checkpoint saved: {epoch_path}")

    # Save best model
    if is_best:
        best_path = CHECKPOINT_DIR / "best_model.pt"
        torch.save(checkpoint, best_path)
        logger.info(f"Best model saved: val_loss={val_loss:.4f}")

    # Save metrics JSON
    metrics_path = CHECKPOINT_DIR / f"metrics_epoch_{epoch}.json"
    with open(metrics_path, "w") as f:
        json.dump({
            "epoch": epoch,
            "val_loss": val_loss,
            "history_last": {
                k: v[-1] if v else None
                for k, v in trainer.history.items()
            }
        }, f, indent=2)


def load_checkpoint(trainer: TripleModalTrainer, checkpoint_path: str) -> int:
    """Load checkpoint and return start epoch."""
    checkpoint = torch.load(checkpoint_path, weights_only=True)

    trainer.vae.load_state_dict(checkpoint["vae_state_dict"])
    trainer.discriminator.load_state_dict(checkpoint["disc_state_dict"])
    trainer.opt_vae.load_state_dict(checkpoint["opt_vae_state_dict"])
    trainer.opt_disc.load_state_dict(checkpoint["opt_disc_state_dict"])
    trainer.history = checkpoint["history"]

    logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    return checkpoint["epoch"]


def main():
    """Main training function."""

    parser = argparse.ArgumentParser(description="Multi-threaded TripleModalVAE training")
    parser.add_argument("--mode", type=str, default="train", choices=["init", "train"])
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume from")
    parser.add_argument("--epochs", type=int, default=N_EPOCHS, help="Maximum epochs")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS, help="DataLoader workers")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--accumulation", type=int, default=ACCUMULATION_STEPS, help="Gradient accumulation steps")
    parser.add_argument("--online_prior_temperature", type=float, default=0.1, help="Temperature for online bulk prior softmax")
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("Multi-threaded TripleModalVAE Training on Rosacea Data")
    logger.info("=" * 70)

    set_seed(SEED)

    # Device check
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        args.device = "cpu"

    logger.info(f"Device: {args.device}, Num workers: {args.num_workers}")

    # Initialize
    vae, discriminator, trainer = initialize_training(
        args.device,
        online_prior_temperature=args.online_prior_temperature,
    )
    logger.info(f"Online prior temperature: {args.online_prior_temperature}")

    # Pre-build guidance graph
    guidance_graph = prebuild_guidance_graph()

    # Setup datasets
    train_dataset, val_dataset, train_st_indices, scrna_group, st_group, bulk_group = setup_map_style_datasets(
        guidance_graph, args.device
    )

    # Spatial graph for trainer
    import zarr
    coords_zarr = zarr.open_group("data/rosacea/spatial_100k.zarr", mode='r')
    spatial_group = coords_zarr['obsm']['spatial']
    coords = np.column_stack([spatial_group['x'][:], spatial_group['y'][:]])

    import squidpy as sq
    import scipy.sparse
    import scanpy as sc
    dummy_adata = sc.AnnData(X=np.zeros((len(train_st_indices), 1)))
    dummy_adata.obsm['spatial'] = coords[train_st_indices]
    sq.gr.spatial_neighbors(dummy_adata, n_neighs=6, coord_type="grid")
    spatial_adj = scipy.sparse.csr_matrix(dummy_adata.obsp['spatial_connectivities'])

    trainer.set_spatial_graph(spatial_adj)
    logger.info(f"Spatial graph: {spatial_adj.nnz} edges")

    # Bulk proportions - Chunk 2: Use condition-level bulk prior (NEW API)
    try:
        import scanpy as sc
        from gseapy.parser import read_gmt
        from tglue.deconv.bulk_prior import preprocess_bulk_with_metadata

        bulk_adata = sc.read_h5ad("data/rosacea/array_test.h5ad")
        markers = read_gmt(str(MARKERS_PATH))

        # Use validated ssGSEA with condition aggregation
        bulk_meta = preprocess_bulk_with_metadata(
            bulk_adata, markers, condition_col="condition", sample_id_col="sample"
        )

        # Set condition-level proportions (NEW API)
        trainer.set_bulk_condition_proportions(
            bulk_meta.condition_proportions,
            bulk_meta.condition_names,
        )
        logger.info(
            f"Condition-level bulk proportions set: "
            f"conditions={bulk_meta.condition_names}, "
            f"cell_types={len(bulk_meta.cell_type_names)}"
        )

        # Set metadata for condition prior
        # Read cell type labels from Zarr obs (categorical array)
        cell_type_group = scrna_group['obs']['cell_type']
        cell_type_categories = cell_type_group['categories'][:]
        cell_type_codes = cell_type_group['codes'][:]
        # Map integer codes to actual category names for proper alignment
        cell_type_labels = np.array([str(cell_type_categories[c]) for c in cell_type_codes])

        # Read ST condition labels from Zarr obs
        condition_group = st_group['obs']['condition']
        condition_codes = condition_group['codes'][:]
        categories = list(condition_group['categories'][:])

        # Map integer codes to actual category NAMES for proper alignment with bulk
        all_st_conditions = np.array([str(categories[c]) for c in condition_codes])

        # Training-only conditions (for trainer's st_condition_labels)
        st_conditions = all_st_conditions[train_st_indices]

        # Single canonical metadata setup call (normalized path)
        trainer.set_metadata_for_condition_prior(cell_type_labels, st_conditions)

        # Full ST indices for OT deconvolution
        all_st_indices = np.arange(len(condition_codes))

        # Log condition coverage
        unique_train = set(st_conditions)
        unique_all = set(all_st_conditions)
        logger.info(
            f"Condition metadata: "
            f"scRNA cell types={len(cell_type_labels)}, "
            f"ST train conditions={sorted(unique_train)}, "
            f"ST all conditions={sorted(unique_all)}"
        )

    except Exception as e:
        logger.warning(f"Could not load bulk proportions: {e}")
        # Fallback to uniform proportions (not recommended for real training)
        trainer.set_bulk_condition_proportions(
            torch.ones(2, 16) / 16,
            ["Normal", "Rosacea"],
        )
        # Fallback: use training-only indices for OT
        all_st_indices = train_st_indices
        all_st_conditions = None

    # DataLoader with multi-threading
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,  # Each __getitem__ returns a batch already
        shuffle=False,  # Dataset handles shuffling internally
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=(args.device == "cuda"),
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=2 if args.num_workers > 0 else None,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=min(args.num_workers, 2),  # Fewer workers for validation
        collate_fn=collate_fn,
        pin_memory=(args.device == "cuda"),
    )

    logger.info(f"DataLoader: {len(train_loader)} train batches, {len(val_loader)} val batches")

    # Resume checkpoint
    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(trainer, args.resume)

    # Training loop
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        logger.info(f"Epoch {epoch}/{args.epochs} starting...")

        # Reshuffle training dataset for new epoch
        train_dataset.reshuffle(seed=SEED + epoch)

        # Chunk 1: Epoch-start condition prior refresh (using shared helper)
        from tglue.train.condition_prior import refresh_condition_prior_for_epoch
        
        refresh_condition_prior_for_epoch(
            trainer=trainer,
            dataset=train_dataset,
            epoch=epoch,
            device=args.device,
            ot_prior_start_epoch=20,
            chunk_size=512,
            all_st_indices=all_st_indices,
            all_st_conditions=all_st_conditions,
        )

        # Train
        avg_losses = train_epoch(
            trainer, train_loader, epoch, args.device, args.accumulation
        )

        # P2: Per-epoch parameter contamination scan
        vae_bad, disc_bad = check_parameter_contamination(trainer)
        if vae_bad > 0 or disc_bad > 0:
            logger.error(
                f"Epoch {epoch}: Parameter contamination detected "
                f"(VAE non-finite={vae_bad}, Disc non-finite={disc_bad}). "
                f"Rolling back to last good checkpoint."
            )
            rollback_path = CHECKPOINT_DIR / f"checkpoint_epoch_{epoch - 1}.pt"
            if rollback_path.exists():
                load_checkpoint(trainer, str(rollback_path))
                logger.info(f"Rolled back to {rollback_path}")
            else:
                logger.warning(f"No rollback checkpoint found at {rollback_path}")
            break

        # Validate
        val_loss = validate_epoch(trainer, val_loader, args.device)

        epoch_time = time.time() - epoch_start
        logger.info(
            f"Epoch {epoch}: vae_loss={avg_losses.get('vae_loss', 0):.4f}, "
            f"val_loss={val_loss:.4f}, time={epoch_time:.1f}s"
        )

        # Update history
        for k, v in avg_losses.items():
            trainer.history.setdefault(k, []).append(v)
        trainer.history.setdefault("val_loss", []).append(val_loss)

        # Checkpoint
        if not np.isfinite(val_loss):
            logger.warning("Skip checkpoint ranking: val_loss is non-finite")
            is_best = False
        else:
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                patience_counter = 0
                save_checkpoint(trainer, epoch, val_loss, is_best=True)
                logger.info(f"  -> New best model (val_loss={val_loss:.4f})")
            else:
                patience_counter += 1

        # Periodic checkpoint
        if epoch % CHECKPOINT_EVERY == 0:
            save_checkpoint(trainer, epoch, val_loss, is_best=False)

        # Early stopping
        if patience_counter >= PATIENCE:
            logger.info(f"Early stopping at epoch {epoch}")
            break

    logger.info("Training completed!")
    logger.info(f"Best val_loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    import scanpy as sc  # Import for spatial_neighbors
    main()
