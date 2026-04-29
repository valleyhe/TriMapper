#!/usr/bin/env python3
"""CLI entry point for ResultsPipeline facade (Approach B).

Wraps src/tglue/results/pipeline.py with argparse, matching the same CLI
interface as scripts/generate_all_results.py (Approach A).

Usage:
    python scripts/run_results_pipeline.py
    python scripts/run_results_pipeline.py --device cpu --output-dir results/rosacea/full_outputs
    python scripts/run_results_pipeline.py --test-smoke --device cpu
"""

import argparse
import logging
import sys
import tempfile
from pathlib import Path

import scanpy as sc

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tglue.results import ResultsPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Generate all result outputs via ResultsPipeline (Approach B)",
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/rosacea/best_model.pt",
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--spatial",
        default="data/rosacea/spatial_100k.h5ad",
        help="Path to spatial .h5ad file",
    )
    parser.add_argument(
        "--scrna",
        default="data/rosacea/sc_reference.h5ad",
        help="Path to scRNA .h5ad file",
    )
    parser.add_argument(
        "--bulk",
        default="data/rosacea/array_test.h5ad",
        help="Path to bulk .h5ad file",
    )
    parser.add_argument(
        "--output-dir",
        default="results/rosacea/full_outputs",
        help="Output directory",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Torch device",
    )
    parser.add_argument(
        "--leiden-resolution",
        type=float,
        default=1.0,
        help="Leiden clustering resolution",
    )
    parser.add_argument(
        "--spot-size",
        type=float,
        default=1.0,
        help="Default spot size for scatter plots",
    )
    parser.add_argument(
        "--test-smoke",
        action="store_true",
        help="Run with 500-obs subsets for quick validation",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("ResultsPipeline (Approach B) -- Triple-Modal VAE")
    logger.info("=" * 60)

    spatial_path = args.spatial
    scrna_path = args.scrna
    bulk_path = args.bulk
    tmpdir = None

    # Handle --test-smoke: subset data to 500 obs, write to temp files
    if args.test_smoke:
        import numpy as np

        logger.info("SMOKE TEST: subsetting to 500 obs each")
        tmpdir = tempfile.mkdtemp(prefix="pipeline_smoke_")

        st_adata = sc.read_h5ad(args.spatial)
        st_adata.obsm["spatial"] = np.asarray(st_adata.obsm["spatial"])
        st_adata = st_adata[:500].copy()
        st_path = Path(tmpdir) / "spatial_subset.h5ad"
        st_adata.write(st_path)
        spatial_path = str(st_path)

        sc_adata = sc.read_h5ad(args.scrna)
        sc_adata = sc_adata[:500].copy()
        sc_path = Path(tmpdir) / "scrna_subset.h5ad"
        sc_adata.write(sc_path)
        scrna_path = str(sc_path)

        bulk_adata = sc.read_h5ad(args.bulk)
        bulk_adata = bulk_adata[:500].copy()
        bulk_path_tmp = Path(tmpdir) / "bulk_subset.h5ad"
        bulk_adata.write(bulk_path_tmp)
        bulk_path = str(bulk_path_tmp)

        logger.info(f"  Wrote subset files to {tmpdir}")

    # Build chain (D-05 order)
    try:
        pipeline = (
            ResultsPipeline()
            .from_checkpoint(
                checkpoint_path=args.checkpoint,
                spatial_path=spatial_path,
                scrna_path=scrna_path,
                bulk_path=bulk_path,
                output_dir=args.output_dir,
                device=args.device,
                spot_size=args.spot_size,
            )
            .export_embeddings()
            .spatial_clustering(resolution=args.leiden_resolution)
            .export_deconvolution()
            .export_mapping()
            .generate_report()
        )
    finally:
        # Clean up temp files
        if tmpdir is not None:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)
            logger.info(f"  Cleaned up temp dir: {tmpdir}")

    logger.info("=" * 60)
    logger.info(f"All outputs saved to: {args.output_dir}")
    logger.info(f"Report: {Path(args.output_dir) / 'report.md'}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
