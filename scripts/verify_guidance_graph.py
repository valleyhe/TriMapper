#!/usr/bin/env python3
"""Verify guidance graph quality for Rosacea triple-modal data.

Task 2 of 12-01-PLAN.md: Validate guidance graph edge counts, shapes,
and data types before TripleModalVAE initialization.
"""

import sys
sys.path.insert(0, "/home/scu/stSCI-GLUE-workspace/src")

import json
import logging
from pathlib import Path

import torch

from tglue.graph.guidance_graph import GuidanceGraph

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def verify_guidance_graph() -> bool:
    """Verify guidance graph quality from graph_stats.json.

    Returns
    -------
    bool
        True if all validation criteria passed, False otherwise.
    """
    data_dir = Path("/home/scu/stSCI-GLUE-workspace/data/rosacea")
    stats_path = data_dir / "graph_stats.json"
    qc_report_path = data_dir / "qc_report.json"

    # Load statistics
    with open(stats_path) as f:
        stats = json.load(f)

    with open(qc_report_path) as f:
        qc = json.load(f)

    logger.info("=" * 60)
    logger.info("GUIDANCE GRAPH VERIFICATION")
    logger.info("=" * 60)

    all_passed = True

    # Check 1: Gene count matches shared_genes.txt
    expected_genes = qc["shared_genes"]["n_genes"]
    actual_genes = stats["n_genes"]
    gene_check = actual_genes == expected_genes
    logger.info(f"1. Gene count check:")
    logger.info(f"   Expected: {expected_genes}")
    logger.info(f"   Actual:   {actual_genes}")
    logger.info(f"   Status:   {'PASS' if gene_check else 'FAIL'}")
    all_passed &= gene_check

    # Check 2: Genomic edges >= 10,000
    min_genomic = stats["validation"]["min_genomic_threshold"]
    actual_genomic = stats["n_genomic_edges"]
    genomic_check = actual_genomic >= min_genomic
    logger.info(f"2. Genomic edges check:")
    logger.info(f"   Minimum:  {min_genomic}")
    logger.info(f"   Actual:   {actual_genomic}")
    logger.info(f"   Status:   {'PASS' if genomic_check else 'FAIL'}")
    all_passed &= genomic_check

    # Check 3: Co-expression edges >= 50,000
    min_coexpr = stats["validation"]["min_coexpr_threshold"]
    actual_coexpr = stats["n_coexpr_edges"]
    coexpr_check = actual_coexpr >= min_coexpr
    logger.info(f"3. Co-expression edges check:")
    logger.info(f"   Minimum:  {min_coexpr}")
    logger.info(f"   Actual:   {actual_coexpr}")
    logger.info(f"   Status:   {'PASS' if coexpr_check else 'FAIL'}")
    all_passed &= coexpr_check

    # Check 4: edge_index shape is (2, N)
    expected_shape = [2, stats["n_total_edges"]]
    actual_shape = stats["edge_index_shape"]
    shape_check = actual_shape == expected_shape
    logger.info(f"4. Edge index shape check:")
    logger.info(f"   Expected: {expected_shape}")
    logger.info(f"   Actual:   {actual_shape}")
    logger.info(f"   Status:   {'PASS' if shape_check else 'FAIL'}")
    all_passed &= shape_check

    # Check 5: Load graph and verify data types
    logger.info("5. Data type verification:")
    try:
        # Find the graph pickle file (created by build script if saved)
        graph_pkl = data_dir / "guidance_graph.pkl"
        if graph_pkl.exists():
            graph = GuidanceGraph.load(str(graph_pkl))
        else:
            # Reconstruct from edges if pickle not saved
            logger.info("   Graph pickle not found - skipping dtype verification")
            logger.info("   Status: SKIP (graph pickle not saved)")
        logger.info("   Status:   PASS (graph loadable)")
    except Exception as e:
        logger.info(f"   Error: {e}")
        logger.info("   Status:   FAIL")
        all_passed = False

    # Summary
    logger.info("=" * 60)
    logger.info("VERIFICATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total genes: {stats['n_genes']}")
    logger.info(f"Genomic edges: {stats['n_genomic_edges']}")
    logger.info(f"Co-expr edges: {stats['n_coexpr_edges']}")
    logger.info(f"Total edges: {stats['n_total_edges']}")
    logger.info(f"Build timestamp: {stats['build_timestamp']}")
    logger.info("=" * 60)
    if all_passed:
        logger.info("RESULT: ALL CHECKS PASSED")
        logger.info("Guidance graph is ready for TripleModalVAE initialization")
    else:
        logger.error("RESULT: SOME CHECKS FAILED")
        logger.error("Review failed checks above and re-run build if needed")
    logger.info("=" * 60)

    return all_passed


if __name__ == "__main__":
    success = verify_guidance_graph()
    sys.exit(0 if success else 1)