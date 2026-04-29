#!/usr/bin/env python3
"""Run full ResultsPipeline on real Rosacea data (345K spots, 76K cells)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tglue.results import ResultsPipeline

def main():
    checkpoint = "results/rosacea/fair_ablation_01/best_model.pt"
    spatial = "data/rosacea/spatial_100k.h5ad"
    scrna = "data/rosacea/sc_reference.h5ad"
    bulk = "data/rosacea/array_test.h5ad"
    output_dir = "results/rosacea/full_outputs_real"
    
    print("=" * 60)
    print("ResultsPipeline -- Full Rosacea Data")
    print("=" * 60)
    print(f"ST: 100K spots")
    print(f"scRNA: 76K cells")
    print(f"Bulk: 58 samples")
    print(f"Output: {output_dir}")
    print("=" * 60)
    
    pipeline = (
        ResultsPipeline()
        .from_checkpoint(
            checkpoint_path=checkpoint,
            spatial_path=spatial,
            scrna_path=scrna,
            bulk_path=bulk,
            output_dir=output_dir,
            device="cuda",
            n_genes=17825,
            latent_dim=64,      # checkpoint latent dim
            enc_sc_hidden=128,  # checkpoint hidden dim (统一)
        )
        .export_embeddings(batch_size=256)
        .spatial_clustering(resolution=1.0, use_spatial_neighbors=False)  # latent-based, not spatial coords
        .export_deconvolution(
            epsilon=0.5,
            k_neighbors=20,
            chunk_size=500,
        )
        .export_mapping(topk=10)
        .generate_report()
    )
    
    print("=" * 60)
    print("完成！")
    print("=" * 60)

if __name__ == "__main__":
    main()
