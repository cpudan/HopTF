#!/usr/bin/env python3
"""Export perturbation-level PCA endpoints from the TF Atlas h5ad."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from hopfield_fitting_common import DEFAULT_H5AD, DEFAULT_METADATA, DEFAULT_OUTDIR, ensure_dir, load_metadata, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5ad", type=Path, default=DEFAULT_H5AD)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = ensure_dir(args.outdir)
    adata = ad.read_h5ad(args.h5ad, backed="r")
    if "pca" not in adata.uns:
        raise ValueError(f"{args.h5ad} does not contain uns['pca']")
    pca = adata.uns["pca"]
    required = ["pca", "perturbation_id", "perturbation_n_cells", "response_score", "control_mean", "control_sd"]
    missing = [key for key in required if key not in pca]
    if missing:
        raise ValueError(f"uns['pca'] is missing keys: {missing}")

    perturbation_id = np.asarray([str(value) for value in pca["perturbation_id"]], dtype=object)
    latent = np.asarray(pca["pca"], dtype=np.float32)
    n_cells = np.asarray(pca["perturbation_n_cells"], dtype=np.int64)
    response_score = np.asarray(pca["response_score"], dtype=np.float32)
    control_mean = np.asarray(pca["control_mean"], dtype=np.float32)
    control_sd = np.asarray(pca["control_sd"], dtype=np.float32)

    if latent.ndim != 2:
        raise ValueError(f"uns['pca']['pca'] must be 2D, got {latent.shape}")
    if latent.shape[0] != perturbation_id.shape[0]:
        raise ValueError("PCA rows do not match perturbation_id rows")

    metadata = load_metadata(args.metadata, require_columns=["perturbation_id"])
    latent_df = pd.DataFrame(
        {
            "perturbation_id": perturbation_id,
            "latent_index": np.arange(perturbation_id.shape[0], dtype=np.int64),
            "latent_n_cells": n_cells,
            "latent_response_score": response_score,
        }
    )
    merged = latent_df.merge(metadata, on="perturbation_id", how="left", validate="one_to_one")

    npz_path = outdir / "perturbation_latents.npz"
    csv_path = outdir / "perturbation_latents_metadata.csv"
    report_path = outdir / "perturbation_latents_report.json"
    np.savez_compressed(
        npz_path,
        perturbation_id=perturbation_id.astype(str),
        latent_pca=latent,
        perturbation_n_cells=n_cells,
        response_score=response_score,
        control_mean=control_mean,
        control_sd=control_sd,
        explained_variance_ratio=np.asarray(pca.get("explained_variance_ratio", []), dtype=np.float32),
    )
    merged.to_csv(csv_path, index=False)
    write_json(
        report_path,
        {
            "h5ad": args.h5ad,
            "metadata": args.metadata,
            "npz_path": npz_path,
            "metadata_path": csv_path,
            "latent_shape": list(latent.shape),
            "n_perturbations": int(latent.shape[0]),
            "n_metadata_matched": int(merged["isoform_embedding_id"].notna().sum())
            if "isoform_embedding_id" in merged.columns
            else None,
            "control_mean_shape": list(control_mean.shape),
            "control_sd_shape": list(control_sd.shape),
        },
    )
    print(f"wrote {npz_path} {csv_path} {report_path}")


if __name__ == "__main__":
    main()
