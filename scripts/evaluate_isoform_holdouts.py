#!/usr/bin/env python3
"""Evaluate leave-one-isoform candidate groups and simple baselines."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_distances

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from hopfield_fitting_common import DEFAULT_ESM, DEFAULT_METADATA, DEFAULT_OUTDIR, DEFAULT_VOCAB, candidate_isoform_groups, ensure_dir, load_metadata, load_vocab, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--embedding-matrix", type=Path, default=DEFAULT_ESM)
    parser.add_argument("--vocab", type=Path, default=DEFAULT_VOCAB)
    parser.add_argument("--latents", type=Path, default=DEFAULT_OUTDIR / "perturbation_latents.npz")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--gene", action="append", default=None)
    parser.add_argument("--min-cells", type=int, default=5)
    parser.add_argument("--max-groups", type=int, default=12)
    return parser.parse_args()


def load_joined(args: argparse.Namespace) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    metadata = load_metadata(
        args.metadata,
        require_columns=["perturbation_id", "isoform_embedding_id", "gene_symbol", "isoform_id", "n_cells", "response_score", "label_status"],
    )
    vocab = load_vocab(args.vocab)
    vocab_index = {value: index for index, value in enumerate(vocab)}
    esm_matrix = np.load(args.embedding_matrix, mmap_mode="r")
    latent = np.load(args.latents, allow_pickle=False)
    latent_ids = {str(value): index for index, value in enumerate(latent["perturbation_id"])}
    rows = []
    esm_indices = []
    latent_indices = []
    for _, row in metadata.iterrows():
        if pd.to_numeric(row["n_cells"], errors="coerce") < args.min_cells:
            continue
        iso_id = str(row["isoform_embedding_id"])
        pert_id = str(row["perturbation_id"])
        if iso_id in vocab_index and pert_id in latent_ids:
            rows.append(row)
            esm_indices.append(vocab_index[iso_id])
            latent_indices.append(latent_ids[pert_id])
    if not rows:
        raise ValueError("no rows matched metadata, ESM-C vocab, and latent endpoints")
    row_df = pd.DataFrame(rows).reset_index(drop=True)
    esm = np.asarray(esm_matrix[np.asarray(esm_indices, dtype=np.int64)], dtype=np.float32)
    latent_pca = np.asarray(latent["latent_pca"][np.asarray(latent_indices, dtype=np.int64)], dtype=np.float32)
    return row_df, esm, latent_pca


def main() -> None:
    args = parse_args()
    outdir = ensure_dir(args.outdir)
    rows, esm, latent = load_joined(args)
    groups = candidate_isoform_groups(rows, min_cells=args.min_cells)
    if args.gene:
        genes = [str(gene) for gene in args.gene]
    else:
        mixed = groups.loc[(groups["n_responders"] > 0) & (groups["n_nonresponders"] > 0), "gene_symbol"].head(args.max_groups)
        if mixed.empty:
            mixed = groups["gene_symbol"].head(args.max_groups)
        genes = [str(gene) for gene in mixed]

    records = []
    for gene in genes:
        idx = np.flatnonzero(rows["gene_symbol"].astype(str).to_numpy() == gene)
        if idx.size < 2:
            continue
        esm_dist = cosine_distances(esm[idx])
        latent_dist = cosine_distances(latent[idx])
        for local_pos, global_idx in enumerate(idx):
            sibling_positions = np.asarray([pos for pos in range(idx.size) if pos != local_pos], dtype=np.int64)
            if sibling_positions.size == 0:
                continue
            nearest_esm_pos = sibling_positions[np.argmin(esm_dist[local_pos, sibling_positions])]
            label = str(rows.iloc[global_idx]["label_status"])
            sibling_labels = rows.iloc[idx[sibling_positions]]["label_status"].astype(str).to_numpy()
            same_mask = sibling_labels == label
            diff_mask = ~same_mask
            same_latent_distance = (
                float(np.min(latent_dist[local_pos, sibling_positions[same_mask]])) if same_mask.any() else np.nan
            )
            different_latent_distance = (
                float(np.min(latent_dist[local_pos, sibling_positions[diff_mask]])) if diff_mask.any() else np.nan
            )
            nearest_global = idx[nearest_esm_pos]
            response = float(pd.to_numeric(rows.iloc[global_idx]["response_score"], errors="coerce"))
            nearest_response = float(pd.to_numeric(rows.iloc[nearest_global]["response_score"], errors="coerce"))
            records.append(
                {
                    "gene_symbol": gene,
                    "heldout_perturbation_id": rows.iloc[global_idx]["perturbation_id"],
                    "heldout_isoform_id": rows.iloc[global_idx]["isoform_id"],
                    "heldout_label_status": label,
                    "heldout_n_cells": int(rows.iloc[global_idx]["n_cells"]),
                    "heldout_protein_aa_length": int(rows.iloc[global_idx]["protein_aa_length"]),
                    "heldout_response_score": response,
                    "nearest_esm_sibling": rows.iloc[nearest_global]["isoform_id"],
                    "nearest_esm_sibling_label_status": rows.iloc[nearest_global]["label_status"],
                    "nearest_esm_sibling_response_score": nearest_response,
                    "nearest_esm_cosine_distance": float(esm_dist[local_pos, nearest_esm_pos]),
                    "latent_distance_to_nearest_esm_sibling": float(latent_dist[local_pos, nearest_esm_pos]),
                    "same_label_min_latent_cosine_distance": same_latent_distance,
                    "different_label_min_latent_cosine_distance": different_latent_distance,
                    "nearest_sibling_abs_response_error": abs(response - nearest_response),
                    "same_label_closer_than_different_label": bool(same_latent_distance < different_latent_distance)
                    if np.isfinite(same_latent_distance) and np.isfinite(different_latent_distance)
                    else None,
                }
            )

    result = pd.DataFrame(records)
    csv_path = outdir / "isoform_holdout_baselines.csv"
    result.to_csv(csv_path, index=False)
    summary = {
        "genes": genes,
        "n_records": int(result.shape[0]),
        "metrics_csv": csv_path,
        "mean_nearest_sibling_abs_response_error": float(result["nearest_sibling_abs_response_error"].mean())
        if not result.empty
        else None,
        "same_label_closer_fraction": float(result["same_label_closer_than_different_label"].dropna().mean())
        if "same_label_closer_than_different_label" in result and result["same_label_closer_than_different_label"].notna().any()
        else None,
    }
    write_json(outdir / "isoform_holdout_baselines_summary.json", summary)
    print(f"wrote {csv_path} {outdir / 'isoform_holdout_baselines_summary.json'}")


if __name__ == "__main__":
    main()
