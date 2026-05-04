#!/usr/bin/env python3
"""Inspect data contracts for the Hopfield/OT-CFM overnight pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from hopfield_fitting_common import (
    DEFAULT_ESM,
    DEFAULT_METADATA,
    DEFAULT_OUTDIR,
    DEFAULT_VOCAB,
    candidate_isoform_groups,
    ensure_dir,
    load_metadata,
    load_vocab,
    spearman,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--embedding-matrix", type=Path, default=DEFAULT_ESM)
    parser.add_argument("--vocab", type=Path, default=DEFAULT_VOCAB)
    parser.add_argument("--alphagenome-keys", type=Path, default=None)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--min-cells", type=int, default=5)
    parser.add_argument("--top-groups", type=int, default=50)
    return parser.parse_args()


def inspect_key_artifact(path: Path) -> dict[str, object]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    payload: dict[str, object] = {"path": str(path), "exists": True, "size_bytes": int(path.stat().st_size)}
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            payload["format"] = "npz"
            payload["arrays"] = {
                key: {"shape": list(data[key].shape), "dtype": str(data[key].dtype)}
                for key in data.files
            }
    else:
        arr = np.load(path, mmap_mode="r")
        payload["format"] = "npy"
        payload["shape"] = list(arr.shape)
        payload["dtype"] = str(arr.dtype)
    return payload


def main() -> None:
    args = parse_args()
    outdir = ensure_dir(args.outdir)
    metadata = load_metadata(
        args.metadata,
        require_columns=[
            "perturbation_id",
            "isoform_embedding_id",
            "gene_symbol",
            "isoform_id",
            "protein_aa_length",
            "n_cells",
            "response_score",
            "label_status",
        ],
    )
    vocab = load_vocab(args.vocab)
    vocab_index = {value: index for index, value in enumerate(vocab)}
    embeddings = np.load(args.embedding_matrix, mmap_mode="r")
    if embeddings.ndim != 2:
        raise ValueError(f"embedding matrix must be 2D, got {embeddings.shape}")
    if embeddings.shape[0] != len(vocab):
        raise ValueError(f"embedding rows ({embeddings.shape[0]}) do not match vocab ({len(vocab)})")

    embedding_ids = metadata["isoform_embedding_id"].astype(str)
    missing_vocab = sorted(set(embedding_ids) - set(vocab_index))
    duplicated_ids = sorted(embedding_ids[embedding_ids.duplicated()].unique())
    labels = metadata["label_status"].astype(str)
    groups = candidate_isoform_groups(metadata, min_cells=args.min_cells)

    numeric = metadata.assign(
        protein_aa_length=pd.to_numeric(metadata["protein_aa_length"], errors="coerce"),
        n_cells=pd.to_numeric(metadata["n_cells"], errors="coerce"),
        response_score=pd.to_numeric(metadata["response_score"], errors="coerce"),
    )
    report = {
        "paths": {
            "metadata": args.metadata,
            "embedding_matrix": args.embedding_matrix,
            "vocab": args.vocab,
            "alphagenome_keys": args.alphagenome_keys,
            "outdir": outdir,
        },
        "embedding_matrix": {"shape": list(embeddings.shape), "dtype": str(embeddings.dtype)},
        "vocab": {"length": len(vocab), "first_entries": vocab[:5]},
        "metadata": {
            "shape": list(metadata.shape),
            "columns": list(metadata.columns),
            "label_counts": labels.value_counts(dropna=False).to_dict(),
            "non_control_rows": int((metadata.get("is_control", False).astype(str) != "True").sum())
            if "is_control" in metadata.columns
            else None,
            "missing_vocab_count": len(missing_vocab),
            "missing_vocab_examples": missing_vocab[:20],
            "duplicated_isoform_embedding_ids": duplicated_ids[:20],
        },
        "correlations": {
            "protein_aa_length_vs_n_cells": spearman(numeric["protein_aa_length"], numeric["n_cells"]),
            "response_score_vs_n_cells": spearman(numeric["response_score"], numeric["n_cells"]),
            "protein_aa_length_vs_response_score": spearman(numeric["protein_aa_length"], numeric["response_score"]),
        },
        "candidate_groups": {
            "min_cells": int(args.min_cells),
            "n_groups": int(groups.shape[0]),
            "top": groups.head(args.top_groups).to_dict(orient="records"),
        },
    }
    if args.alphagenome_keys is not None:
        report["alphagenome_keys"] = inspect_key_artifact(args.alphagenome_keys)

    groups.to_csv(outdir / "candidate_isoform_groups.csv", index=False)
    write_json(outdir / "input_contract.json", report)
    print(json.dumps({"ok": True, "out": str(outdir / "input_contract.json")}, indent=2))


if __name__ == "__main__":
    main()
