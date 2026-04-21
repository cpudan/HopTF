#!/usr/bin/env python3
"""Load an ESM-C embedding matrix and print its basic shape and row summaries."""

# Example:
# python show_esmc_embeddings.py \
#   --embeddings /path/to/tf_atlas_morf_isoforms_esmc_600m_full.npy \
#   --vocab /path/to/tf_atlas_morf_isoform_vocab.json \
#   --mask /path/to/tf_atlas_morf_isoforms_esmc_600m_full_mask.npy

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show basic information about an ESM-C embedding matrix.")
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--vocab", type=Path, default=None)
    parser.add_argument("--mask", type=Path, default=None)
    parser.add_argument("--show", type=int, default=5)
    return parser.parse_args()


def load_vocab(path: Path | None) -> list[str] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("vocab must be a JSON list")
    return [str(value) for value in payload]


def main() -> None:
    args = parse_args()
    matrix = np.load(args.embeddings)
    vocab = load_vocab(args.vocab)
    mask = None if args.mask is None else np.load(args.mask)

    if matrix.ndim != 2:
        raise ValueError(f"expected a 2D embedding matrix, got {matrix.shape}")
    if vocab is not None and len(vocab) != matrix.shape[0]:
        raise ValueError(f"vocab length {len(vocab)} does not match matrix rows {matrix.shape[0]}")
    if mask is not None and mask.shape[0] != matrix.shape[0]:
        raise ValueError(f"mask length {mask.shape[0]} does not match matrix rows {matrix.shape[0]}")

    row_norms = np.linalg.norm(matrix, axis=1)

    print(f"path: {args.embeddings}")
    print(f"shape: {matrix.shape}")
    print(f"dtype: {matrix.dtype}")
    print(f"row_norm_min: {row_norms.min():.4f}")
    print(f"row_norm_median: {np.median(row_norms):.4f}")
    print(f"row_norm_max: {row_norms.max():.4f}")
    if mask is not None:
        print(f"embedded_rows: {int(mask.sum())}")
    print()
    print("examples:")
    for row_index in range(min(int(args.show), matrix.shape[0])):
        label = str(row_index) if vocab is None else vocab[row_index]
        print(f"- row {row_index}: {label}")
        if mask is not None:
            print(f"  embedded: {bool(mask[row_index])}")
        print(f"  first_8: {np.array2string(matrix[row_index, :8], precision=4, separator=', ')}")


if __name__ == "__main__":
    main()
