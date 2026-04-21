#!/usr/bin/env python3
"""Load a tokenized Borzoi HDF5 file and print its basic shape and examples."""

# Example:
# python show_borzoi_tokens.py \
#   --tokens /path/to/borzoi_tokens.h5 \
#   --rows /path/to/borzoi_tokens_rows.tsv

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show basic information about a tokenized Borzoi output.")
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--rows", type=Path, default=None)
    parser.add_argument("--show", type=int, default=3)
    return parser.parse_args()


def load_row_ids(path: Path | None) -> list[str] | None:
    if path is None:
        return None
    rows = pd.read_csv(path, sep="\t")
    if "row_id" not in rows.columns:
        raise ValueError("rows file must contain a row_id column")
    return [str(value) for value in rows["row_id"].tolist()]


def main() -> None:
    args = parse_args()
    row_ids = load_row_ids(args.rows)

    with h5py.File(args.tokens, "r") as h5:
        if "tokens" not in h5:
            raise ValueError("missing tokens dataset")
        tokens = h5["tokens"]
        if tokens.ndim != 3:
            raise ValueError(f"expected tokens with shape [rows, bins, tasks], got {tokens.shape}")
        if row_ids is not None and len(row_ids) != tokens.shape[0]:
            raise ValueError(f"row ID count {len(row_ids)} does not match token rows {tokens.shape[0]}")

        print(f"path: {args.tokens}")
        print(f"shape: {tokens.shape}")
        print(f"dtype: {tokens.dtype}")
        print()
        print("examples:")
        for row_index in range(min(int(args.show), tokens.shape[0])):
            label = str(row_index) if row_ids is None else row_ids[row_index]
            row = np.asarray(tokens[row_index], dtype=np.float32)
            print(f"- row {row_index}: {label}")
            print(f"  bins: {row.shape[0]}")
            print(f"  tasks: {row.shape[1]}")
            print(f"  first_bin_first_8: {np.array2string(row[0, :8], precision=4, separator=', ')}")


if __name__ == "__main__":
    main()
