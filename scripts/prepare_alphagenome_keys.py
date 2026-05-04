#!/usr/bin/env python3
"""Prepare normalized AlphaGenome/Hopfield key matrices."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from hopfield_fitting_common import DEFAULT_OUTDIR, ensure_dir, normalize_rows, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("synthetic", "gene_pooled_npz", "chip_tf_k_matrix"), default="synthetic")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--synthetic-key-count", type=int, default=256)
    parser.add_argument("--synthetic-key-dim", type=int, default=64)
    parser.add_argument("--chrom", default=None, help="Optional chromosome filter for gene_pooled_npz, e.g. chr1.")
    parser.add_argument("--max-keys", type=int, default=None)
    return parser.parse_args()


def _decode_array(values: np.ndarray) -> list[str]:
    out = []
    for value in values:
        if isinstance(value, bytes):
            out.append(value.decode("utf-8"))
        else:
            out.append(str(value))
    return out


def synthetic_keys(args: argparse.Namespace) -> tuple[np.ndarray, pd.DataFrame, dict[str, object]]:
    rng = np.random.default_rng(args.seed)
    keys = rng.normal(size=(args.synthetic_key_count, args.synthetic_key_dim)).astype(np.float32)
    keys = normalize_rows(keys)
    metadata = pd.DataFrame(
        {
            "key_id": [f"synthetic_key_{idx:05d}" for idx in range(keys.shape[0])],
            "source": "synthetic",
            "chrom": "synthetic",
            "start": np.arange(keys.shape[0], dtype=np.int64),
            "end": np.arange(keys.shape[0], dtype=np.int64) + 1,
        }
    )
    report = {"source": "synthetic", "shape": list(keys.shape), "seed": int(args.seed)}
    return keys, metadata, report


def gene_pooled_npz_keys(args: argparse.Namespace) -> tuple[np.ndarray, pd.DataFrame, dict[str, object]]:
    if args.input is None:
        raise ValueError("--input is required for gene_pooled_npz")
    # The AI4Genome gene-pooled archive stores string metadata as object arrays.
    # Restrict pickle-enabled loading to this explicit user-provided NPZ path.
    with np.load(args.input, allow_pickle=True) as data:
        if "pooled" not in data.files:
            raise ValueError(f"{args.input} does not contain a 'pooled' array; found {data.files}")
        keys = np.asarray(data["pooled"], dtype=np.float32)
        chroms = _decode_array(data["chroms"]) if "chroms" in data.files else [""] * keys.shape[0]
        keep = np.ones(keys.shape[0], dtype=bool)
        if args.chrom:
            keep &= np.asarray(chroms, dtype=object) == args.chrom
        if args.max_keys is not None:
            selected = np.flatnonzero(keep)[: args.max_keys]
        else:
            selected = np.flatnonzero(keep)
        keys = normalize_rows(keys[selected])
        metadata = pd.DataFrame({"source_index": selected, "chrom": np.asarray(chroms, dtype=object)[selected]})
        optional_fields = ["gene_ids", "gene_symbols", "starts", "ends", "strands", "seq_lengths"]
        for field in optional_fields:
            if field in data.files:
                values = data[field][selected]
                if values.dtype.kind in {"S", "U", "O"}:
                    metadata[field] = _decode_array(values)
                else:
                    metadata[field] = values
        if "gene_ids" in metadata.columns:
            metadata.insert(0, "key_id", metadata["gene_ids"].astype(str))
        else:
            metadata.insert(0, "key_id", [f"gene_pooled_{idx:08d}" for idx in selected])
        report = {
            "source": "gene_pooled_npz",
            "input": str(args.input),
            "allow_pickle": True,
            "allow_pickle_reason": "gene-pooled metadata arrays are stored with dtype object",
            "input_arrays": {key: {"shape": list(data[key].shape), "dtype": str(data[key].dtype)} for key in data.files},
            "chrom_filter": args.chrom,
            "max_keys": args.max_keys,
            "shape": list(keys.shape),
        }
    return keys, metadata, report


def chip_tf_k_matrix_keys(args: argparse.Namespace) -> tuple[np.ndarray, pd.DataFrame, dict[str, object]]:
    if args.input is None:
        raise ValueError("--input is required for chip_tf_k_matrix")
    arr = np.asarray(np.load(args.input, mmap_mode="r"), dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"CHIP_TF K matrix must be 2D, got {arr.shape}")
    if args.max_keys is not None:
        arr = arr[: args.max_keys]
    keys = normalize_rows(arr)
    metadata = pd.DataFrame(
        {
            "key_id": [f"chip_tf_locus_{idx:08d}" for idx in range(keys.shape[0])],
            "source": "chip_tf_k_matrix",
            "source_index": np.arange(keys.shape[0], dtype=np.int64),
        }
    )
    report = {"source": "chip_tf_k_matrix", "input": str(args.input), "shape": list(keys.shape)}
    return keys, metadata, report


def main() -> None:
    args = parse_args()
    outdir = ensure_dir(args.outdir)
    if args.source == "synthetic":
        keys, metadata, report = synthetic_keys(args)
    elif args.source == "gene_pooled_npz":
        keys, metadata, report = gene_pooled_npz_keys(args)
    else:
        keys, metadata, report = chip_tf_k_matrix_keys(args)

    keys_path = outdir / "alphagenome_keys.npy"
    metadata_path = outdir / "alphagenome_keys_metadata.csv"
    report_path = outdir / "alphagenome_keys_report.json"
    np.save(keys_path, keys.astype(np.float32))
    metadata.to_csv(metadata_path, index=False)
    report.update({"keys_path": str(keys_path), "metadata_path": str(metadata_path)})
    write_json(report_path, report)
    print(f"wrote {keys_path} {metadata_path} {report_path}")


if __name__ == "__main__":
    main()
