#!/usr/bin/env python3
# Example:
# python make_borzoi_tokens.py \
#   --intervals-bed /path/to/windows.bed \
#   --out /path/to/borzoi_tokens.h5 \
#   --assay RNA \
#   --sample-contains K562

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate full tokenized Borzoi representations for DNA windows."
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rows-out", type=Path, default=None)
    parser.add_argument("--tasks-out", type=Path, default=None)
    parser.add_argument("--meta-out", type=Path, default=None)
    parser.add_argument("--intervals-bed", type=Path, default=None)
    parser.add_argument("--seq-tsv", type=Path, default=None)
    parser.add_argument("--sequence-column", default="sequence")
    parser.add_argument("--genome", default="hg38")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--assay", default=None)
    parser.add_argument("--sample-contains", default=None)
    parser.add_argument("--dtype", choices=("float32", "float16"), default="float32")
    return parser.parse_args()


def load_input_rows(args: argparse.Namespace) -> tuple[pd.DataFrame, list[str]]:
    """
    Load the Borzoi input rows and stable row IDs.

    Use `--seq-tsv` with an `id` column when possible. That produces meaningful
    row labels in the sidecar file.
    """
    if (args.intervals_bed is None) == (args.seq_tsv is None):
        raise ValueError("provide exactly one of --intervals-bed or --seq-tsv")

    if args.intervals_bed is not None:
        rows = pd.read_csv(
            args.intervals_bed,
            sep="\t",
            header=None,
            comment="#",
            usecols=[0, 1, 2],
            names=["chrom", "start", "end"],
        )
        if args.max_rows is not None:
            rows = rows.iloc[: int(args.max_rows)].copy()
        row_ids = [
            f"{chrom}:{start}-{end}"
            for chrom, start, end in rows[["chrom", "start", "end"]].itertuples(index=False, name=None)
        ]
        return rows, row_ids

    rows = pd.read_csv(args.seq_tsv, sep="\t")
    if args.sequence_column not in rows.columns:
        raise ValueError(f"missing sequence column: {args.sequence_column}")
    if args.max_rows is not None:
        rows = rows.iloc[: int(args.max_rows)].copy()
    row_ids = [str(i) for i in rows.index.tolist()]
    if "id" in rows.columns:
        row_ids = [str(value) for value in rows["id"].tolist()]
    return rows, row_ids


def load_model(args: argparse.Namespace):
    import grelu.resources

    model = grelu.resources.load_model(repo_id="Genentech/borzoi-model", filename="human_rep0.ckpt")
    model.eval()
    return model


def select_tasks(model, assay: str | None, sample_contains: str | None) -> tuple[np.ndarray, pd.DataFrame]:
    """Select Borzoi output tracks and return both indices and task metadata."""
    tasks = pd.DataFrame(model.data_params["tasks"])
    mask = np.ones(len(tasks), dtype=bool)
    if assay is not None:
        mask &= tasks["assay"].astype(str).str.lower().eq(str(assay).lower()).to_numpy()
    if sample_contains is not None:
        mask &= tasks["sample"].astype(str).str.contains(str(sample_contains), case=False, regex=False).to_numpy()
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        raise ValueError("task filter removed every Borzoi task")
    return indices.astype(np.int64), tasks.iloc[indices].reset_index(drop=True).copy()


def to_sequences(rows: pd.DataFrame, args: argparse.Namespace) -> list[str]:
    """Convert either intervals or an explicit sequence table into DNA strings."""
    if args.seq_tsv is not None:
        return [str(value) for value in rows[args.sequence_column].tolist()]

    import grelu.sequence.format

    return list(
        grelu.sequence.format.convert_input_type(
            rows[["chrom", "start", "end"]].assign(strand="+"),
            output_type="strings",
            genome=args.genome,
        )
    )


def resolve_device(device: str):
    if device == "auto":
        import torch

        return (0 if torch.cuda.is_available() else "cpu"), ("cuda" if torch.cuda.is_available() else "cpu")
    if str(device).isdigit():
        return int(device), f"cuda:{int(device)}"
    return device, str(device)


def main() -> None:
    args = parse_args()
    rows, row_ids = load_input_rows(args)
    model = load_model(args)
    task_indices, task_table = select_tasks(model, args.assay, args.sample_contains)
    sequences = to_sequences(rows, args)
    device, device_name = resolve_device(args.device)
    output_dtype = np.float16 if args.dtype == "float16" else np.float32

    rows_out = args.rows_out or args.out.with_name(args.out.stem + "_rows.tsv")
    tasks_out = args.tasks_out or args.out.with_name(args.out.stem + "_tasks.tsv")
    meta_out = args.meta_out or args.out.with_suffix(".meta.json")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if rows_out.parent != args.out.parent:
        rows_out.parent.mkdir(parents=True, exist_ok=True)
    if tasks_out.parent != args.out.parent:
        tasks_out.parent.mkdir(parents=True, exist_ok=True)
    if meta_out.parent != args.out.parent:
        meta_out.parent.mkdir(parents=True, exist_ok=True)

    tokens_ds = None
    n_bins = None

    with h5py.File(args.out, "w") as h5:
        for start in range(0, len(sequences), int(args.batch_size)):
            batch = sequences[start : start + int(args.batch_size)]
            preds = np.asarray(model.predict_on_seqs(batch, device=device))
            if preds.ndim != 3:
                raise ValueError(f"expected Borzoi predictions with shape [B, T, L], got {preds.shape}")
            selected = preds[:, task_indices, :]
            tokens = np.transpose(selected, (0, 2, 1)).astype(output_dtype, copy=False)
            if tokens_ds is None:
                n_bins = int(tokens.shape[1])
                tokens_ds = h5.create_dataset(
                    "tokens",
                    shape=(len(sequences), int(tokens.shape[1]), int(tokens.shape[2])),
                    dtype=output_dtype,
                    chunks=(1, int(tokens.shape[1]), int(tokens.shape[2])),
                )
            tokens_ds[start : start + len(batch)] = tokens

        if tokens_ds is None or n_bins is None:
            raise ValueError("no Borzoi windows were generated")

    pd.DataFrame({"row_id": row_ids}).to_csv(rows_out, sep="\t", index=False)
    task_table.to_csv(tasks_out, sep="\t", index=False)
    meta_out.write_text(
        json.dumps(
            {
                "out": str(args.out),
                "rows_out": str(rows_out),
                "tasks_out": str(tasks_out),
                "n_rows": int(len(sequences)),
                "n_bins": int(n_bins),
                "n_tasks": int(task_indices.size),
                "dtype": args.dtype,
                "repo_id": "Genentech/borzoi-model",
                "filename": "human_rep0.ckpt",
                "genome": None if args.seq_tsv is not None else args.genome,
                "device": device_name,
                "assay": args.assay,
                "sample_contains": args.sample_contains,
                "representation": "per_window_tokenized_borzoi",
                "token_axis": "sequence_bins",
                "feature_axis": "selected_tasks",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(args.out)


if __name__ == "__main__":
    main()
