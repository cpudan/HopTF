#!/usr/bin/env python3
# Example:
# python add_hvg_pca.py \
#   --h5ad /path/to/input.h5ad \
#   --out /path/to/output_with_pca.h5ad

from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import anndata
import h5py
import numpy as np
import scanpy as sc
from scipy import sparse
from sklearn.decomposition import PCA


PCA_GROUP = "uns/pca"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add perturbation-level HVG PCA results to an H5AD."
    )
    parser.add_argument("--h5ad", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--perturbation-column", default="TF")
    parser.add_argument("--row-block-size", type=int, default=8192)
    parser.add_argument("--max-cells", type=int, default=None)
    parser.add_argument("--n-components", type=int, default=50)
    parser.add_argument("--hvg-n", type=int, default=5000)
    return parser.parse_args()


def _decode_text_array(values: np.ndarray) -> np.ndarray:
    """Convert HDF5 string-like values into plain Python strings."""
    return np.asarray(
        [
            value.decode("utf-8") if isinstance(value, (bytes, np.bytes_)) else str(value)
            for value in values
        ],
        dtype=object,
    )


def read_string_column(ad_file: h5py.File, key: str) -> np.ndarray:
    """
    Read one H5AD column as plain strings.

    AnnData may store strings as direct datasets, categorical groups, or older
    integer-code datasets with categories under `__categories`.
    """
    obj = ad_file[key]
    if isinstance(obj, h5py.Group):
        categories = _decode_text_array(obj["categories"][:])
        codes = np.asarray(obj["codes"][:], dtype=np.int64)
        out = np.empty(codes.shape[0], dtype=object)
        valid = (codes >= 0) & (codes < len(categories))
        out[~valid] = ""
        out[valid] = categories[codes[valid]]
        return out
    if np.issubdtype(obj.dtype, np.integer):
        parent = str(Path(key).parent)
        name = Path(key).name
        category_key = (
            f"{parent}/__categories/{name}" if parent not in {"", "."} else f"__categories/{name}"
        )
        if category_key in ad_file:
            categories = _decode_text_array(ad_file[category_key][:])
            codes = np.asarray(obj[:], dtype=np.int64)
            out = np.empty(codes.shape[0], dtype=object)
            valid = (codes >= 0) & (codes < len(categories))
            out[~valid] = ""
            out[valid] = categories[codes[valid]]
            return out
    return _decode_text_array(obj[:])


def iter_csr_row_blocks(ad_file: h5py.File, row_block_size: int, n_rows: int):
    """Yield CSR row blocks from `X` without loading the whole matrix."""
    if row_block_size < 1:
        raise ValueError("row_block_size must be positive")
    group = ad_file["X"]
    row_offsets_ds = group["indptr"]
    column_indices_ds = group["indices"]
    values_ds = group["data"]
    n_vars = int(group.attrs["shape"][1])
    for start in range(0, n_rows, row_block_size):
        end = min(start + row_block_size, n_rows)
        row_offsets = np.asarray(row_offsets_ds[start : end + 1], dtype=np.int64)
        nnz_start = int(row_offsets[0])
        nnz_end = int(row_offsets[-1])
        yield start, end, sparse.csr_matrix(
            (
                np.asarray(values_ds[nnz_start:nnz_end], dtype=np.float32),
                np.asarray(column_indices_ds[nnz_start:nnz_end], dtype=np.int32),
                row_offsets - nnz_start,
            ),
            shape=(end - start, n_vars),
        )


def compute_hvg_pca(
    h5ad_path: Path,
    *,
    perturbation_column: str,
    row_block_size: int,
    max_cells: int | None,
    n_components: int,
    hvg_n: int,
) -> dict[str, np.ndarray]:
    """
    Compute perturbation means on Seurat-v3 HVGs and fit PCA.

    The H5AD is expected to have:
    - `X` as log1p-normalized CSR expression
    - `layers/counts` as raw-count CSR
    - one perturbation label per cell in `obs`
    """
    if not h5ad_path.exists():
        raise FileNotFoundError(h5ad_path)

    with h5py.File(h5ad_path, "r") as ad_file:
        gene_key = "var/gene_symbol" if "var/gene_symbol" in ad_file else "var/_index"
        genes = read_string_column(ad_file, gene_key)
        perturbation_ids = read_string_column(ad_file, f"obs/{perturbation_column}")
        if "obs/n_counts" not in ad_file:
            raise KeyError("missing obs/n_counts")
        total_counts = np.asarray(ad_file["obs/n_counts"][:], dtype=np.float32)

        if max_cells is not None:
            limit = int(max_cells)
            perturbation_ids = perturbation_ids[:limit]
            total_counts = total_counts[:limit]
        else:
            limit = int(total_counts.shape[0])

        valid_mask = total_counts > 0
        if not np.any(valid_mask):
            raise ValueError("no nonzero-total cells were retained")

        counts_group = ad_file["layers"]["counts"]
        counts_csr = sparse.csr_matrix(
            (
                np.asarray(counts_group["data"][:], dtype=np.float32),
                np.asarray(counts_group["indices"][:], dtype=np.int32),
                np.asarray(counts_group["indptr"][:], dtype=np.int64),
            ),
            shape=tuple(int(v) for v in counts_group.attrs["shape"]),
        )
        counts_csr = counts_csr[:limit][valid_mask]

        hvg_ad = anndata.AnnData(X=counts_csr)
        sc.pp.highly_variable_genes(
            hvg_ad,
            flavor="seurat_v3",
            n_top_genes=min(int(hvg_n), counts_csr.shape[1]),
            subset=False,
        )
        hvg_indices = np.flatnonzero(np.asarray(hvg_ad.var["highly_variable"], dtype=bool))
        if hvg_indices.size == 0:
            raise ValueError("seurat_v3 did not select any highly variable genes")

        realized_ids = perturbation_ids[valid_mask]
        ordered_ids = sorted({str(value) for value in realized_ids})
        perturbation_index = {value: idx for idx, value in enumerate(ordered_ids)}
        mean_log1p_hvg = np.zeros((len(ordered_ids), len(hvg_indices)), dtype=np.float32)
        perturbation_n_cells = np.zeros(len(ordered_ids), dtype=np.int64)

        for start, end, block in iter_csr_row_blocks(ad_file, row_block_size=row_block_size, n_rows=limit):
            block_valid = valid_mask[start:end]
            if not np.any(block_valid):
                continue
            block_ids = perturbation_ids[start:end][block_valid]
            row_codes = np.asarray(
                [perturbation_index[str(value)] for value in block_ids], dtype=np.int64
            )
            perturbation_n_cells += np.bincount(row_codes, minlength=len(ordered_ids)).astype(np.int64)
            design = sparse.csr_matrix(
                (
                    np.ones(row_codes.shape[0], dtype=np.float32),
                    (np.arange(row_codes.shape[0], dtype=np.int64), row_codes),
                ),
                shape=(row_codes.shape[0], len(ordered_ids)),
            )
            reduced = design.T @ block[block_valid][:, hvg_indices]
            mean_log1p_hvg += reduced.toarray().astype(np.float32, copy=False)

    mean_log1p_hvg /= np.maximum(perturbation_n_cells, 1).astype(np.float32)[:, None]
    gene_mean = mean_log1p_hvg.mean(axis=0, dtype=np.float32)
    gene_sd = np.maximum(mean_log1p_hvg.std(axis=0, dtype=np.float32), np.float32(1.0e-6))
    z = ((mean_log1p_hvg - gene_mean[None, :]) / gene_sd[None, :]).astype(np.float32, copy=False)
    realized_components = min(int(n_components), int(z.shape[0]), int(z.shape[1]))
    pca = PCA(n_components=realized_components, svd_solver="randomized", random_state=0)
    states = pca.fit_transform(z).astype(np.float32, copy=False)

    return {
        "perturbation_id": np.asarray(ordered_ids, dtype=object),
        "perturbation_n_cells": perturbation_n_cells.astype(np.int64),
        "hvg_gene": np.asarray([genes[i] for i in hvg_indices.tolist()], dtype=object),
        "mean_log1p_hvg": mean_log1p_hvg,
        "pca": states,
        "gene_mean": gene_mean.astype(np.float32),
        "gene_sd": gene_sd.astype(np.float32),
        "pca_components": pca.components_.T.astype(np.float32, copy=False),
        "explained_variance_ratio": pca.explained_variance_ratio_.astype(np.float32, copy=False),
    }


def main() -> None:
    args = parse_args()
    target_path = args.h5ad if args.out is None else args.out
    if args.out is not None:
        if args.out.exists():
            raise FileExistsError(args.out)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.h5ad, args.out)

    payload = compute_hvg_pca(
        args.h5ad,
        perturbation_column=args.perturbation_column,
        row_block_size=args.row_block_size,
        max_cells=args.max_cells,
        n_components=args.n_components,
        hvg_n=args.hvg_n,
    )

    with h5py.File(target_path, "r+") as ad_file:
        if PCA_GROUP in ad_file:
            raise ValueError(f"{PCA_GROUP} already exists")
        group = ad_file.require_group("uns").create_group("pca")
        string_dtype = h5py.string_dtype(encoding="utf-8")
        for key, value in payload.items():
            if key in {"perturbation_id", "hvg_gene"}:
                group.create_dataset(key, data=np.asarray(value, dtype=object), dtype=string_dtype)
            else:
                group.create_dataset(key, data=np.asarray(value))
    print(target_path)


if __name__ == "__main__":
    main()
