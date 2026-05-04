#!/usr/bin/env python3
"""Shared utilities for the Hopfield/OT-CFM overnight smoke pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_METADATA = Path("data/processed/linear_probe/tfatlas_subsample/PERTURBATION_METADATA_hard_local_subsample.csv")
DEFAULT_ESM = Path(
    "data/hf_min/processed/protein_embeddings/"
    "tf_atlas_morf_isoforms_esmc_600m/"
    "tf_atlas_morf_isoforms_esmc_600m_mean_non_special.npy"
)
DEFAULT_VOCAB = Path(
    "data/hf_min/processed/protein_embeddings/"
    "tf_atlas_morf_isoforms_esmc_600m/metadata/"
    "tf_atlas_morf_isoform_vocab.json"
)
DEFAULT_H5AD = Path("data/raw_sources/joung_tfatlas/published_h5ad/GSE217460_210322_TFAtlas_subsample_raw_csr.h5ad")
DEFAULT_OUTDIR = Path("tmp/hopfield_fitting")


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    if isinstance(value, pd.Series):
        return value.to_dict()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_vocab(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list) and payload:
        return [str(item) for item in payload]
    if isinstance(payload, dict) and payload:
        ordered = sorted(payload.items(), key=lambda item: int(item[1]))
        return [str(item[0]) for item in ordered]
    raise ValueError(f"vocab must be a non-empty JSON list or id->index dict: {path}")


def load_metadata(path: Path, *, require_columns: list[str] | None = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = sorted(set(require_columns or []) - set(df.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    return df


def normalize_rows(matrix: np.ndarray, eps: float = 1.0e-8) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, eps)


def spearman(x: pd.Series | np.ndarray, y: pd.Series | np.ndarray) -> dict[str, float | int | None]:
    try:
        from scipy.stats import spearmanr
    except Exception:
        return {"rho": None, "pvalue": None, "n": int(len(x))}
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    if int(mask.sum()) < 3:
        return {"rho": None, "pvalue": None, "n": int(mask.sum())}
    stat = spearmanr(x_arr[mask], y_arr[mask])
    return {"rho": float(stat.statistic), "pvalue": float(stat.pvalue), "n": int(mask.sum())}


def load_esmc_for_metadata(
    metadata: pd.DataFrame,
    embedding_matrix: Path,
    vocab: Path,
    *,
    require_labeled: bool = False,
) -> tuple[np.ndarray, pd.DataFrame]:
    required = ["isoform_embedding_id", "perturbation_id", "gene_symbol"]
    if require_labeled:
        required.extend(["label_status", "responder_label"])
    missing = sorted(set(required) - set(metadata.columns))
    if missing:
        raise ValueError(f"metadata is missing required columns: {missing}")

    vocab_values = load_vocab(vocab)
    vocab_index = {value: index for index, value in enumerate(vocab_values)}
    matrix = np.load(embedding_matrix, mmap_mode="r")
    if matrix.ndim != 2:
        raise ValueError(f"ESM-C embedding matrix must be 2D, got shape {matrix.shape}")

    keep_mask = metadata["isoform_embedding_id"].astype(str).isin(vocab_index)
    if require_labeled:
        keep_mask &= metadata["label_status"].astype(str).isin({"responder", "nonresponder"})
    rows = metadata.loc[keep_mask].copy().reset_index(drop=True)
    indices = np.asarray([vocab_index[str(value)] for value in rows["isoform_embedding_id"]], dtype=np.int64)
    return np.asarray(matrix[indices], dtype=np.float32), rows


def load_latent_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def merge_metadata_with_latents(metadata: pd.DataFrame, latent_npz: Path) -> tuple[np.ndarray, pd.DataFrame]:
    latent = load_latent_npz(latent_npz)
    required = {"perturbation_id", "latent_pca"}
    missing = sorted(required - set(latent))
    if missing:
        raise ValueError(f"{latent_npz} is missing arrays: {missing}")
    latent_ids = [str(value) for value in latent["perturbation_id"]]
    latent_index = {value: index for index, value in enumerate(latent_ids)}
    keep_mask = metadata["perturbation_id"].astype(str).isin(latent_index)
    rows = metadata.loc[keep_mask].copy().reset_index(drop=True)
    indices = np.asarray([latent_index[str(value)] for value in rows["perturbation_id"]], dtype=np.int64)
    return np.asarray(latent["latent_pca"][indices], dtype=np.float32), rows


def candidate_isoform_groups(metadata: pd.DataFrame, *, min_cells: int = 5) -> pd.DataFrame:
    required = {"gene_symbol", "isoform_id", "n_cells", "label_status", "response_score"}
    missing = sorted(required - set(metadata.columns))
    if missing:
        raise ValueError(f"metadata is missing required columns: {missing}")
    rows = metadata.loc[pd.to_numeric(metadata["n_cells"], errors="coerce").fillna(0) >= min_cells].copy()
    grouped = []
    for gene, group in rows.groupby("gene_symbol", dropna=False):
        labels = group["label_status"].astype(str)
        scores = pd.to_numeric(group["response_score"], errors="coerce")
        grouped.append(
            {
                "gene_symbol": str(gene),
                "n_isoforms": int(group.shape[0]),
                "n_labeled": int(labels.isin({"responder", "nonresponder"}).sum()),
                "n_responders": int((labels == "responder").sum()),
                "n_nonresponders": int((labels == "nonresponder").sum()),
                "n_ambiguous": int((labels == "ambiguous").sum()),
                "min_cells": int(pd.to_numeric(group["n_cells"], errors="coerce").min()),
                "max_cells": int(pd.to_numeric(group["n_cells"], errors="coerce").max()),
                "response_score_min": float(scores.min()) if scores.notna().any() else np.nan,
                "response_score_max": float(scores.max()) if scores.notna().any() else np.nan,
                "response_score_span": float(scores.max() - scores.min()) if scores.notna().any() else np.nan,
            }
        )
    out = pd.DataFrame(grouped)
    if out.empty:
        return out
    return out.sort_values(
        ["n_responders", "n_nonresponders", "response_score_span", "n_isoforms"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
