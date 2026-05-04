#!/usr/bin/env python3
"""Evaluate endpoint baselines for sequence-conditioned perturbation prediction."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from hopfield_fitting_common import (
    DEFAULT_ESM,
    DEFAULT_METADATA,
    DEFAULT_OUTDIR,
    DEFAULT_VOCAB,
    ensure_dir,
    load_metadata,
    load_vocab,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--embedding-matrix", type=Path, default=DEFAULT_ESM)
    parser.add_argument("--vocab", type=Path, default=DEFAULT_VOCAB)
    parser.add_argument("--latents", type=Path, default=DEFAULT_OUTDIR / "perturbation_latents.npz")
    parser.add_argument("--hopfield-checkpoint", type=Path, default=None)
    parser.add_argument("--otcfm-metrics-dir", type=Path, default=None)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--min-cells", type=int, default=5)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--ridge-alpha", type=float, default=100.0)
    parser.add_argument("--random-state", type=int, default=0)
    return parser.parse_args()


def load_joined(args: argparse.Namespace) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    metadata = load_metadata(
        args.metadata,
        require_columns=[
            "perturbation_id",
            "isoform_embedding_id",
            "gene_symbol",
            "protein_aa_length",
            "n_cells",
            "response_score",
        ],
    )
    metadata = metadata.loc[pd.to_numeric(metadata["n_cells"], errors="coerce").fillna(0) >= args.min_cells].copy()
    vocab = load_vocab(args.vocab)
    vocab_index = {value: index for index, value in enumerate(vocab)}
    esm_matrix = np.load(args.embedding_matrix, mmap_mode="r")
    latent = np.load(args.latents, allow_pickle=False)
    latent_ids = {str(value): index for index, value in enumerate(latent["perturbation_id"])}

    rows = []
    esm_indices = []
    latent_indices = []
    for _, row in metadata.iterrows():
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
    y = np.asarray(latent["latent_pca"][np.asarray(latent_indices, dtype=np.int64)], dtype=np.float32)
    control_mean = np.asarray(latent["control_mean"], dtype=np.float32)
    return row_df, esm, y, control_mean


def numeric_covariates(rows: pd.DataFrame, columns: list[str]) -> np.ndarray:
    values = []
    for column in columns:
        if column == "log1p_n_cells":
            values.append(np.log1p(pd.to_numeric(rows["n_cells"], errors="coerce").fillna(0).to_numpy(dtype=np.float32)))
        else:
            values.append(pd.to_numeric(rows[column], errors="coerce").fillna(0).to_numpy(dtype=np.float32))
    return np.vstack(values).T.astype(np.float32)


def cosine_similarity_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    numerator = np.sum(a * b, axis=1)
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return numerator / np.maximum(denom, 1.0e-8)


def evaluate_predictions(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, object]:
    per_row_mse = np.mean((y_true - y_pred) ** 2, axis=1)
    cosine = cosine_similarity_rows(y_true, y_pred)
    return {
        "feature_set": name,
        "mse": float(mean_squared_error(y_true, y_pred)),
        "median_row_mse": float(np.median(per_row_mse)),
        "mean_cosine_similarity": float(np.mean(cosine)),
        "median_cosine_similarity": float(np.median(cosine)),
    }


def evaluate_grouped_ridge(
    name: str,
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    n_splits: int,
    ridge_alpha: float,
) -> tuple[dict[str, object], np.ndarray]:
    unique_groups = np.unique(groups.astype(str))
    realized_splits = min(int(n_splits), int(unique_groups.shape[0]))
    if realized_splits < 2:
        raise ValueError("at least two groups are required for grouped ridge evaluation")
    splitter = GroupKFold(n_splits=realized_splits)
    pred = np.zeros_like(y, dtype=np.float32)
    fold_records = []
    for fold_index, (train_idx, test_idx) in enumerate(splitter.split(x, y, groups), start=1):
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=float(ridge_alpha))),
            ]
        )
        model.fit(x[train_idx], y[train_idx])
        pred[test_idx] = model.predict(x[test_idx]).astype(np.float32)
        fold_records.append(
            {
                "fold": int(fold_index),
                "n_train": int(train_idx.size),
                "n_test": int(test_idx.size),
                "n_test_genes": int(np.unique(groups[test_idx].astype(str)).shape[0]),
            }
        )
    metrics = evaluate_predictions(name, y, pred)
    metrics.update({"n_splits": int(realized_splits), "split_kind": "group_kfold_gene_symbol", "folds": fold_records})
    return metrics, pred


def load_hopfield_query_features(checkpoint: Path, esm: np.ndarray) -> np.ndarray:
    import torch

    from train_hopfield_projection import make_model

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = make_model(int(payload["esm_dim"]), int(payload["key_dim"]), float(payload.get("beta", 15.0)))
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    with torch.no_grad():
        q = model.query(torch.as_tensor(np.asarray(esm, dtype=np.float32), dtype=torch.float32))
        q = torch.nn.functional.normalize(q, dim=-1)
    return q.numpy().astype(np.float32)


def collect_otcfm_metrics(path: Path | None) -> dict[str, object]:
    if path is None:
        return {}
    out = {}
    for name in [
        "otcfm_synthetic_metrics.json",
        "otcfm_real_overfit_metrics.json",
        "otcfm_leave_one_HNF4A_summary.json",
        "otcfm_leave_one_TP53_summary.json",
        "otcfm_leave_one_NFATC1_summary.json",
    ]:
        candidate = path / name
        if candidate.exists():
            out[name.removesuffix(".json")] = json.loads(candidate.read_text(encoding="utf-8"))
    return out


def main() -> None:
    args = parse_args()
    outdir = ensure_dir(args.outdir)
    rows, esm, y, control_mean = load_joined(args)
    groups = rows["gene_symbol"].astype(str).to_numpy()
    rng = np.random.default_rng(args.random_state)

    feature_sets: dict[str, np.ndarray] = {
        "length_only": numeric_covariates(rows, ["protein_aa_length"]),
        "cell_count_only": numeric_covariates(rows, ["log1p_n_cells"]),
        "length_plus_cell_count": numeric_covariates(rows, ["protein_aa_length", "log1p_n_cells"]),
        "esm_c": esm,
        "esm_c_shuffled_features": esm[rng.permutation(esm.shape[0])],
    }
    if "orf_nt_length" in rows.columns:
        feature_sets["length_cell_orf"] = numeric_covariates(
            rows, ["protein_aa_length", "orf_nt_length", "log1p_n_cells"]
        )
    if args.hopfield_checkpoint is not None and args.hopfield_checkpoint.exists():
        feature_sets["hopfield_query"] = load_hopfield_query_features(args.hopfield_checkpoint, esm)

    metrics = []
    predictions: dict[str, np.ndarray] = {}
    control_pred = np.repeat(control_mean.reshape(1, -1), repeats=y.shape[0], axis=0).astype(np.float32)
    metrics.append(evaluate_predictions("control_mean", y, control_pred))
    predictions["control_mean"] = control_pred

    for name, x in feature_sets.items():
        result, pred = evaluate_grouped_ridge(
            name,
            np.asarray(x, dtype=np.float32),
            y,
            groups,
            n_splits=args.n_splits,
            ridge_alpha=args.ridge_alpha,
        )
        metrics.append(result)
        predictions[name] = pred

    shuffled_y = y[rng.permutation(y.shape[0])]
    shuffled_result, shuffled_pred = evaluate_grouped_ridge(
        "esm_c_shuffled_endpoint_labels",
        esm,
        shuffled_y,
        groups,
        n_splits=args.n_splits,
        ridge_alpha=args.ridge_alpha,
    )
    shuffled_result["evaluated_against"] = "true_endpoints_after_training_on_shuffled_endpoints"
    shuffled_result.update(evaluate_predictions("esm_c_shuffled_endpoint_labels", y, shuffled_pred))
    metrics.append(shuffled_result)
    predictions["esm_c_shuffled_endpoint_labels"] = shuffled_pred

    metrics_df = pd.DataFrame(metrics).sort_values("mse").reset_index(drop=True)
    metrics_path = outdir / "sequence_endpoint_baselines.csv"
    json_path = outdir / "sequence_endpoint_baselines_summary.json"
    md_path = outdir / "sequence_endpoint_baselines.md"
    predictions_path = outdir / "sequence_endpoint_baselines_predictions.npz"
    metrics_df.to_csv(metrics_path, index=False)
    np.savez_compressed(predictions_path, **predictions, target=y)

    summary = {
        "n_rows": int(rows.shape[0]),
        "latent_dim": int(y.shape[1]),
        "n_gene_groups": int(np.unique(groups).shape[0]),
        "n_splits": int(args.n_splits),
        "ridge_alpha": float(args.ridge_alpha),
        "metrics_csv": metrics_path,
        "predictions_npz": predictions_path,
        "best_feature_set": str(metrics_df.iloc[0]["feature_set"]) if not metrics_df.empty else None,
        "metrics": metrics_df.to_dict(orient="records"),
        "otcfm_metrics": collect_otcfm_metrics(args.otcfm_metrics_dir),
    }
    write_json(json_path, summary)

    lines = [
        "# Sequence Endpoint Baselines",
        "",
        f"Rows: `{rows.shape[0]}`",
        f"Latent dim: `{y.shape[1]}`",
        f"Gene groups: `{np.unique(groups).shape[0]}`",
        f"Split: grouped by `gene_symbol`, `{args.n_splits}` folds",
        "",
        "## Metrics",
        "",
        "| Feature set | MSE | Median row MSE | Mean cosine |",
        "|---|---:|---:|---:|",
    ]
    for _, row in metrics_df.iterrows():
        lines.append(
            f"| `{row['feature_set']}` | {float(row['mse']):.6f} | "
            f"{float(row['median_row_mse']):.6f} | {float(row['mean_cosine_similarity']):.6f} |"
        )
    if summary["otcfm_metrics"]:
        lines.extend(["", "## TorchCFM Smoke Metrics", ""])
        for name, payload in summary["otcfm_metrics"].items():
            if "endpoint_mse_fraction_of_baseline" in payload:
                lines.append(f"- `{name}` endpoint MSE fraction of baseline: `{payload['endpoint_mse_fraction_of_baseline']}`")
            elif "mean_endpoint_mse_fraction_of_baseline" in payload:
                lines.append(
                    f"- `{name}` mean endpoint MSE fraction of baseline: "
                    f"`{payload['mean_endpoint_mse_fraction_of_baseline']}`"
                )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {metrics_path} {json_path} {md_path}")


if __name__ == "__main__":
    main()
