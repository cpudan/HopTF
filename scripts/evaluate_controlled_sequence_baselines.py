#!/usr/bin/env python3
"""Evaluate whether sequence features add endpoint signal after artifact controls."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
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


@dataclass
class EvaluationResult:
    metrics: dict[str, object]
    predictions: np.ndarray
    evaluated_mask: np.ndarray
    stratum_records: list[dict[str, object]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--embedding-matrix", type=Path, default=DEFAULT_ESM)
    parser.add_argument("--vocab", type=Path, default=DEFAULT_VOCAB)
    parser.add_argument("--latents", type=Path, default=DEFAULT_OUTDIR / "perturbation_latents.npz")
    parser.add_argument("--hopfield-checkpoint", type=Path, default=None)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--min-cells", type=int, default=5)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-bins", type=int, default=4)
    parser.add_argument("--min-stratum-rows", type=int, default=80)
    parser.add_argument("--min-stratum-groups", type=int, default=8)
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
            "orf_nt_length",
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
        perturbation_id = str(row["perturbation_id"])
        if iso_id in vocab_index and perturbation_id in latent_ids:
            rows.append(row)
            esm_indices.append(vocab_index[iso_id])
            latent_indices.append(latent_ids[perturbation_id])
    if not rows:
        raise ValueError("no rows matched metadata, ESM-C vocab, and latent endpoints")

    row_df = pd.DataFrame(rows).reset_index(drop=True)
    esm = np.asarray(esm_matrix[np.asarray(esm_indices, dtype=np.int64)], dtype=np.float32)
    y = np.asarray(latent["latent_pca"][np.asarray(latent_indices, dtype=np.int64)], dtype=np.float32)
    control_mean = np.asarray(latent["control_mean"], dtype=np.float32)
    return row_df, esm, y, control_mean


def numeric_covariates(rows: pd.DataFrame) -> np.ndarray:
    protein_len = pd.to_numeric(rows["protein_aa_length"], errors="coerce").fillna(0).to_numpy(dtype=np.float32)
    orf_len = pd.to_numeric(rows["orf_nt_length"], errors="coerce").fillna(0).to_numpy(dtype=np.float32)
    log_cells = np.log1p(pd.to_numeric(rows["n_cells"], errors="coerce").fillna(0).to_numpy(dtype=np.float32))
    return np.vstack([protein_len, orf_len, log_cells]).T.astype(np.float32)


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


def make_bins(values: pd.Series, *, n_bins: int, prefix: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    try:
        binned = pd.qcut(numeric, q=n_bins, labels=False, duplicates="drop")
    except ValueError:
        binned = pd.Series(np.zeros(len(values), dtype=np.int64), index=values.index)
    binned = binned.astype("Int64")
    return binned.map(lambda value: f"{prefix}{int(value)}" if pd.notna(value) else f"{prefix}missing")


def add_match_bins(rows: pd.DataFrame, *, n_bins: int) -> pd.DataFrame:
    out = rows.copy()
    out["length_bin"] = make_bins(out["protein_aa_length"], n_bins=n_bins, prefix="length_q")
    out["cell_count_bin"] = make_bins(np.log1p(pd.to_numeric(out["n_cells"], errors="coerce").fillna(0)), n_bins=n_bins, prefix="cells_q")
    out["length_cell_bin"] = out["length_bin"].astype(str) + "__" + out["cell_count_bin"].astype(str)
    return out


def cosine_similarity_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    numerator = np.sum(a * b, axis=1)
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return numerator / np.maximum(denom, 1.0e-8)


def metric_dict(
    *,
    context: str,
    feature_set: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    evaluated_mask: np.ndarray,
    n_partitions: int,
    n_partitions_evaluated: int,
    n_partitions_skipped: int,
) -> dict[str, object]:
    y_eval = y_true[evaluated_mask]
    pred_eval = y_pred[evaluated_mask]
    per_row_mse = np.mean((y_eval - pred_eval) ** 2, axis=1)
    cosine = cosine_similarity_rows(y_eval, pred_eval)
    return {
        "context": context,
        "feature_set": feature_set,
        "n_rows_total": int(y_true.shape[0]),
        "n_rows_evaluated": int(evaluated_mask.sum()),
        "n_partitions": int(n_partitions),
        "n_partitions_evaluated": int(n_partitions_evaluated),
        "n_partitions_skipped": int(n_partitions_skipped),
        "mse": float(mean_squared_error(y_eval, pred_eval)),
        "median_row_mse": float(np.median(per_row_mse)),
        "mean_cosine_similarity": float(np.mean(cosine)),
        "median_cosine_similarity": float(np.median(cosine)),
    }


def fit_predict_grouped_ridge(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    n_splits: int,
    ridge_alpha: float,
) -> np.ndarray:
    unique_groups = np.unique(groups.astype(str))
    realized_splits = min(int(n_splits), int(unique_groups.shape[0]))
    if realized_splits < 2:
        raise ValueError("at least two groups are required")
    pred = np.zeros_like(y, dtype=np.float32)
    splitter = GroupKFold(n_splits=realized_splits)
    for train_idx, test_idx in splitter.split(x, y, groups):
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=float(ridge_alpha))),
            ]
        )
        model.fit(x[train_idx], y[train_idx])
        pred[test_idx] = model.predict(x[test_idx]).astype(np.float32)
    return pred


def evaluate_context(
    *,
    context: str,
    partition: np.ndarray,
    feature_set: str,
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
    ridge_alpha: float,
    min_stratum_rows: int,
    min_stratum_groups: int,
) -> EvaluationResult:
    pred = np.full_like(y, np.nan, dtype=np.float32)
    evaluated_mask = np.zeros(y.shape[0], dtype=bool)
    stratum_records: list[dict[str, object]] = []
    partitions = np.unique(partition.astype(str))
    for value in partitions:
        idx = np.flatnonzero(partition.astype(str) == value)
        unique_groups = np.unique(groups[idx].astype(str))
        if idx.size < min_stratum_rows or unique_groups.shape[0] < min_stratum_groups:
            stratum_records.append(
                {
                    "context": context,
                    "feature_set": feature_set,
                    "stratum": value,
                    "status": "skipped",
                    "n_rows": int(idx.size),
                    "n_groups": int(unique_groups.shape[0]),
                    "reason": "too_few_rows_or_groups",
                }
            )
            continue
        pred[idx] = fit_predict_grouped_ridge(
            x[idx],
            y[idx],
            groups[idx],
            n_splits=n_splits,
            ridge_alpha=ridge_alpha,
        )
        evaluated_mask[idx] = True
        stratum_metrics = metric_dict(
            context=context,
            feature_set=feature_set,
            y_true=y[idx],
            y_pred=pred[idx],
            evaluated_mask=np.ones(idx.size, dtype=bool),
            n_partitions=1,
            n_partitions_evaluated=1,
            n_partitions_skipped=0,
        )
        stratum_metrics.update({"stratum": value, "status": "evaluated", "n_groups": int(unique_groups.shape[0])})
        stratum_records.append(stratum_metrics)

    if not evaluated_mask.any():
        raise ValueError(f"{context}/{feature_set} evaluated no rows")
    metrics = metric_dict(
        context=context,
        feature_set=feature_set,
        y_true=y,
        y_pred=np.nan_to_num(pred, nan=0.0),
        evaluated_mask=evaluated_mask,
        n_partitions=int(partitions.shape[0]),
        n_partitions_evaluated=int(sum(record["status"] == "evaluated" for record in stratum_records)),
        n_partitions_skipped=int(sum(record["status"] == "skipped" for record in stratum_records)),
    )
    return EvaluationResult(metrics=metrics, predictions=np.nan_to_num(pred, nan=0.0), evaluated_mask=evaluated_mask, stratum_records=stratum_records)


def evaluate_overall(
    *,
    feature_set: str,
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
    ridge_alpha: float,
) -> EvaluationResult:
    pred = fit_predict_grouped_ridge(x, y, groups, n_splits=n_splits, ridge_alpha=ridge_alpha)
    mask = np.ones(y.shape[0], dtype=bool)
    metrics = metric_dict(
        context="overall_grouped",
        feature_set=feature_set,
        y_true=y,
        y_pred=pred,
        evaluated_mask=mask,
        n_partitions=1,
        n_partitions_evaluated=1,
        n_partitions_skipped=0,
    )
    return EvaluationResult(metrics=metrics, predictions=pred, evaluated_mask=mask, stratum_records=[])


def add_artifact_deltas(metrics: pd.DataFrame) -> pd.DataFrame:
    out = metrics.copy()
    out["mse_delta_vs_artifacts"] = np.nan
    out["mse_pct_change_vs_artifacts"] = np.nan
    out["beats_artifacts"] = pd.NA
    for context, group in out.groupby("context"):
        baseline = group.loc[group["feature_set"] == "artifacts", "mse"]
        if baseline.empty:
            continue
        base_mse = float(baseline.iloc[0])
        idx = out["context"] == context
        out.loc[idx, "mse_delta_vs_artifacts"] = out.loc[idx, "mse"].astype(float) - base_mse
        out.loc[idx, "mse_pct_change_vs_artifacts"] = 100.0 * (out.loc[idx, "mse"].astype(float) / base_mse - 1.0)
        out.loc[idx, "beats_artifacts"] = out.loc[idx, "mse"].astype(float) < base_mse
    return out


def main() -> None:
    args = parse_args()
    outdir = ensure_dir(args.outdir)
    rows, esm, y, control_mean = load_joined(args)
    rows = add_match_bins(rows, n_bins=args.n_bins)
    groups = rows["gene_symbol"].astype(str).to_numpy()
    artifacts = numeric_covariates(rows)
    rng = np.random.default_rng(args.random_state)

    feature_sets: dict[str, np.ndarray] = {
        "artifacts": artifacts,
        "esm_c": esm,
        "artifacts_plus_esm_c": np.concatenate([artifacts, esm], axis=1).astype(np.float32),
        "artifacts_plus_shuffled_esm_c": np.concatenate([artifacts, esm[rng.permutation(esm.shape[0])]], axis=1).astype(np.float32),
    }
    if args.hopfield_checkpoint is not None and args.hopfield_checkpoint.exists():
        hopfield_query = load_hopfield_query_features(args.hopfield_checkpoint, esm)
        feature_sets["hopfield_query"] = hopfield_query
        feature_sets["artifacts_plus_hopfield_query"] = np.concatenate([artifacts, hopfield_query], axis=1).astype(np.float32)

    contexts = {
        "overall_grouped": np.asarray(["all"] * rows.shape[0], dtype=object),
        "length_matched": rows["length_bin"].astype(str).to_numpy(),
        "cell_count_matched": rows["cell_count_bin"].astype(str).to_numpy(),
        "length_cell_matched": rows["length_cell_bin"].astype(str).to_numpy(),
    }

    metric_records: list[dict[str, object]] = []
    stratum_records: list[dict[str, object]] = []
    prediction_payload: dict[str, np.ndarray] = {"target": y}

    control_pred = np.repeat(control_mean.reshape(1, -1), repeats=y.shape[0], axis=0).astype(np.float32)
    control_metrics = metric_dict(
        context="overall_grouped",
        feature_set="control_mean",
        y_true=y,
        y_pred=control_pred,
        evaluated_mask=np.ones(y.shape[0], dtype=bool),
        n_partitions=1,
        n_partitions_evaluated=1,
        n_partitions_skipped=0,
    )
    metric_records.append(control_metrics)
    prediction_payload["overall_grouped__control_mean"] = control_pred

    for context, partition in contexts.items():
        for feature_name, x in feature_sets.items():
            if context == "overall_grouped":
                result = evaluate_overall(
                    feature_set=feature_name,
                    x=x,
                    y=y,
                    groups=groups,
                    n_splits=args.n_splits,
                    ridge_alpha=args.ridge_alpha,
                )
            else:
                result = evaluate_context(
                    context=context,
                    partition=partition,
                    feature_set=feature_name,
                    x=x,
                    y=y,
                    groups=groups,
                    n_splits=args.n_splits,
                    ridge_alpha=args.ridge_alpha,
                    min_stratum_rows=args.min_stratum_rows,
                    min_stratum_groups=args.min_stratum_groups,
                )
            metric_records.append(result.metrics)
            stratum_records.extend(result.stratum_records)
            prediction_payload[f"{context}__{feature_name}"] = result.predictions
            prediction_payload[f"{context}__{feature_name}__mask"] = result.evaluated_mask.astype(np.uint8)

    metrics_df = add_artifact_deltas(pd.DataFrame(metric_records))
    metrics_df = metrics_df.sort_values(["context", "mse"]).reset_index(drop=True)
    strata_df = pd.DataFrame(stratum_records)

    metrics_path = outdir / "controlled_sequence_baselines.csv"
    strata_path = outdir / "controlled_sequence_strata.csv"
    summary_path = outdir / "controlled_sequence_baselines_summary.json"
    report_path = outdir / "controlled_evaluation_report.md"
    predictions_path = outdir / "controlled_sequence_predictions.npz"
    metrics_df.to_csv(metrics_path, index=False)
    strata_df.to_csv(strata_path, index=False)
    np.savez_compressed(predictions_path, **prediction_payload)

    summary = {
        "n_rows": int(rows.shape[0]),
        "latent_dim": int(y.shape[1]),
        "n_gene_groups": int(np.unique(groups).shape[0]),
        "n_splits": int(args.n_splits),
        "n_bins": int(args.n_bins),
        "min_stratum_rows": int(args.min_stratum_rows),
        "min_stratum_groups": int(args.min_stratum_groups),
        "ridge_alpha": float(args.ridge_alpha),
        "metrics_csv": metrics_path,
        "strata_csv": strata_path,
        "predictions_npz": predictions_path,
        "report_md": report_path,
        "metrics": metrics_df.to_dict(orient="records"),
    }
    write_json(summary_path, summary)

    lines = [
        "# Controlled Sequence Baseline Evaluation",
        "",
        "Question: do ESM-C or Hopfield-query sequence features add endpoint-prediction signal after controlling for length, ORF length, and cell-count artifacts?",
        "",
        f"Rows: `{rows.shape[0]}`",
        f"Latent dim: `{y.shape[1]}`",
        f"Gene groups: `{np.unique(groups).shape[0]}`",
        f"Ridge alpha: `{args.ridge_alpha}`",
        "",
        "## Overall Result",
        "",
    ]
    overall = metrics_df.loc[metrics_df["context"] == "overall_grouped"].copy()
    lines.extend(
        [
            "| Feature set | MSE | Delta vs artifacts | % change vs artifacts | Mean cosine |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for _, row in overall.iterrows():
        delta = "" if pd.isna(row["mse_delta_vs_artifacts"]) else f"{float(row['mse_delta_vs_artifacts']):.6f}"
        pct = "" if pd.isna(row["mse_pct_change_vs_artifacts"]) else f"{float(row['mse_pct_change_vs_artifacts']):.2f}%"
        lines.append(
            f"| `{row['feature_set']}` | {float(row['mse']):.6f} | {delta} | {pct} | "
            f"{float(row['mean_cosine_similarity']):.6f} |"
        )

    lines.extend(["", "## Matched Contexts", ""])
    for context in ["length_matched", "cell_count_matched", "length_cell_matched"]:
        current = metrics_df.loc[metrics_df["context"] == context].copy()
        lines.extend([f"### {context}", "", "| Feature set | Rows | MSE | Delta vs artifacts | % change vs artifacts |", "|---|---:|---:|---:|---:|"])
        for _, row in current.iterrows():
            lines.append(
                f"| `{row['feature_set']}` | {int(row['n_rows_evaluated'])} | {float(row['mse']):.6f} | "
                f"{float(row['mse_delta_vs_artifacts']):.6f} | {float(row['mse_pct_change_vs_artifacts']):.2f}% |"
            )
        lines.append("")

    lines.extend(
        [
            "## Interpretation",
            "",
            "- Negative `Delta vs artifacts` means the feature set improved over the artifact-only model in that context.",
            "- `artifacts_plus_shuffled_esm_c` is the control for extra dimensionality after artifact covariates.",
            "- The matched contexts train/evaluate within length bins, cell-count bins, or combined length-cell bins to reduce gross artifact shortcuts.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {metrics_path} {strata_path} {summary_path} {report_path}")


if __name__ == "__main__":
    main()
