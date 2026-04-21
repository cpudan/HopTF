#!/usr/bin/env python3
# Example:
# python run_linear_probe.py \
#   --metadata /path/to/PERTURBATION_METADATA.csv \
#   --embedding-matrix /path/to/esmc_embeddings.npy \
#   --vocab /path/to/vocab.json \
#   --out /path/to/linear_probe_results.json

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
DEFAULT_FEATURE_SETS = ("esmc", "protein_length", "aa_composition", "gene_symbol_onehot")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run responder-vs-nonresponder linear-probe ablations on TF isoform features."
    )
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--embedding-matrix", type=Path, required=True)
    parser.add_argument("--vocab", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--feature-set", action="append", default=None)
    parser.add_argument("--group-column", default="gene_symbol")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--solver-tol", type=float, default=1.0e-2)
    parser.add_argument("--c", type=float, default=1.0)
    return parser.parse_args()


def load_metadata_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    labeled = [row for row in rows if str(row.get("label_status", "")) in {"responder", "nonresponder"}]
    if not labeled:
        raise ValueError("no responder/nonresponder rows found")
    return labeled


def load_vocab(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("vocab must be a non-empty JSON list")
    return [str(value) for value in payload]


def amino_acid_composition(sequence: str) -> np.ndarray:
    counts = np.asarray([str(sequence).count(aa) for aa in AMINO_ACIDS], dtype=np.float32)
    total = counts.sum()
    if total == 0:
        return counts
    return counts / total


def choose_splitter(
    y: np.ndarray,
    groups: np.ndarray | None,
    *,
    n_splits: int,
    random_state: int,
):
    """
    Use gene-grouped CV when it is feasible, otherwise fall back to ordinary
    stratified folds.
    """
    class_counts = np.bincount(y.astype(np.int64), minlength=2)
    if np.any(class_counts == 0):
        raise ValueError("both classes are required")
    realized_splits = min(int(n_splits), int(class_counts.min()))
    if realized_splits < 2:
        raise ValueError("at least two samples per class are required")
    if groups is None:
        splitter = StratifiedKFold(n_splits=realized_splits, shuffle=True, random_state=random_state)
        return splitter, realized_splits, "stratified"

    grouped_labels: dict[str, set[int]] = {}
    for group, label in zip(groups.astype(str), y.astype(int)):
        grouped_labels.setdefault(group, set()).add(int(label))
    class_group_counts = [
        sum(1 for labels in grouped_labels.values() if class_id in labels)
        for class_id in (0, 1)
    ]
    group_splits = min(realized_splits, min(class_group_counts))
    if group_splits < 2:
        splitter = StratifiedKFold(n_splits=realized_splits, shuffle=True, random_state=random_state)
        return splitter, realized_splits, "stratified_fallback"

    splitter = StratifiedGroupKFold(n_splits=group_splits, shuffle=True, random_state=random_state)
    return splitter, group_splits, "stratified_group"


def build_feature_matrix(
    rows: list[dict[str, str]],
    feature_set: str,
    embedding_matrix: np.ndarray,
    vocab: list[str],
) -> np.ndarray:
    if feature_set == "esmc":
        vocab_index = {value: idx for idx, value in enumerate(vocab)}
        indices = [vocab_index[str(row["isoform_embedding_id"])] for row in rows]
        return np.asarray(embedding_matrix[np.asarray(indices, dtype=np.int64)], dtype=np.float32)

    if feature_set == "protein_length":
        return np.asarray([[float(row["protein_aa_length"])] for row in rows], dtype=np.float32)

    if feature_set == "aa_composition":
        return np.vstack([amino_acid_composition(row.get("sequence", "")) for row in rows]).astype(np.float32)

    if feature_set == "gene_symbol_onehot":
        values = np.asarray([[str(row.get("gene_symbol", ""))] for row in rows], dtype=object)
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        return encoder.fit_transform(values).astype(np.float32)

    raise ValueError(f"unsupported feature set: {feature_set}")


def run_probe(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray | None,
    *,
    n_splits: int,
    random_state: int,
    max_iter: int,
    solver_tol: float,
    c: float,
) -> dict[str, object]:
    """
    Fit standardized logistic regression with out-of-fold evaluation.
    """
    splitter, realized_splits, split_kind = choose_splitter(
        y, groups, n_splits=n_splits, random_state=random_state
    )
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    penalty="l2",
                    solver="lbfgs",
                    class_weight="balanced",
                    max_iter=max_iter,
                    tol=solver_tol,
                    C=float(c),
                ),
            ),
        ]
    )

    oof_prob = np.zeros(y.shape[0], dtype=np.float64)
    oof_pred = np.zeros(y.shape[0], dtype=np.int64)
    folds: list[dict[str, int]] = []

    split_iter = splitter.split(x, y, groups) if split_kind == "stratified_group" else splitter.split(x, y)
    for fold_index, (train_idx, test_idx) in enumerate(split_iter, start=1):
        model.fit(x[train_idx], y[train_idx])
        prob = model.predict_proba(x[test_idx])[:, 1]
        pred = (prob >= 0.5).astype(np.int64)
        oof_prob[test_idx] = prob
        oof_pred[test_idx] = pred
        folds.append(
            {
                "fold": int(fold_index),
                "n_train": int(train_idx.size),
                "n_test": int(test_idx.size),
                "test_positive": int(y[test_idx].sum()),
            }
        )

    return {
        "AUROC": float(roc_auc_score(y, oof_prob)),
        "AUPRC": float(average_precision_score(y, oof_prob)),
        "balanced_accuracy": float(balanced_accuracy_score(y, oof_pred)),
        "n_samples": int(y.shape[0]),
        "n_positive": int(y.sum()),
        "n_negative": int((1 - y).sum()),
        "n_splits": int(realized_splits),
        "split_kind": split_kind,
        "c": float(c),
        "folds": folds,
    }


def main() -> None:
    args = parse_args()
    rows = load_metadata_rows(args.metadata)
    embedding_matrix = np.load(args.embedding_matrix, mmap_mode="r")
    if embedding_matrix.ndim != 2:
        raise ValueError("embedding matrix must be 2D")
    vocab = load_vocab(args.vocab)

    y = np.asarray([int(float(row["responder_label"])) for row in rows], dtype=np.int64)
    if args.group_column.lower() == "none":
        groups = None
    else:
        groups = np.asarray([str(row.get(args.group_column, "")) for row in rows], dtype=object)

    feature_sets = tuple(args.feature_set or DEFAULT_FEATURE_SETS)
    results: list[dict[str, object]] = []
    for feature_set in feature_sets:
        x = build_feature_matrix(rows, feature_set, np.asarray(embedding_matrix, dtype=np.float32), vocab)
        metrics = run_probe(
            x,
            y,
            groups,
            n_splits=args.n_splits,
            random_state=args.random_state,
            max_iter=args.max_iter,
            solver_tol=args.solver_tol,
            c=args.c,
        )
        results.append(
            {
                "feature_set": feature_set,
                "label_source": "metadata",
                "metrics": metrics,
            }
        )

    shuffled = np.random.default_rng(args.random_state).permutation(y)
    x = build_feature_matrix(rows, "esmc", np.asarray(embedding_matrix, dtype=np.float32), vocab)
    shuffled_metrics = run_probe(
        x,
        shuffled,
        groups,
        n_splits=args.n_splits,
        random_state=args.random_state,
        max_iter=args.max_iter,
        solver_tol=args.solver_tol,
        c=args.c,
    )
    results.append(
        {
            "feature_set": "esmc",
            "label_source": "shuffled_metadata_labels",
            "metrics": shuffled_metrics,
        }
    )

    payload = {
        "config": {
            "feature_sets": list(feature_sets),
            "group_column": None if groups is None else args.group_column,
            "n_splits": int(args.n_splits),
            "random_state": int(args.random_state),
            "max_iter": int(args.max_iter),
            "solver_tol": float(args.solver_tol),
            "c": float(args.c),
        },
        "n_labeled_rows": int(len(rows)),
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.out)


if __name__ == "__main__":
    main()
