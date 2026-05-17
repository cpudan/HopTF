#!/usr/bin/env python3
"""Measure how WT-vs-mutant TF sequences change Hopfield retrieval weights."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


LABEL_NAMES = {
    "benign": "Benign / likely benign",
    "deleterious": "Pathogenic / likely pathogenic",
}
PLOT_LABEL_NAMES = {
    "benign": "Benign",
    "deleterious": "Pathogenic",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--wt-embedding-matrix", type=Path, required=True)
    parser.add_argument("--wt-vocab", type=Path, required=True)
    parser.add_argument("--mutant-embedding-matrix", type=Path, required=True)
    parser.add_argument("--mutant-vocab", type=Path, required=True)
    parser.add_argument("--keys", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument(
        "--panel-description",
        default="the 400 coordinate-checked ClinVar missense variants from the expanded panel",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--topk", type=int, nargs="+", default=[100, 500, 1000])
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def load_json(path: Path):
    return json.loads(path.read_text())


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def require_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except Exception as exc:  # pragma: no cover - dependency checked at runtime
        raise RuntimeError("PyTorch is required for retrieval-delta analysis") from exc
    return torch, nn, F


def require_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def finite_float(value):
    if value is None:
        return None
    value = float(value)
    if math.isfinite(value):
        return value
    return None


def auc_from_scores(y_true: np.ndarray, scores: np.ndarray) -> float | None:
    mask = np.isfinite(scores)
    y = y_true[mask].astype(int)
    s = scores[mask].astype(float)
    n_pos = int(y.sum())
    n_neg = int((1 - y).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)
    _, inverse, counts = np.unique(s, return_inverse=True, return_counts=True)
    tied_rank_sum = np.bincount(inverse, weights=ranks)
    avg_ranks = tied_rank_sum / counts
    ranks = avg_ranks[inverse]
    rank_sum_pos = float(ranks[y == 1].sum())
    u_pos = rank_sum_pos - n_pos * (n_pos + 1) / 2.0
    return float(u_pos / (n_pos * n_neg))


def mannwhitney(values_a: np.ndarray, values_b: np.ndarray, alternative: str) -> dict:
    values_a = values_a[np.isfinite(values_a)]
    values_b = values_b[np.isfinite(values_b)]
    if len(values_a) == 0 or len(values_b) == 0:
        return {"statistic": None, "p_value": None, "alternative": alternative}
    try:
        from scipy.stats import mannwhitneyu

        result = mannwhitneyu(values_a, values_b, alternative=alternative)
        return {
            "statistic": finite_float(result.statistic),
            "p_value": finite_float(result.pvalue),
            "alternative": alternative,
        }
    except Exception:
        return {"statistic": None, "p_value": None, "alternative": alternative}


def spearman(x: np.ndarray, y: np.ndarray) -> dict:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3:
        return {"rho": None, "p_value": None, "n": int(mask.sum())}
    try:
        from scipy.stats import spearmanr

        result = spearmanr(x[mask], y[mask])
        return {"rho": finite_float(result.statistic), "p_value": finite_float(result.pvalue), "n": int(mask.sum())}
    except Exception:
        rho = pd.Series(x[mask]).corr(pd.Series(y[mask]), method="spearman")
        return {"rho": finite_float(rho), "p_value": None, "n": int(mask.sum())}


def make_projector(checkpoint: dict, device: str):
    torch, nn, F = require_torch()

    class QueryProjector(nn.Module):
        def __init__(self, esm_dim: int, key_dim: int) -> None:
            super().__init__()
            self.query = nn.Sequential(nn.Linear(esm_dim, key_dim), nn.LayerNorm(key_dim))

        def forward(self, esm):
            return F.normalize(self.query(esm), dim=-1)

    model = QueryProjector(int(checkpoint["esm_dim"]), int(checkpoint["key_dim"])).to(device)
    state = {
        key: value
        for key, value in checkpoint["model_state_dict"].items()
        if key.startswith("query.")
    }
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def resolve_device(value: str):
    torch, _, _ = require_torch()
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    requested = torch.device(value)
    if requested.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return requested


def prepare_rows(panel: pd.DataFrame, wt_vocab: list[str], mutant_vocab: list[str]) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    wt_index = {str(value): index for index, value in enumerate(wt_vocab)}
    mutant_index = {str(value): index for index, value in enumerate(mutant_vocab)}

    rows = panel.copy()
    rows = rows.loc[rows["status"].astype(str).eq("ok")]
    rows = rows.loc[rows["prediction_status"].astype(str).eq("ok")]
    rows = rows.loc[rows["label_class"].astype(str).isin(["benign", "deleterious"])]
    rows["wt_embedding_row"] = rows["isoform_embedding_id"].map(lambda value: wt_index.get(str(value), -1))
    rows["mutant_embedding_row"] = rows["mutant_embedding_id"].map(lambda value: mutant_index.get(str(value), -1))
    rows = rows.loc[(rows["wt_embedding_row"] >= 0) & (rows["mutant_embedding_row"] >= 0)].copy()
    rows = rows.reset_index(drop=True)
    if rows.empty:
        raise ValueError("No panel rows matched WT and mutant embedding vocabularies")
    return rows, rows["wt_embedding_row"].to_numpy(np.int64), rows["mutant_embedding_row"].to_numpy(np.int64)


def compute_metrics(args: argparse.Namespace) -> tuple[pd.DataFrame, dict]:
    torch, _, F = require_torch()
    device = resolve_device(args.device)
    args.outdir.mkdir(parents=True, exist_ok=True)

    panel = pd.read_csv(args.panel)
    wt_vocab = load_json(args.wt_vocab)
    mutant_vocab = load_json(args.mutant_vocab)
    rows, wt_indices, mutant_indices = prepare_rows(panel, wt_vocab, mutant_vocab)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    beta = float(checkpoint.get("beta", 1.0))
    model = make_projector(checkpoint, device)

    wt_embeddings = np.load(args.wt_embedding_matrix, mmap_mode="r")
    mutant_embeddings = np.load(args.mutant_embedding_matrix, mmap_mode="r")
    keys_np = np.asarray(np.load(args.keys, mmap_mode="r"), dtype=np.float32)
    if keys_np.shape[1] != int(checkpoint["key_dim"]):
        raise ValueError(f"Key dimension {keys_np.shape[1]} does not match checkpoint key_dim {checkpoint['key_dim']}")

    keys = torch.as_tensor(keys_np, dtype=torch.float32, device=device)
    keys = F.normalize(keys, dim=-1)
    eps = 1.0e-12
    max_topk = max(args.topk)
    records: list[dict] = []

    with torch.no_grad():
        for start in range(0, len(rows), args.batch_size):
            stop = min(start + args.batch_size, len(rows))
            wt_batch = np.asarray(wt_embeddings[wt_indices[start:stop]], dtype=np.float32)
            mutant_batch = np.asarray(mutant_embeddings[mutant_indices[start:stop]], dtype=np.float32)
            esmc_l2 = np.linalg.norm(mutant_batch - wt_batch, axis=1)
            wt_t = torch.as_tensor(wt_batch, dtype=torch.float32, device=device)
            mutant_t = torch.as_tensor(mutant_batch, dtype=torch.float32, device=device)

            q_wt = model(wt_t)
            q_mut = model(mutant_t)
            logits_wt = beta * (q_wt @ keys.T)
            logits_mut = beta * (q_mut @ keys.T)
            attn_wt = logits_wt.softmax(dim=-1)
            attn_mut = logits_mut.softmax(dim=-1)
            midpoint = 0.5 * (attn_wt + attn_mut)

            entropy_wt = -(attn_wt * attn_wt.clamp_min(eps).log()).sum(dim=-1)
            entropy_mut = -(attn_mut * attn_mut.clamp_min(eps).log()).sum(dim=-1)
            jsd = 0.5 * (
                (attn_wt * (attn_wt.clamp_min(eps).log() - midpoint.clamp_min(eps).log())).sum(dim=-1)
                + (attn_mut * (attn_mut.clamp_min(eps).log() - midpoint.clamp_min(eps).log())).sum(dim=-1)
            )
            attention_l1 = (attn_wt - attn_mut).abs().sum(dim=-1)
            attention_cosine = F.cosine_similarity(attn_wt, attn_mut, dim=-1)
            q_l2 = (q_mut - q_wt).norm(dim=-1)
            q_cosine = (q_wt * q_mut).sum(dim=-1)
            mu_wt = attn_wt @ keys
            mu_mut = attn_mut @ keys
            mu_l2 = (mu_mut - mu_wt).norm(dim=-1)
            mu_cosine = F.cosine_similarity(mu_wt, mu_mut, dim=-1)

            wt_sorted = torch.topk(attn_wt, k=max_topk, dim=-1)
            mut_sorted = torch.topk(attn_mut, k=max_topk, dim=-1)
            wt_top_values = wt_sorted.values
            mut_top_values = mut_sorted.values
            wt_top_indices = wt_sorted.indices.detach().cpu().numpy()
            mut_top_indices = mut_sorted.indices.detach().cpu().numpy()

            batch_values = {
                "q_l2": q_l2.detach().cpu().numpy(),
                "q_cosine": q_cosine.detach().cpu().numpy(),
                "attention_jsd": jsd.detach().cpu().numpy(),
                "attention_l1": attention_l1.detach().cpu().numpy(),
                "attention_cosine": attention_cosine.detach().cpu().numpy(),
                "attention_entropy_wt": entropy_wt.detach().cpu().numpy(),
                "attention_entropy_mutant": entropy_mut.detach().cpu().numpy(),
                "attention_max_wt": attn_wt.max(dim=-1).values.detach().cpu().numpy(),
                "attention_max_mutant": attn_mut.max(dim=-1).values.detach().cpu().numpy(),
                "mu_l2": mu_l2.detach().cpu().numpy(),
                "mu_cosine": mu_cosine.detach().cpu().numpy(),
            }

            for offset in range(stop - start):
                row = rows.iloc[start + offset].to_dict()
                record = dict(row)
                if (
                    "esmc_l2_distance_from_wt" not in record
                    or pd.isna(record["esmc_l2_distance_from_wt"])
                ):
                    record["esmc_l2_distance_from_wt"] = finite_float(esmc_l2[offset])
                record["computed_esmc_l2_distance_from_wt"] = finite_float(esmc_l2[offset])
                for key, values in batch_values.items():
                    record[key] = finite_float(values[offset])
                record["delta_attention_entropy"] = record["attention_entropy_mutant"] - record["attention_entropy_wt"]
                record["delta_attention_max"] = record["attention_max_mutant"] - record["attention_max_wt"]
                for k in args.topk:
                    wt_set = set(wt_top_indices[offset, :k].tolist())
                    mut_set = set(mut_top_indices[offset, :k].tolist())
                    record[f"top{k}_overlap_fraction"] = len(wt_set & mut_set) / float(k)
                    record[f"top{k}_mass_wt"] = finite_float(wt_top_values[offset, :k].sum().detach().cpu())
                    record[f"top{k}_mass_mutant"] = finite_float(mut_top_values[offset, :k].sum().detach().cpu())
                    record[f"delta_top{k}_mass"] = record[f"top{k}_mass_mutant"] - record[f"top{k}_mass_wt"]
                records.append(record)

    metrics = pd.DataFrame(records)
    if "percent_delta" not in metrics.columns and {
        "mutant_minus_wt_predicted_response_l2",
        "wt_predicted_response_l2",
    }.issubset(metrics.columns):
        denom = pd.to_numeric(metrics["wt_predicted_response_l2"], errors="coerce").replace(0, np.nan)
        metrics["percent_delta"] = (
            100.0
            * pd.to_numeric(metrics["mutant_minus_wt_predicted_response_l2"], errors="coerce")
            / denom
        )
    run_info = {
        "status": "ok",
        "n_rows": int(len(metrics)),
        "n_genes": int(metrics["gene_symbol"].nunique()),
        "counts_by_class": {str(k): int(v) for k, v in metrics["label_class"].value_counts().sort_index().items()},
        "genes_by_class": {
            str(k): int(v)
            for k, v in metrics.groupby("label_class")["gene_symbol"].nunique().sort_index().items()
        },
        "checkpoint": str(args.checkpoint),
        "checkpoint_beta": beta,
        "key_matrix": str(args.keys),
        "n_keys": int(keys_np.shape[0]),
        "key_dim": int(keys_np.shape[1]),
        "query_dim": int(checkpoint["key_dim"]),
        "device": str(device),
        "topk": [int(value) for value in args.topk],
        "panel_description": str(args.panel_description),
        "retrieved_representation": "attention-weighted mean of L2-normalized AlphaGenome key vectors",
    }
    return metrics, run_info


def summarize(metrics: pd.DataFrame, run_info: dict, topk: list[int]) -> dict:
    class_summary: dict[str, dict] = {}
    summary_columns = [
        "esmc_l2_distance_from_wt",
        "q_l2",
        "attention_jsd",
        "attention_l1",
        "mu_l2",
        "percent_delta",
    ] + [f"top{k}_overlap_fraction" for k in topk]

    for label, group in metrics.groupby("label_class"):
        label_summary = {
            "n": int(len(group)),
            "genes": int(group["gene_symbol"].nunique()),
        }
        for column in summary_columns:
            if column in group:
                values = pd.to_numeric(group[column], errors="coerce")
                label_summary[column] = {
                    "median": finite_float(values.median()),
                    "mean": finite_float(values.mean()),
                    "q25": finite_float(values.quantile(0.25)),
                    "q75": finite_float(values.quantile(0.75)),
                }
        class_summary[str(label)] = label_summary

    y = metrics["label_class"].astype(str).eq("deleterious").to_numpy(int)
    benign = metrics.loc[metrics["label_class"].astype(str).eq("benign")]
    deleterious = metrics.loc[metrics["label_class"].astype(str).eq("deleterious")]
    tests = {}
    for column in ["q_l2", "attention_jsd", "attention_l1", "mu_l2"]:
        tests[f"{column}_deleterious_greater_than_benign"] = mannwhitney(
            pd.to_numeric(deleterious[column], errors="coerce").to_numpy(float),
            pd.to_numeric(benign[column], errors="coerce").to_numpy(float),
            alternative="greater",
        )
    for k in topk:
        column = f"top{k}_overlap_fraction"
        tests[f"{column}_deleterious_less_than_benign"] = mannwhitney(
            pd.to_numeric(deleterious[column], errors="coerce").to_numpy(float),
            pd.to_numeric(benign[column], errors="coerce").to_numpy(float),
            alternative="less",
        )

    auroc = {}
    high_deleterious_columns = ["q_l2", "attention_jsd", "attention_l1", "mu_l2", "esmc_l2_distance_from_wt"]
    for column in high_deleterious_columns:
        auroc[column] = finite_float(auc_from_scores(y, pd.to_numeric(metrics[column], errors="coerce").to_numpy(float)))
    for k in topk:
        column = f"top{k}_overlap_fraction"
        auroc[f"negative_{column}"] = finite_float(
            auc_from_scores(y, -pd.to_numeric(metrics[column], errors="coerce").to_numpy(float))
        )

    correlations = {}
    for column in ["q_l2", "attention_jsd", "attention_l1", "mu_l2"] + [f"top{k}_overlap_fraction" for k in topk]:
        values = pd.to_numeric(metrics[column], errors="coerce").to_numpy(float)
        correlations[f"{column}_vs_esmc_l2_distance_from_wt"] = spearman(
            values, pd.to_numeric(metrics["esmc_l2_distance_from_wt"], errors="coerce").to_numpy(float)
        )
        correlations[f"{column}_vs_percent_delta"] = spearman(
            values, pd.to_numeric(metrics["percent_delta"], errors="coerce").to_numpy(float)
        )

    return {
        **run_info,
        "class_summary": class_summary,
        "one_sided_tests": tests,
        "auroc_for_deleterious": auroc,
        "spearman_correlations": correlations,
    }


def box_and_points(ax, data: pd.DataFrame, column: str, ylabel: str) -> None:
    labels = ["benign", "deleterious"]
    values = [pd.to_numeric(data.loc[data["label_class"].eq(label), column], errors="coerce").dropna() for label in labels]
    positions = np.arange(len(labels), dtype=float)
    ax.boxplot(values, positions=positions, widths=0.5, showfliers=False)
    rng = np.random.default_rng(7)
    colors = ["#4C78A8", "#E45756"]
    for position, series, color in zip(positions, values, colors):
        jitter = rng.normal(0, 0.045, size=len(series))
        ax.scatter(np.full(len(series), position) + jitter, series, s=12, alpha=0.55, color=color, edgecolors="none")
    ax.set_xticks(positions)
    ax.set_xticklabels([PLOT_LABEL_NAMES[label] for label in labels])
    ax.set_xlabel("ClinVar label group")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)


def make_plots(metrics: pd.DataFrame, outdir: Path, dpi: int) -> None:
    plt = require_matplotlib()
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10})

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.0))
    box_and_points(axes[0], metrics, "q_l2", "WT-mutant query distance")
    axes[0].set_title("Projected TF-sequence query")
    box_and_points(axes[1], metrics, "attention_jsd", "WT-mutant key-weight divergence")
    axes[1].set_title("AlphaGenome-key weighting")
    box_and_points(axes[2], metrics, "mu_l2", "Retrieved representation distance")
    axes[2].set_title("Attention-weighted key vector")
    fig.suptitle("Do pathogenic missense labels show larger Hopfield retrieval changes?", y=1.04)
    fig.tight_layout()
    fig.savefig(outdir / "variant_retrieval_delta_by_class.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    overlap_columns = [column for column in metrics.columns if column.startswith("top") and column.endswith("_overlap_fraction")]
    if overlap_columns:
        fig, axes = plt.subplots(1, len(overlap_columns), figsize=(4.2 * len(overlap_columns), 4.0), squeeze=False)
        for ax, column in zip(axes.ravel(), overlap_columns):
            k = column.split("_", 1)[0].replace("top", "")
            box_and_points(ax, metrics, column, f"Top-{k} overlap fraction")
            ax.set_title(f"Top-{k} key overlap")
            ax.set_ylim(-0.03, 1.03)
        fig.suptitle("WT-mutant overlap among high-weight AlphaGenome keys", y=1.04)
        fig.tight_layout()
        fig.savefig(outdir / "variant_retrieval_top_overlap_by_class.png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    colors = metrics["label_class"].map({"benign": "#4C78A8", "deleterious": "#E45756"}).fillna("#777777")
    fig, ax = plt.subplots(figsize=(6.8, 5.0))
    ax.scatter(
        pd.to_numeric(metrics["esmc_l2_distance_from_wt"], errors="coerce"),
        pd.to_numeric(metrics["attention_jsd"], errors="coerce"),
        c=colors,
        s=20,
        alpha=0.7,
        edgecolors="none",
    )
    ax.set_xlabel("ESM-C distance from the WT isoform")
    ax.set_ylabel("WT-mutant AlphaGenome-key weighting divergence")
    ax.set_title("Retrieval change tracks the size of the sequence-embedding change")
    ax.grid(alpha=0.25)
    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#4C78A8", markersize=7, label=LABEL_NAMES["benign"]),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#E45756", markersize=7, label=LABEL_NAMES["deleterious"]),
    ]
    ax.legend(handles=legend_handles, frameon=False)
    fig.tight_layout()
    fig.savefig(outdir / "variant_retrieval_delta_vs_esmc_distance.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 5.0))
    ax.scatter(
        pd.to_numeric(metrics["percent_delta"], errors="coerce"),
        pd.to_numeric(metrics["attention_jsd"], errors="coerce"),
        c=colors,
        s=20,
        alpha=0.7,
        edgecolors="none",
    )
    ax.axvline(0, color="#333333", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("Predicted perturbed-cell activity change (%)")
    ax.set_ylabel("WT-mutant AlphaGenome-key weighting divergence")
    ax.set_title("Retrieval change versus predicted change in perturbed-cell state")
    ax.grid(alpha=0.25)
    ax.legend(handles=legend_handles, frameon=False)
    fig.tight_layout()
    fig.savefig(outdir / "variant_retrieval_delta_vs_predicted_state_shift.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def format_metric(summary: dict, label: str, column: str) -> str:
    value = summary["class_summary"].get(label, {}).get(column, {}).get("median")
    if value is None:
        return "NA"
    return f"{value:.4g}"


def write_report(path: Path, summary: dict) -> None:
    tests = summary["one_sided_tests"]
    lines = [
        "# Variant Retrieval-Change Report",
        "",
        "This analysis asks whether WT and mutant TF sequence representations produce different Hopfield query vectors and different weighting patterns over AlphaGenome keys.",
        "",
        f"It uses the frozen Hopfield projection checkpoint and {summary['panel_description']}.",
        "",
        "The retrieved representation in this analysis is the attention-weighted mean of L2-normalized AlphaGenome key vectors. It is a retrieval diagnostic, not a binding-site call.",
        "",
        "## Inputs",
        "",
        f"- Checkpoint: `{summary['checkpoint']}`",
        f"- AlphaGenome keys: `{summary['key_matrix']}`",
        f"- Variants analyzed: {summary['n_rows']}",
        f"- Genes analyzed: {summary['n_genes']}",
        f"- Checkpoint beta: {summary['checkpoint_beta']}",
        "",
        "## Class Summary",
        "",
        "| Class | Variants | Genes | Median ESM-C distance | Median query distance | Median key-weight divergence | Median retrieved-vector distance | Median top-100 overlap |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label in ["benign", "deleterious"]:
        label_summary = summary["class_summary"].get(label, {})
        lines.append(
            "| "
            + LABEL_NAMES[label]
            + f" | {label_summary.get('n', 0)}"
            + f" | {label_summary.get('genes', 0)}"
            + f" | {format_metric(summary, label, 'esmc_l2_distance_from_wt')}"
            + f" | {format_metric(summary, label, 'q_l2')}"
            + f" | {format_metric(summary, label, 'attention_jsd')}"
            + f" | {format_metric(summary, label, 'mu_l2')}"
            + f" | {format_metric(summary, label, 'top100_overlap_fraction')}"
            + " |"
        )
    lines.extend(
        [
            "",
            "## Tests",
            "",
            "| Question | One-sided p-value | AUROC for pathogenic/likely pathogenic label |",
            "|---|---:|---:|",
        ]
    )
    rows = [
        ("Projected TF-sequence query distance is larger for pathogenic labels", "q_l2_deleterious_greater_than_benign", "q_l2"),
        ("AlphaGenome-key weighting divergence is larger for pathogenic labels", "attention_jsd_deleterious_greater_than_benign", "attention_jsd"),
        ("Retrieved-vector distance is larger for pathogenic labels", "mu_l2_deleterious_greater_than_benign", "mu_l2"),
        ("Top-100 key overlap is lower for pathogenic labels", "top100_overlap_fraction_deleterious_less_than_benign", "negative_top100_overlap_fraction"),
    ]
    for question, test_key, auc_key in rows:
        p_value = tests.get(test_key, {}).get("p_value")
        auc = summary["auroc_for_deleterious"].get(auc_key)
        p_text = "NA" if p_value is None else f"{p_value:.3g}"
        auc_text = "NA" if auc is None else f"{auc:.3f}"
        lines.append(f"| {question} | {p_text} | {auc_text} |")

    rho = summary["spearman_correlations"]["attention_jsd_vs_esmc_l2_distance_from_wt"]
    rho_text = "NA" if rho["rho"] is None else f"{rho['rho']:.3f}"
    p_text = "NA" if rho["p_value"] is None else f"{rho['p_value']:.3g}"
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"Key-weight divergence correlates with ESM-C distance from WT (Spearman rho {rho_text}, p {p_text}).",
            "That means any clinical-label separation should be interpreted together with mutation size in the protein-embedding space.",
            "",
            "Use this analysis as evidence that sequence changes propagate through the Hopfield retrieval layer. Avoid claiming that the retrieved loci are direct TF binding sites.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    metrics, run_info = compute_metrics(args)
    metrics_path = args.outdir / "variant_retrieval_delta_metrics.csv"
    metrics.to_csv(metrics_path, index=False)

    summary = summarize(metrics, run_info, [int(value) for value in args.topk])
    summary["metrics_csv"] = str(metrics_path)
    summary_path = args.outdir / "variant_retrieval_delta_summary.json"
    write_json(summary_path, summary)

    make_plots(metrics, args.outdir, args.dpi)
    report_path = args.outdir / "variant_retrieval_delta_report.md"
    write_report(report_path, summary)
    print(f"wrote {metrics_path}")
    print(f"wrote {summary_path}")
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
