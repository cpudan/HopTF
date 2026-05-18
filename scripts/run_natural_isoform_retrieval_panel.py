#!/usr/bin/env python3
"""Analyze Hopfield retrieval differences between natural same-gene TF isoforms."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_variant_retrieval_delta import (
    auc_from_scores,
    finite_float,
    load_json,
    make_projector,
    mannwhitney,
    resolve_device,
    spearman,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--embedding-matrix", type=Path, required=True)
    parser.add_argument("--vocab", type=Path, required=True)
    parser.add_argument("--keys", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--min-cells", type=int, default=5)
    parser.add_argument("--topk", type=int, nargs="+", default=[100, 500, 1000])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def require_torch():
    import torch
    import torch.nn.functional as F

    return torch, F


def require_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def load_isoforms(args: argparse.Namespace) -> tuple[pd.DataFrame, np.ndarray]:
    metadata = pd.read_csv(args.metadata)
    vocab = load_json(args.vocab)
    vocab_index = {str(value): index for index, value in enumerate(vocab)}
    rows = metadata.loc[
        metadata["label_status"].astype(str).eq("responder")
        & (pd.to_numeric(metadata["n_cells"], errors="coerce").fillna(0) >= args.min_cells)
        & metadata["sequence"].notna()
    ].copy()
    rows["embedding_row"] = rows["isoform_embedding_id"].map(lambda value: vocab_index.get(str(value), -1))
    rows = rows.loc[rows["embedding_row"] >= 0].copy()
    counts = rows.groupby("gene_symbol")["isoform_embedding_id"].nunique()
    multi_gene = counts.loc[counts >= 2].index
    rows = rows.loc[rows["gene_symbol"].isin(multi_gene)].copy()
    rows = rows.sort_values(["gene_symbol", "isoform_id", "isoform_embedding_id"]).reset_index(drop=True)
    if rows.empty:
        raise ValueError("No same-gene responder isoform groups found")
    return rows, rows["embedding_row"].to_numpy(np.int64)


def compute_isoform_retrieval(args: argparse.Namespace) -> tuple[pd.DataFrame, dict]:
    torch, F = require_torch()
    device = resolve_device(args.device)
    args.outdir.mkdir(parents=True, exist_ok=True)
    rows, embedding_indices = load_isoforms(args)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    beta = float(checkpoint.get("beta", 1.0))
    model = make_projector(checkpoint, device)

    embeddings_np = np.load(args.embedding_matrix, mmap_mode="r")
    esm = np.asarray(embeddings_np[embedding_indices], dtype=np.float32)
    keys_np = np.asarray(np.load(args.keys, mmap_mode="r"), dtype=np.float32)
    keys = torch.as_tensor(keys_np, dtype=torch.float32, device=device)
    keys = F.normalize(keys, dim=-1)
    max_topk = max(args.topk)
    eps = 1.0e-12

    q_chunks = []
    attn_chunks = []
    mu_chunks = []
    entropy_chunks = []
    max_chunks = []
    top_index_chunks = []
    top_value_chunks = []
    with torch.no_grad():
        for start in range(0, len(rows), args.batch_size):
            stop = min(start + args.batch_size, len(rows))
            esm_t = torch.as_tensor(esm[start:stop], dtype=torch.float32, device=device)
            q = model(esm_t)
            logits = beta * (q @ keys.T)
            attn = logits.softmax(dim=-1)
            mu = attn @ keys
            top = torch.topk(attn, k=max_topk, dim=-1)
            entropy = -(attn * attn.clamp_min(eps).log()).sum(dim=-1)
            q_chunks.append(q.cpu().numpy())
            attn_chunks.append(attn.cpu().numpy())
            mu_chunks.append(mu.cpu().numpy())
            entropy_chunks.append(entropy.cpu().numpy())
            max_chunks.append(attn.max(dim=-1).values.cpu().numpy())
            top_index_chunks.append(top.indices.cpu().numpy())
            top_value_chunks.append(top.values.cpu().numpy())

    q_all = np.concatenate(q_chunks, axis=0)
    attn_all = np.concatenate(attn_chunks, axis=0)
    mu_all = np.concatenate(mu_chunks, axis=0)
    entropy_all = np.concatenate(entropy_chunks, axis=0)
    max_all = np.concatenate(max_chunks, axis=0)
    top_indices = np.concatenate(top_index_chunks, axis=0)
    top_values = np.concatenate(top_value_chunks, axis=0)

    row_records = []
    pair_records = []
    for idx, row in rows.iterrows():
        row_records.append(
            {
                "gene_symbol": row["gene_symbol"],
                "isoform_id": row["isoform_id"],
                "isoform_embedding_id": row["isoform_embedding_id"],
                "n_cells": int(row["n_cells"]),
                "response_score": finite_float(row["response_score"]),
                "protein_aa_length": int(row["protein_aa_length"]),
                "attention_entropy": finite_float(entropy_all[idx]),
                "attention_max": finite_float(max_all[idx]),
            }
        )

    for gene_symbol, group in rows.groupby("gene_symbol", sort=True):
        indices = group.index.to_list()
        for left, right in itertools.combinations(indices, 2):
            row_a = rows.loc[left]
            row_b = rows.loc[right]
            midpoint = 0.5 * (attn_all[left] + attn_all[right])
            jsd = 0.5 * (
                np.sum(attn_all[left] * (np.log(np.clip(attn_all[left], eps, None)) - np.log(np.clip(midpoint, eps, None))))
                + np.sum(attn_all[right] * (np.log(np.clip(attn_all[right], eps, None)) - np.log(np.clip(midpoint, eps, None))))
            )
            record = {
                "gene_symbol": gene_symbol,
                "isoform_a": row_a["isoform_id"],
                "isoform_b": row_b["isoform_id"],
                "isoform_embedding_id_a": row_a["isoform_embedding_id"],
                "isoform_embedding_id_b": row_b["isoform_embedding_id"],
                "n_cells_a": int(row_a["n_cells"]),
                "n_cells_b": int(row_b["n_cells"]),
                "min_n_cells": int(min(row_a["n_cells"], row_b["n_cells"])),
                "response_score_a": finite_float(row_a["response_score"]),
                "response_score_b": finite_float(row_b["response_score"]),
                "response_score_abs_diff": finite_float(abs(float(row_a["response_score"]) - float(row_b["response_score"]))),
                "protein_length_a": int(row_a["protein_aa_length"]),
                "protein_length_b": int(row_b["protein_aa_length"]),
                "protein_length_abs_diff": int(abs(int(row_a["protein_aa_length"]) - int(row_b["protein_aa_length"]))),
                "esmc_l2_distance": finite_float(np.linalg.norm(esm[left] - esm[right])),
                "q_l2": finite_float(np.linalg.norm(q_all[left] - q_all[right])),
                "q_cosine": finite_float(float(np.sum(q_all[left] * q_all[right]))),
                "attention_jsd": finite_float(jsd),
                "attention_l1": finite_float(np.abs(attn_all[left] - attn_all[right]).sum()),
                "mu_l2": finite_float(np.linalg.norm(mu_all[left] - mu_all[right])),
                "mu_cosine": finite_float(float(np.dot(mu_all[left], mu_all[right]) / (np.linalg.norm(mu_all[left]) * np.linalg.norm(mu_all[right]) + eps))),
                "attention_entropy_abs_diff": finite_float(abs(entropy_all[left] - entropy_all[right])),
                "attention_max_abs_diff": finite_float(abs(max_all[left] - max_all[right])),
            }
            for k in args.topk:
                left_set = set(top_indices[left, :k].tolist())
                right_set = set(top_indices[right, :k].tolist())
                record[f"top{k}_overlap_fraction"] = len(left_set & right_set) / float(k)
                record[f"top{k}_mass_a"] = finite_float(np.sum(top_values[left, :k]))
                record[f"top{k}_mass_b"] = finite_float(np.sum(top_values[right, :k]))
                record[f"top{k}_mass_abs_diff"] = finite_float(abs(record[f"top{k}_mass_a"] - record[f"top{k}_mass_b"]))
            pair_records.append(record)

    pair_df = pd.DataFrame(pair_records)
    if pair_df.empty:
        raise ValueError("No same-gene isoform pairs were created")
    q33, q67 = pair_df["response_score_abs_diff"].quantile([1 / 3, 2 / 3]).to_list()
    pair_df["observed_response_difference_group"] = np.select(
        [
            pair_df["response_score_abs_diff"] <= q33,
            pair_df["response_score_abs_diff"] >= q67,
        ],
        ["lower third", "upper third"],
        default="middle third",
    )
    isoform_df = pd.DataFrame(row_records)
    run_info = {
        "status": "ok",
        "n_isoforms": int(len(rows)),
        "n_genes": int(rows["gene_symbol"].nunique()),
        "n_pairs": int(len(pair_df)),
        "min_cells": int(args.min_cells),
        "checkpoint": str(args.checkpoint),
        "checkpoint_beta": beta,
        "n_keys": int(keys_np.shape[0]),
        "key_dim": int(keys_np.shape[1]),
        "topk": [int(value) for value in args.topk],
        "response_difference_quantiles": {"q33": finite_float(q33), "q67": finite_float(q67)},
        "retrieved_representation": "attention-weighted mean of L2-normalized AlphaGenome key vectors",
    }
    return isoform_df, pair_df, run_info


def summarize(pair_df: pd.DataFrame, run_info: dict) -> dict:
    summary = dict(run_info)
    metrics = ["esmc_l2_distance", "q_l2", "attention_jsd", "mu_l2", "top100_overlap_fraction"]
    by_group = {}
    for group, data in pair_df.groupby("observed_response_difference_group"):
        by_group[group] = {
            "n_pairs": int(len(data)),
            "n_genes": int(data["gene_symbol"].nunique()),
            **{
                metric: {
                    "median": finite_float(pd.to_numeric(data[metric], errors="coerce").median()),
                    "mean": finite_float(pd.to_numeric(data[metric], errors="coerce").mean()),
                }
                for metric in metrics
            },
        }
    summary["by_observed_response_difference_group"] = by_group
    summary["spearman_correlations"] = {}
    for metric in ["esmc_l2_distance", "q_l2", "attention_jsd", "mu_l2", "protein_length_abs_diff"]:
        summary["spearman_correlations"][f"{metric}_vs_response_score_abs_diff"] = spearman(
            pd.to_numeric(pair_df[metric], errors="coerce").to_numpy(float),
            pd.to_numeric(pair_df["response_score_abs_diff"], errors="coerce").to_numpy(float),
        )
    low = pair_df.loc[pair_df["observed_response_difference_group"].eq("lower third")]
    high = pair_df.loc[pair_df["observed_response_difference_group"].eq("upper third")]
    summary["upper_vs_lower_response_difference_tests"] = {}
    for metric in ["esmc_l2_distance", "q_l2", "attention_jsd", "mu_l2"]:
        summary["upper_vs_lower_response_difference_tests"][metric] = mannwhitney(
            pd.to_numeric(high[metric], errors="coerce").to_numpy(float),
            pd.to_numeric(low[metric], errors="coerce").to_numpy(float),
            alternative="greater",
        )
    summary["upper_response_difference_auroc"] = {}
    y = pair_df["observed_response_difference_group"].eq("upper third").to_numpy(int)
    for metric in ["esmc_l2_distance", "q_l2", "attention_jsd", "mu_l2"]:
        summary["upper_response_difference_auroc"][metric] = finite_float(
            auc_from_scores(y, pd.to_numeric(pair_df[metric], errors="coerce").to_numpy(float))
        )
    return summary


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def box_points(ax, data: pd.DataFrame, metric: str, ylabel: str) -> None:
    order = ["lower third", "middle third", "upper third"]
    colors = ["#4C78A8", "#B279A2", "#E45756"]
    values = [pd.to_numeric(data.loc[data["observed_response_difference_group"].eq(group), metric], errors="coerce").dropna() for group in order]
    positions = np.arange(len(order), dtype=float)
    ax.boxplot(values, positions=positions, widths=0.55, showfliers=False)
    rng = np.random.default_rng(23)
    for position, series, color in zip(positions, values, colors):
        ax.scatter(np.full(len(series), position) + rng.normal(0, 0.04, len(series)), series, s=14, alpha=0.55, color=color, edgecolors="none")
    ax.set_xticks(positions)
    ax.set_xticklabels(["Lower", "Middle", "Upper"])
    ax.set_xlabel("Observed response-score difference")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)


def make_plots(pair_df: pd.DataFrame, outdir: Path, dpi: int) -> None:
    plt = require_matplotlib()
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10})
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.0))
    box_points(axes[0], pair_df, "q_l2", "Isoform-pair query distance")
    axes[0].set_title("Projected TF-sequence query")
    box_points(axes[1], pair_df, "attention_jsd", "Isoform-pair key-weight divergence")
    axes[1].set_title("AlphaGenome-key weighting")
    box_points(axes[2], pair_df, "mu_l2", "Isoform-pair key-vector distance")
    axes[2].set_title("Attention-weighted key vector")
    fig.suptitle("Natural same-gene isoform pairs grouped by observed response-score difference", y=1.04)
    fig.tight_layout()
    fig.savefig(outdir / "natural_isoform_retrieval_by_response_difference.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    colors = pair_df["observed_response_difference_group"].map({"lower third": "#4C78A8", "middle third": "#B279A2", "upper third": "#E45756"}).fillna("#777777")
    fig, ax = plt.subplots(figsize=(6.8, 5.0))
    ax.scatter(pair_df["response_score_abs_diff"], pair_df["attention_jsd"], c=colors, s=20, alpha=0.7, edgecolors="none")
    ax.set_xlabel("Absolute observed response-score difference")
    ax.set_ylabel("AlphaGenome-key weighting divergence")
    ax.set_title("Natural isoform retrieval change versus observed response-score difference")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "natural_isoform_retrieval_vs_response_difference.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 5.0))
    ax.scatter(pair_df["esmc_l2_distance"], pair_df["attention_jsd"], c=colors, s=20, alpha=0.7, edgecolors="none")
    ax.set_xlabel("ESM-C distance between same-gene isoforms")
    ax.set_ylabel("AlphaGenome-key weighting divergence")
    ax.set_title("Natural isoform retrieval change versus ESM-C distance")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "natural_isoform_retrieval_vs_esmc_distance.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def write_report(path: Path, summary: dict) -> None:
    lines = [
        "# Natural Same-Gene Isoform Retrieval Report",
        "",
        "This analysis compares naturally occurring responder isoforms from the same TF gene.",
        "It asks whether larger natural isoform differences in sequence and Hopfield retrieval tend to align with larger observed response-score differences.",
        "",
        "## Inputs",
        "",
        f"- Isoforms: {summary['n_isoforms']}",
        f"- Genes: {summary['n_genes']}",
        f"- Same-gene isoform pairs: {summary['n_pairs']}",
        f"- Checkpoint beta: {summary['checkpoint_beta']}",
        "",
        "## Group Summary",
        "",
        "| Observed response-score difference group | Pairs | Genes | Median ESM-C distance | Median query distance | Median key-weight divergence | Median key-vector distance | Median top-100 overlap |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group in ["lower third", "middle third", "upper third"]:
        values = summary["by_observed_response_difference_group"].get(group, {})
        lines.append(
            f"| {group} | {values.get('n_pairs', 0)} | {values.get('n_genes', 0)} | "
            f"{values.get('esmc_l2_distance', {}).get('median', 'NA'):.4g} | "
            f"{values.get('q_l2', {}).get('median', 'NA'):.4g} | "
            f"{values.get('attention_jsd', {}).get('median', 'NA'):.4g} | "
            f"{values.get('mu_l2', {}).get('median', 'NA'):.4g} | "
            f"{values.get('top100_overlap_fraction', {}).get('median', 'NA'):.4g} |"
        )
    rho = summary["spearman_correlations"]["attention_jsd_vs_response_score_abs_diff"]
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"AlphaGenome-key weighting divergence versus observed response-score difference: Spearman rho {rho['rho']:.3f}, p {rho['p_value']:.3g}.",
            "This is an observational same-gene isoform check. It does not prove domain mechanism, because explicit domain annotations were not used.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    isoform_df, pair_df, run_info = compute_isoform_retrieval(args)
    isoform_df.to_csv(args.outdir / "natural_isoform_retrieval_isoforms.csv", index=False)
    pair_df.to_csv(args.outdir / "natural_isoform_retrieval_pairs.csv", index=False)
    summary = summarize(pair_df, run_info)
    write_json(args.outdir / "natural_isoform_retrieval_summary.json", summary)
    make_plots(pair_df, args.outdir, args.dpi)
    write_report(args.outdir / "natural_isoform_retrieval_report.md", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
