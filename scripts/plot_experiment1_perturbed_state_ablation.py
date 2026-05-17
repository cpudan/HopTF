#!/usr/bin/env python3
"""Create paper-facing plots and compact tables for HopTF Experiment 1."""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def clean_label(value: str) -> str:
    labels = {
        "artifacts": "protein length + ORF length + recovered cell count",
        "artifacts_plus_hopfield_query": "protein length + ORF length + recovered cell count + ESM-C projected into AlphaGenome key space",
        "artifacts_plus_hopfield_query_from_esm_dbp": "protein length + ORF length + recovered cell count + ESM-DBP projected into AlphaGenome key space",
        "artifacts_plus_esm_c": "protein length + ORF length + recovered cell count + raw ESM-C",
        "artifacts_plus_esm_dbp": "protein length + ORF length + recovered cell count + raw ESM-DBP",
        "artifacts_plus_shuffled_esm_c": "protein length + ORF length + recovered cell count + shuffled ESM-C",
        "artifacts_plus_shuffled_esm_dbp": "protein length + ORF length + recovered cell count + shuffled ESM-DBP",
        "hopfield_query": "ESM-C projected into AlphaGenome key space only",
        "hopfield_query_from_esm_dbp": "ESM-DBP projected into AlphaGenome key space only",
        "esm_c": "raw ESM-C only",
        "esm_dbp": "raw ESM-DBP only",
        "esm_c_shuffled_features": "shuffled ESM-C only",
        "esm_dbp_shuffled_features": "shuffled ESM-DBP only",
        "esm_c_shuffled_endpoint_labels": "ESM-C trained on shuffled perturbed-cell states",
        "esm_dbp_shuffled_endpoint_labels": "ESM-DBP trained on shuffled perturbed-cell states",
        "control_mean": "training-set mean perturbed-cell state",
        "length_cell_orf": "protein length + ORF length + recovered cell count",
        "length_plus_cell_count": "protein length + recovered cell count",
        "length_only": "protein length only",
        "cell_count_only": "recovered cell count only",
        "overall_grouped": "all held-out genes",
        "length_matched": "within protein-length groups",
        "cell_count_matched": "within recovered-cell-count groups",
        "length_cell_matched": "within length and cell-count groups",
    }
    return labels.get(str(value), str(value).replace("_", " "))


def clean_encoder(value: str) -> str:
    return {"esm_c": "ESM-C run", "esm_dbp": "ESM-DBP run"}.get(str(value), str(value))


def clean_added_feature(value: str) -> str:
    labels = {
        "artifacts_plus_hopfield_query": "+ ESM-C projected into AlphaGenome key space",
        "artifacts_plus_hopfield_query_from_esm_dbp": "+ ESM-DBP projected into AlphaGenome key space",
        "artifacts_plus_esm_c": "+ raw ESM-C",
        "artifacts_plus_esm_dbp": "+ raw ESM-DBP",
        "artifacts_plus_shuffled_esm_c": "+ shuffled ESM-C",
        "artifacts_plus_shuffled_esm_dbp": "+ shuffled ESM-DBP",
    }
    return labels.get(str(value), clean_label(value))


def save_barplot(df: pd.DataFrame, out: Path, *, dpi: int, title: str, y: str = "mse") -> None:
    fig_height = max(5.2, 0.48 * max(len(df), 1))
    fig, ax = plt.subplots(figsize=(12.8, fig_height))
    labels = [textwrap.fill(clean_label(row.reported_feature_set), width=54) for row in df.itertuples()]
    palette = {
        "artifacts": "#5B6470",
        "artifacts_plus_esm_c": "#4C78A8",
        "artifacts_plus_hopfield_query": "#2F8F5B",
        "artifacts_plus_shuffled_esm_c": "#9CA3AF",
        "artifacts_plus_esm_dbp": "#F58518",
        "artifacts_plus_hopfield_query_from_esm_dbp": "#B279A2",
        "artifacts_plus_shuffled_esm_dbp": "#BAB0AC",
    }
    colors = [palette.get(str(row.reported_feature_set), "#4C78A8") for row in df.itertuples()]
    values = df[y].astype(float).to_numpy()
    y_pos = np.arange(len(df))
    ax.barh(y_pos, values, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("MSE for predicted perturbed-cell state")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    for idx, value in enumerate(values):
        ax.text(value + 0.08, idx, f"{value:.2f}", va="center", fontsize=8)
    ax.set_xlim(0, max(values) * 1.12)
    fig.subplots_adjust(left=0.43, right=0.98, top=0.90, bottom=0.12)
    fig.savefig(out, dpi=dpi, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def save_delta_plot(df: pd.DataFrame, out: Path, *, dpi: int) -> None:
    plot_df = df.loc[
        df["context"].isin(["overall_grouped", "length_matched", "cell_count_matched", "length_cell_matched"])
        & df["mse_pct_change_vs_artifacts"].notna()
    ].copy()
    keep_features = [
        "artifacts_plus_esm_c",
        "artifacts_plus_esm_dbp",
        "artifacts_plus_hopfield_query",
        "artifacts_plus_hopfield_query_from_esm_dbp",
        "artifacts_plus_shuffled_esm_c",
        "artifacts_plus_shuffled_esm_dbp",
    ]
    plot_df = plot_df.loc[plot_df["reported_feature_set"].isin(keep_features)].copy()
    if plot_df.empty:
        return
    contexts = ["overall_grouped", "length_matched", "cell_count_matched", "length_cell_matched"]
    features = [feature for feature in keep_features if feature in set(plot_df["reported_feature_set"])]
    n_rows = len(contexts)
    fig, axes = plt.subplots(
        n_rows,
        1,
        figsize=(13.2, max(8.0, 1.9 * n_rows)),
        sharex=True,
        constrained_layout=False,
    )
    if n_rows == 1:
        axes = [axes]
    palette = {
        "artifacts_plus_esm_c": "#4C78A8",
        "artifacts_plus_hopfield_query": "#2F8F5B",
        "artifacts_plus_shuffled_esm_c": "#9CA3AF",
        "artifacts_plus_esm_dbp": "#F58518",
        "artifacts_plus_hopfield_query_from_esm_dbp": "#B279A2",
        "artifacts_plus_shuffled_esm_dbp": "#BAB0AC",
    }
    for ax, context in zip(axes, contexts):
        rows = []
        for feature in features:
            current = plot_df.loc[
                (plot_df["context"] == context) & (plot_df["reported_feature_set"] == feature),
                "mse_pct_change_vs_artifacts",
            ]
            if current.empty:
                continue
            rows.append((feature, float(current.iloc[0])))
        rows.sort(key=lambda item: item[1])
        labels = [textwrap.fill(clean_added_feature(feature), width=44) for feature, _ in rows]
        values = [value for _, value in rows]
        colors = [palette.get(feature, "#4C78A8") for feature, _ in rows]
        y_pos = np.arange(len(rows))
        ax.barh(y_pos, values, color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.axvline(0, color="black", linewidth=1.0)
        ax.grid(axis="x", alpha=0.25)
        ax.set_title(clean_label(context), loc="left", fontsize=10)
        for idx, value in enumerate(values):
            if value >= 0:
                ax.text(value + 1.2, idx, f"{value:+.1f}%", va="center", ha="left", fontsize=8)
            else:
                ax.text(0.8, idx, f"{value:+.1f}%", va="center", ha="left", fontsize=8)
    axes[-1].set_xlabel("% MSE change vs using only protein length + ORF length + recovered cell count")
    fig.suptitle(
        "Does TF sequence help predict perturbed-cell state after\n"
        "protein length, ORF length, and recovered cell count?",
        y=0.99,
        fontsize=15,
    )
    fig.subplots_adjust(left=0.45, right=0.98, top=0.84, bottom=0.08, hspace=0.70)
    fig.savefig(out, dpi=dpi, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def write_latex_table(df: pd.DataFrame, out: Path) -> None:
    cols = [
        "encoder",
        "context",
        "reported_feature_set",
        "mse",
        "median_row_mse",
        "mean_cosine_similarity",
        "mse_pct_change_vs_artifacts",
    ]
    present = [col for col in cols if col in df.columns]
    table = df[present].copy()
    if "encoder" in table.columns:
        table["encoder"] = table["encoder"].map(clean_encoder)
    if "context" in table.columns:
        table["context"] = table["context"].map(clean_label)
    if "reported_feature_set" in table.columns:
        table["reported_feature_set"] = table["reported_feature_set"].map(clean_label)
    for col in ["mse", "median_row_mse", "mean_cosine_similarity", "mse_pct_change_vs_artifacts"]:
        if col in table.columns:
            table[col] = table[col].map(lambda value: "" if pd.isna(value) else f"{float(value):.4f}")
    out.write_text(table.to_latex(index=False, escape=True), encoding="utf-8")


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir
    outdir = args.outdir or results_dir / "paper_assets"
    outdir.mkdir(parents=True, exist_ok=True)
    comparison_path = results_dir / "experiment1_feature_comparison.csv"
    if not comparison_path.exists():
        raise FileNotFoundError(comparison_path)
    df = pd.read_csv(comparison_path)

    controlled = df.loc[df["source_table"] == "controlled_artifact_endpoint"].copy()
    overall = controlled.loc[controlled["context"].astype(str) == "overall_grouped"].copy()
    if not overall.empty:
        overall = overall.sort_values(["encoder", "mse"]).reset_index(drop=True)
        overall_features = [
            "artifacts",
            "artifacts_plus_hopfield_query",
            "artifacts_plus_esm_c",
            "artifacts_plus_shuffled_esm_c",
            "artifacts_plus_hopfield_query_from_esm_dbp",
            "artifacts_plus_esm_dbp",
            "artifacts_plus_shuffled_esm_dbp",
        ]
        plot_overall = overall.loc[overall["reported_feature_set"].isin(overall_features)].copy()
        plot_overall = (
            plot_overall.sort_values(["mse", "reported_feature_set"])
            .drop_duplicates(subset=["reported_feature_set"], keep="first")
            .sort_values("mse")
            .reset_index(drop=True)
        )
        save_barplot(
            plot_overall,
            outdir / "experiment1_controlled_overall_mse.png",
            dpi=args.dpi,
            title="Does TF sequence help predict perturbed-cell state?",
        )
        write_latex_table(overall, outdir / "experiment1_controlled_overall_table.tex")

    save_delta_plot(controlled, outdir / "experiment1_sequence_after_length_cell_count.png", dpi=args.dpi)
    matched = controlled.loc[controlled["context"].astype(str) != "overall_grouped"].copy()
    if not matched.empty:
        write_latex_table(matched, outdir / "experiment1_matched_context_table.tex")

    broad = df.loc[df["source_table"] == "broad_grouped_endpoint"].copy()
    if not broad.empty:
        write_latex_table(broad.sort_values(["encoder", "mse"]), outdir / "experiment1_broad_perturbed_state_table.tex")

    payload = {
        "results_dir": str(results_dir),
        "assets_dir": str(outdir),
        "plots": sorted(path.name for path in outdir.glob("*.png")),
        "tables": sorted(path.name for path in outdir.glob("*.tex")),
    }
    (outdir / "experiment1_paper_assets_manifest.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
