#!/usr/bin/env python3
"""Compare real ClinVar variants with matched random missense controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


LABEL_NAMES = {
    "benign": "Benign / likely benign",
    "deleterious": "Pathogenic / likely pathogenic",
}
METRICS = {
    "q_l2": "Projected query distance",
    "attention_jsd": "AlphaGenome-key weighting divergence",
    "mu_l2": "Attention-weighted key-vector distance",
    "top100_change": "Top-100 key-list change",
    "abs_percent_delta": "Absolute predicted perturbed-cell activity change (%)",
}
SHORT_METRIC_LABELS = {
    "q_l2": "Query distance",
    "attention_jsd": "Key-weight divergence",
    "mu_l2": "Key-vector distance",
    "top100_change": "Top-100 key-list change",
    "abs_percent_delta": "Predicted activity change (%)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-metrics", type=Path, required=True)
    parser.add_argument("--random-metrics", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def finite_float(value):
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def require_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def paired_test(real: np.ndarray, random: np.ndarray) -> dict:
    mask = np.isfinite(real) & np.isfinite(random)
    real = real[mask]
    random = random[mask]
    diff = real - random
    out = {
        "n_pairs": int(len(diff)),
        "median_real": finite_float(np.median(real)) if len(real) else None,
        "median_random": finite_float(np.median(random)) if len(random) else None,
        "median_real_minus_random": finite_float(np.median(diff)) if len(diff) else None,
        "fraction_real_greater": finite_float(np.mean(diff > 0)) if len(diff) else None,
        "wilcoxon_p_real_greater": None,
        "sign_test_p_real_greater": None,
    }
    if len(diff) == 0:
        return out
    try:
        from scipy.stats import binomtest, wilcoxon

        nonzero = diff[diff != 0]
        if len(nonzero):
            out["wilcoxon_p_real_greater"] = finite_float(wilcoxon(nonzero, alternative="greater").pvalue)
            out["sign_test_p_real_greater"] = finite_float(
                binomtest(int((nonzero > 0).sum()), int(len(nonzero)), 0.5, alternative="greater").pvalue
            )
    except Exception:
        pass
    return out


def load_joined(real_path: Path, random_path: Path) -> pd.DataFrame:
    real = pd.read_csv(real_path)
    random = pd.read_csv(random_path)
    real = real.add_prefix("real_")
    random = random.add_prefix("random_")
    joined = random.merge(
        real,
        left_on="random_matched_real_mutant_embedding_id",
        right_on="real_mutant_embedding_id",
        how="inner",
        validate="one_to_one",
    )
    if joined.empty:
        raise ValueError("no matched real/random rows after merge")
    joined["label_class"] = joined["random_label_class"].astype(str)
    for prefix in ["real", "random"]:
        joined[f"{prefix}_top100_change"] = 1.0 - pd.to_numeric(
            joined[f"{prefix}_top100_overlap_fraction"], errors="coerce"
        )
        joined[f"{prefix}_abs_percent_delta"] = pd.to_numeric(
            joined[f"{prefix}_percent_delta"], errors="coerce"
        ).abs()
    for metric in METRICS:
        joined[f"{metric}_real_minus_random"] = (
            pd.to_numeric(joined[f"real_{metric}"], errors="coerce")
            - pd.to_numeric(joined[f"random_{metric}"], errors="coerce")
        )
    return joined


def summarize(joined: pd.DataFrame) -> dict:
    summary = {
        "status": "ok",
        "n_pairs": int(len(joined)),
        "counts_by_matched_label_class": {
            str(k): int(v) for k, v in joined["label_class"].value_counts().sort_index().items()
        },
        "genes_by_matched_label_class": {
            str(k): int(v) for k, v in joined.groupby("label_class")["random_gene_symbol"].nunique().sort_index().items()
        },
        "metrics": {},
    }
    for label in ["benign", "deleterious"]:
        subset = joined.loc[joined["label_class"].eq(label)].copy()
        label_summary = {}
        for metric in METRICS:
            label_summary[metric] = paired_test(
                pd.to_numeric(subset[f"real_{metric}"], errors="coerce").to_numpy(float),
                pd.to_numeric(subset[f"random_{metric}"], errors="coerce").to_numpy(float),
            )
        summary["metrics"][label] = label_summary

    overall = {}
    for metric in METRICS:
        overall[metric] = paired_test(
            pd.to_numeric(joined[f"real_{metric}"], errors="coerce").to_numpy(float),
            pd.to_numeric(joined[f"random_{metric}"], errors="coerce").to_numpy(float),
        )
    summary["metrics"]["overall"] = overall
    return summary


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def boxplot_metric(ax, joined: pd.DataFrame, metric: str, label: str, ylabel: str) -> None:
    subset = joined.loc[joined["label_class"].eq(label)]
    values = [
        pd.to_numeric(subset[f"real_{metric}"], errors="coerce").dropna(),
        pd.to_numeric(subset[f"random_{metric}"], errors="coerce").dropna(),
    ]
    ax.boxplot(values, positions=[0, 1], widths=0.5, showfliers=False)
    rng = np.random.default_rng(11)
    colors = ["#E45756", "#72B7B2"]
    for pos, series, color in zip([0, 1], values, colors):
        ax.scatter(
            np.full(len(series), pos) + rng.normal(0, 0.045, size=len(series)),
            series,
            s=12,
            color=color,
            alpha=0.55,
            edgecolors="none",
        )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Real", "Matched random"])
    ax.set_title(LABEL_NAMES[label])
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)


def make_plots(joined: pd.DataFrame, outdir: Path, dpi: int) -> None:
    plt = require_matplotlib()
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10})

    fig, axes = plt.subplots(2, 3, figsize=(12.0, 7.0))
    plot_specs = [
        ("q_l2", "Projected query distance"),
        ("attention_jsd", "AlphaGenome-key weighting divergence"),
        ("mu_l2", "Attention-weighted key-vector distance"),
    ]
    for row_idx, label in enumerate(["benign", "deleterious"]):
        for col_idx, (metric, ylabel) in enumerate(plot_specs):
            boxplot_metric(axes[row_idx, col_idx], joined, metric, label, ylabel)
    fig.suptitle("Real ClinVar variants versus matched random substitutions", y=1.02)
    fig.tight_layout()
    fig.savefig(outdir / "random_control_real_vs_random_retrieval_change.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.0))
    for ax, metric in zip(axes, ["q_l2", "attention_jsd", "mu_l2"]):
        for label, color, offset in [("benign", "#4C78A8", -0.08), ("deleterious", "#E45756", 0.08)]:
            values = pd.to_numeric(
                joined.loc[joined["label_class"].eq(label), f"{metric}_real_minus_random"], errors="coerce"
            ).dropna()
            rng = np.random.default_rng(19)
            x = np.full(len(values), 0 if label == "benign" else 1, dtype=float) + rng.normal(offset, 0.035, len(values))
            ax.scatter(x, values, s=14, color=color, alpha=0.55, edgecolors="none")
            ax.hlines(values.median(), (0 if label == "benign" else 1) - 0.25, (0 if label == "benign" else 1) + 0.25, color="black", linewidth=2)
        ax.axhline(0, color="#333333", linewidth=0.8)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Benign", "Pathogenic"])
        ax.set_ylabel(f"Real - random\n{SHORT_METRIC_LABELS[metric]}")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Positive values mean the real variant changed retrieval more than its matched random control", y=1.02)
    fig.tight_layout()
    fig.savefig(outdir / "random_control_real_minus_random_by_class.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.0))
    for ax, label in zip(axes, ["benign", "deleterious"]):
        boxplot_metric(axes[0 if label == "benign" else 1], joined, "abs_percent_delta", label, "Absolute predicted perturbed-cell activity change (%)")
    fig.suptitle("Predicted perturbed-cell activity change: real variants versus matched random controls", y=1.04)
    fig.tight_layout()
    fig.savefig(outdir / "random_control_predicted_activity_change.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def format_value(value) -> str:
    if value is None:
        return "NA"
    value = float(value)
    if abs(value) < 0.001 and value != 0:
        return f"{value:.2e}"
    return f"{value:.4g}"


def write_report(path: Path, summary: dict) -> None:
    lines = [
        "# Matched Random-Substitution Control Report",
        "",
        "This analysis compares each real ClinVar missense variant to one same-isoform random missense substitution.",
        "The random substitution is matched to the same TF isoform and approximately the same sequence-length bin, and it uses the same broad amino-acid class as the real mutant residue when possible.",
        "",
        "## Summary",
        "",
        f"- Matched pairs: {summary['n_pairs']}",
        f"- Counts by matched ClinVar label: {summary['counts_by_matched_label_class']}",
        "",
        "| Matched label | Measure | Median real | Median random | Median real minus random | Fraction real > random | Wilcoxon p |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for label in ["benign", "deleterious"]:
        for metric, name in METRICS.items():
            values = summary["metrics"][label][metric]
            lines.append(
                f"| {LABEL_NAMES[label]} | {name} | {format_value(values['median_real'])} | "
                f"{format_value(values['median_random'])} | {format_value(values['median_real_minus_random'])} | "
                f"{format_value(values['fraction_real_greater'])} | {format_value(values['wilcoxon_p_real_greater'])} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Use this as a same-isoform negative control. If real pathogenic variants exceed matched random substitutions, that supports a variant-specific retrieval effect beyond arbitrary same-protein missense noise.",
            "If matched random substitutions are similar or larger, the safer conclusion is that HopTF is sensitive to sequence perturbation size, but this control does not isolate clinical pathogenicity.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    joined = load_joined(args.real_metrics, args.random_metrics)
    joined_path = args.outdir / "random_control_real_vs_random_joined.csv"
    joined.to_csv(joined_path, index=False)
    summary = summarize(joined)
    summary["joined_csv"] = str(joined_path)
    write_json(args.outdir / "random_control_summary.json", summary)
    make_plots(joined, args.outdir, args.dpi)
    write_report(args.outdir / "random_control_report.md", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
