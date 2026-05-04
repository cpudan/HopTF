#!/usr/bin/env python3
"""Run a leave-one-isoform OT-CFM panel across mixed-response TF groups."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[0]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from hopfield_fitting_common import DEFAULT_ESM, DEFAULT_METADATA, DEFAULT_OUTDIR, DEFAULT_VOCAB, ensure_dir, write_json

DEFAULT_PANEL_GENES = ["NFATC1", "ZNF195", "IKZF3", "TP73", "MIER1", "SOX5", "ZNF534"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--embedding-matrix", type=Path, default=DEFAULT_ESM)
    parser.add_argument("--vocab", type=Path, default=DEFAULT_VOCAB)
    parser.add_argument("--latents", type=Path, default=DEFAULT_OUTDIR / "perturbation_latents.npz")
    parser.add_argument("--outdir", type=Path, default=Path("tmp/hopfield_fitting_leave_one_panel"))
    parser.add_argument("--gene", action="append", default=None, help="Gene to include. Repeatable.")
    parser.add_argument("--min-cells", type=int, default=5)
    parser.add_argument("--max-per-label", type=int, default=1)
    parser.add_argument("--max-train-rows", type=int, default=256)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-seeds", type=int, default=1)
    parser.add_argument("--aggregate-pass-rate", type=float, default=0.8)
    parser.add_argument("--nonresponder-response-ratio-threshold", type=float, default=1.5)
    parser.add_argument("--endpoint-loss-weight", type=float, default=0.0)
    parser.add_argument("--response-amplitude-loss-weight", type=float, default=0.0)
    parser.add_argument("--endpoint-loss-steps", type=int, default=8)
    parser.add_argument("--endpoint-loss-interval", type=int, default=10)
    parser.add_argument("--python", default=sys.executable)
    return parser.parse_args()


def select_panel_holdouts(
    metadata: pd.DataFrame,
    gene: str,
    *,
    min_cells: int,
    max_per_label: int,
) -> pd.DataFrame:
    rows = metadata.loc[metadata["gene_symbol"].astype(str) == str(gene)].copy()
    rows["n_cells_numeric"] = pd.to_numeric(rows["n_cells"], errors="coerce")
    rows["response_score_numeric"] = pd.to_numeric(rows["response_score"], errors="coerce")
    rows = rows.loc[rows["n_cells_numeric"].fillna(0) >= int(min_cells)]
    selected = []
    responders = rows.loc[rows["label_status"].astype(str) == "responder"].sort_values(
        "response_score_numeric",
        ascending=False,
    )
    nonresponders = rows.loc[rows["label_status"].astype(str) == "nonresponder"].sort_values(
        "response_score_numeric",
        ascending=True,
    )
    if max_per_label > 0:
        selected.extend(responders.head(int(max_per_label)).index.tolist())
        selected.extend(nonresponders.head(int(max_per_label)).index.tolist())
    return rows.loc[selected].copy().reset_index(drop=True)


def sibling_label_support(metadata: pd.DataFrame, holdouts: pd.DataFrame) -> dict[str, dict[str, int]]:
    support = {}
    for _, row in holdouts.iterrows():
        gene = str(row["gene_symbol"])
        isoform_id = str(row["isoform_id"])
        siblings = metadata.loc[
            (metadata["gene_symbol"].astype(str) == gene)
            & (metadata["isoform_id"].astype(str) != isoform_id)
        ]
        labels = siblings["label_status"].astype(str)
        support[isoform_id] = {
            "sibling_responders": int((labels == "responder").sum()),
            "sibling_nonresponders": int((labels == "nonresponder").sum()),
            "sibling_ambiguous": int((labels == "ambiguous").sum()),
        }
    return support


def label_aware_pass(row: pd.Series | dict[str, Any], *, nonresponder_response_ratio_threshold: float) -> bool:
    label = str(row.get("label_status", ""))
    endpoint_fraction = pd.to_numeric(row.get("control_standardized_endpoint_mse_fraction_of_baseline"), errors="coerce")
    response_ratio = pd.to_numeric(row.get("mean_control_standardized_response_l2_ratio"), errors="coerce")
    if label == "nonresponder":
        return bool(np.isfinite(response_ratio) and response_ratio <= float(nonresponder_response_ratio_threshold))
    if label == "responder":
        return bool(np.isfinite(endpoint_fraction) and endpoint_fraction < 1.0)
    return bool(row.get("passed", False))


def failure_category(row: pd.Series | dict[str, Any], *, aggregate_pass_rate: float, nonresponder_response_ratio_threshold: float) -> str:
    if bool(row.get("stable_label_aware_pass", False)):
        return "stable_pass"
    pass_rate = pd.to_numeric(row.get("label_aware_pass_rate"), errors="coerce")
    label = str(row.get("label_status", ""))
    response_ratio = pd.to_numeric(row.get("mean_control_standardized_response_l2_ratio_mean"), errors="coerce")
    endpoint_fraction = pd.to_numeric(
        row.get("control_standardized_endpoint_mse_fraction_of_baseline_mean"),
        errors="coerce",
    )
    sibling_nonresponders = int(row.get("sibling_nonresponders", 0) or 0)
    sibling_responders = int(row.get("sibling_responders", 0) or 0)
    if np.isfinite(pass_rate) and pass_rate > 0.0:
        return "seed_sensitive"
    if label == "nonresponder" and sibling_nonresponders == 0:
        return "unsupported_nonresponder_sibling"
    if label == "responder" and sibling_responders == 0:
        return "unsupported_responder_sibling"
    if label == "nonresponder" and np.isfinite(response_ratio) and response_ratio > float(nonresponder_response_ratio_threshold):
        return "overtransported_nonresponder"
    if label == "responder" and np.isfinite(endpoint_fraction) and endpoint_fraction >= 1.0:
        return "responder_endpoint_failure"
    if np.isfinite(pass_rate) and pass_rate < float(aggregate_pass_rate):
        return "below_stability_threshold"
    return "hard_failure"


def run_gene(args: argparse.Namespace, gene: str, holdouts: pd.DataFrame, *, seed_index: int, seed: int) -> dict[str, Any]:
    gene_outdir = ensure_dir(args.outdir / f"seed_{seed:03d}" / gene)
    log_dir = ensure_dir(args.outdir / "logs")
    log_path = log_dir / f"{gene}_seed_{seed:03d}.log"
    ids = [str(value) for value in holdouts["isoform_id"]]
    command = [
        args.python,
        "scripts/train_otcfm_sequence_conditioned.py",
        "--mode",
        "leave-one-isoform",
        "--metadata",
        str(args.metadata),
        "--embedding-matrix",
        str(args.embedding_matrix),
        "--vocab",
        str(args.vocab),
        "--latents",
        str(args.latents),
        "--outdir",
        str(gene_outdir),
        "--holdout-gene",
        str(gene),
        "--max-holdouts",
        str(len(ids)),
        "--max-train-rows",
        str(args.max_train_rows),
        "--min-cells",
        str(args.min_cells),
        "--steps",
        str(args.steps),
        "--seed",
        str(seed),
        "--endpoint-loss-weight",
        str(args.endpoint_loss_weight),
        "--response-amplitude-loss-weight",
        str(args.response_amplitude_loss_weight),
        "--endpoint-loss-steps",
        str(args.endpoint_loss_steps),
        "--endpoint-loss-interval",
        str(args.endpoint_loss_interval),
    ]
    for isoform_id in ids:
        command.extend(["--holdout-isoform-id", isoform_id])

    started = time.time()
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(command) + "\n\n")
        handle.flush()
        proc = subprocess.run(command, cwd=REPO_ROOT, stdout=handle, stderr=subprocess.STDOUT, text=True)
    return {
        "gene": gene,
        "seed_index": int(seed_index),
        "seed": int(seed),
        "isoform_ids": ids,
        "returncode": int(proc.returncode),
        "seconds": float(time.time() - started),
        "log": str(log_path),
        "metrics_csv": str(gene_outdir / f"otcfm_leave_one_{gene}_metrics.csv"),
        "summary_json": str(gene_outdir / f"otcfm_leave_one_{gene}_summary.json"),
    }


def summarize_by_isoform(
    metrics: pd.DataFrame,
    *,
    aggregate_pass_rate: float,
    nonresponder_response_ratio_threshold: float,
) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    rows = []
    id_columns = ["gene_symbol", "isoform_id", "label_status"]
    optional_columns = ["n_cells", "response_score", "protein_aa_length"]
    for keys, group in metrics.groupby(id_columns, dropna=False):
        row = dict(zip(id_columns, keys, strict=True))
        for column in optional_columns + ["sibling_responders", "sibling_nonresponders", "sibling_ambiguous"]:
            if column in group.columns:
                row[column] = group[column].iloc[0]
        row.update(
            {
                "n_seed_runs": int(group.shape[0]),
                "endpoint_pass_rate": float(group["passed"].astype(bool).mean()) if "passed" in group else np.nan,
                "label_aware_pass_rate": (
                    float(group["label_aware_pass"].astype(bool).mean()) if "label_aware_pass" in group else np.nan
                ),
                "stable_label_aware_pass": (
                    bool(group["label_aware_pass"].astype(bool).mean() >= float(aggregate_pass_rate))
                    if "label_aware_pass" in group
                    else False
                ),
            }
        )
        for column in [
            "control_standardized_endpoint_mse_fraction_of_baseline",
            "mean_control_standardized_endpoint_l2_delta",
            "mean_control_standardized_response_l2_ratio",
            "mean_predicted_control_standardized_response_l2",
            "mean_observed_control_standardized_response_l2",
        ]:
            if column in group.columns:
                row[f"{column}_mean"] = float(pd.to_numeric(group[column], errors="coerce").mean())
                row[f"{column}_sd"] = float(pd.to_numeric(group[column], errors="coerce").std(ddof=0))
        row["failure_category"] = failure_category(
            row,
            aggregate_pass_rate=float(aggregate_pass_rate),
            nonresponder_response_ratio_threshold=float(nonresponder_response_ratio_threshold),
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["stable_label_aware_pass", "label_aware_pass_rate"], ascending=[True, True])


def compact_markdown_table(df: pd.DataFrame, columns: list[str]) -> list[str]:
    if df.empty:
        return ["No rows."]
    out = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in df.iterrows():
        values = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                value = f"{value:.4g}"
            values.append(str(value))
        out.append("| " + " | ".join(values) + " |")
    return out


def main() -> None:
    args = parse_args()
    outdir = ensure_dir(args.outdir)
    genes = args.gene or DEFAULT_PANEL_GENES
    metadata = pd.read_csv(args.metadata)

    target_rows = []
    run_rows = []
    metric_frames = []
    holdouts_by_gene = {}
    support_by_isoform = {}
    for gene in genes:
        holdouts = select_panel_holdouts(
            metadata,
            gene,
            min_cells=int(args.min_cells),
            max_per_label=int(args.max_per_label),
        )
        if holdouts.empty:
            run_rows.append({"gene": gene, "returncode": None, "seconds": 0.0, "log": None, "metrics_csv": None, "summary_json": None})
            continue
        support_by_isoform.update(sibling_label_support(metadata, holdouts))
        for isoform_id, support in support_by_isoform.items():
            mask = holdouts["isoform_id"].astype(str) == str(isoform_id)
            for key, value in support.items():
                holdouts.loc[mask, key] = int(value)
        holdouts_by_gene[gene] = holdouts
        keep_cols = [
            "gene_symbol",
            "isoform_id",
            "label_status",
            "n_cells",
            "response_score",
            "protein_aa_length",
            "sibling_responders",
            "sibling_nonresponders",
            "sibling_ambiguous",
        ]
        target_rows.extend(holdouts.loc[:, [column for column in keep_cols if column in holdouts.columns]].to_dict(orient="records"))
    for seed_index in range(int(args.n_seeds)):
        seed = int(args.seed) + seed_index
        for gene, holdouts in holdouts_by_gene.items():
            run = run_gene(args, gene, holdouts, seed_index=seed_index, seed=seed)
            run_rows.append(run)
            metrics_path = Path(str(run["metrics_csv"]))
            if metrics_path.exists():
                metrics = pd.read_csv(metrics_path)
                metrics["panel_gene"] = gene
                metrics["panel_seed_index"] = int(seed_index)
                metrics["panel_seed"] = int(seed)
                metrics["panel_returncode"] = run["returncode"]
                metrics["panel_log"] = run["log"]
                for isoform_id, support in support_by_isoform.items():
                    mask = metrics["isoform_id"].astype(str) == str(isoform_id)
                    for key, value in support.items():
                        metrics.loc[mask, key] = int(value)
                metrics["label_aware_pass"] = metrics.apply(
                    label_aware_pass,
                    axis=1,
                    nonresponder_response_ratio_threshold=float(args.nonresponder_response_ratio_threshold),
                )
                metric_frames.append(metrics)

    targets = pd.DataFrame(target_rows)
    runs = pd.DataFrame(run_rows)
    metrics = pd.concat(metric_frames, ignore_index=True) if metric_frames else pd.DataFrame()
    aggregate = summarize_by_isoform(
        metrics,
        aggregate_pass_rate=float(args.aggregate_pass_rate),
        nonresponder_response_ratio_threshold=float(args.nonresponder_response_ratio_threshold),
    )
    targets_path = outdir / "otcfm_leave_one_panel_targets.csv"
    runs_path = outdir / "otcfm_leave_one_panel_runs.csv"
    metrics_path = outdir / "otcfm_leave_one_panel_metrics.csv"
    aggregate_path = outdir / "otcfm_leave_one_panel_aggregate.csv"
    report_path = outdir / "otcfm_leave_one_panel.md"
    targets.to_csv(targets_path, index=False)
    runs.to_csv(runs_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    aggregate.to_csv(aggregate_path, index=False)

    summary = {
        "genes": genes,
        "min_cells": int(args.min_cells),
        "max_per_label": int(args.max_per_label),
        "steps": int(args.steps),
        "seed_start": int(args.seed),
        "n_seeds": int(args.n_seeds),
        "max_train_rows": int(args.max_train_rows),
        "aggregate_pass_rate": float(args.aggregate_pass_rate),
        "nonresponder_response_ratio_threshold": float(args.nonresponder_response_ratio_threshold),
        "endpoint_loss_weight": float(args.endpoint_loss_weight),
        "response_amplitude_loss_weight": float(args.response_amplitude_loss_weight),
        "endpoint_loss_steps": int(args.endpoint_loss_steps),
        "endpoint_loss_interval": int(args.endpoint_loss_interval),
        "n_genes": int(len(genes)),
        "n_genes_with_targets": int(len(holdouts_by_gene)),
        "n_holdouts": int(metrics.shape[0]),
        "n_passed": int(metrics["passed"].sum()) if "passed" in metrics else 0,
        "n_label_aware_passed": int(metrics["label_aware_pass"].sum()) if "label_aware_pass" in metrics else 0,
        "n_failed": int((~metrics["passed"].astype(bool)).sum()) if "passed" in metrics and not metrics.empty else 0,
        "n_label_aware_failed": (
            int((~metrics["label_aware_pass"].astype(bool)).sum()) if "label_aware_pass" in metrics and not metrics.empty else 0
        ),
        "n_isoforms": int(aggregate.shape[0]),
        "n_stable_label_aware_passed": (
            int(aggregate["stable_label_aware_pass"].sum()) if "stable_label_aware_pass" in aggregate else 0
        ),
        "failure_categories": (
            aggregate["failure_category"].value_counts().sort_index().to_dict()
            if "failure_category" in aggregate
            else {}
        ),
        "passed_all": bool(metrics["passed"].all()) if "passed" in metrics and not metrics.empty else False,
        "label_aware_passed_all": (
            bool(metrics["label_aware_pass"].all()) if "label_aware_pass" in metrics and not metrics.empty else False
        ),
        "stable_label_aware_passed_all": (
            bool(aggregate["stable_label_aware_pass"].all()) if "stable_label_aware_pass" in aggregate and not aggregate.empty else False
        ),
        "targets_csv": str(targets_path),
        "runs_csv": str(runs_path),
        "metrics_csv": str(metrics_path),
        "aggregate_csv": str(aggregate_path),
        "report_md": str(report_path),
    }
    write_json(outdir / "otcfm_leave_one_panel_summary.json", summary)

    lines = [
        "# OT-CFM Leave-One Isoform Panel",
        "",
        f"Status: {'PASS' if summary['stable_label_aware_passed_all'] else 'NEEDS DEBUG'}",
        f"Endpoint criterion: {summary['n_passed']} / {summary['n_holdouts']} seed-level holdouts passed",
        f"Label-aware criterion: {summary['n_label_aware_passed']} / {summary['n_holdouts']} seed-level holdouts passed",
        f"Stable label-aware isoforms: {summary['n_stable_label_aware_passed']} / {summary['n_isoforms']} at pass-rate >= {summary['aggregate_pass_rate']}",
        f"Endpoint loss weight: `{summary['endpoint_loss_weight']}`",
        f"Response-amplitude loss weight: `{summary['response_amplitude_loss_weight']}`",
        f"Genes: {', '.join(genes)}",
        "",
        "## Targets",
        "",
    ]
    lines.extend(
        compact_markdown_table(
            targets,
            [
                "gene_symbol",
                "isoform_id",
                "label_status",
                "n_cells",
                "response_score",
                "protein_aa_length",
                "sibling_responders",
                "sibling_nonresponders",
                "sibling_ambiguous",
            ],
        )
    )
    lines.extend(["", "## Aggregate", ""])
    aggregate_columns = [
        "gene_symbol",
        "isoform_id",
        "label_status",
        "n_cells",
        "response_score",
        "n_seed_runs",
        "endpoint_pass_rate",
        "label_aware_pass_rate",
        "stable_label_aware_pass",
        "failure_category",
        "sibling_responders",
        "sibling_nonresponders",
        "sibling_ambiguous",
        "control_standardized_endpoint_mse_fraction_of_baseline_mean",
        "mean_control_standardized_response_l2_ratio_mean",
    ]
    lines.extend(compact_markdown_table(aggregate, [column for column in aggregate_columns if column in aggregate.columns]))
    lines.extend(["", "## Seed-Level Metrics", ""])
    metric_columns = [
        "panel_seed",
        "gene_symbol",
        "isoform_id",
        "label_status",
        "n_cells",
        "response_score",
        "endpoint_mse_fraction_of_baseline",
        "control_standardized_endpoint_mse_fraction_of_baseline",
        "mean_control_standardized_endpoint_l2_delta",
        "mean_predicted_control_standardized_response_l2",
        "mean_observed_control_standardized_response_l2",
        "mean_control_standardized_response_l2_ratio",
        "passed",
        "label_aware_pass",
    ]
    lines.extend(compact_markdown_table(metrics, [column for column in metric_columns if column in metrics.columns]))
    lines.extend(["", "## Runs", ""])
    lines.extend(compact_markdown_table(runs, ["gene", "seed", "returncode", "seconds", "log", "metrics_csv"]))
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(
        f"wrote {outdir / 'otcfm_leave_one_panel_summary.json'} {report_path}; "
        f"stable_label_aware_passed_all={summary['stable_label_aware_passed_all']}"
    )
    if not summary["stable_label_aware_passed_all"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
