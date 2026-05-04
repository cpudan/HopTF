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


def run_gene(args: argparse.Namespace, gene: str, holdouts: pd.DataFrame) -> dict[str, Any]:
    gene_outdir = ensure_dir(args.outdir / gene)
    log_dir = ensure_dir(args.outdir / "logs")
    log_path = log_dir / f"{gene}.log"
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
        str(args.seed),
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
        "isoform_ids": ids,
        "returncode": int(proc.returncode),
        "seconds": float(time.time() - started),
        "log": str(log_path),
        "metrics_csv": str(gene_outdir / f"otcfm_leave_one_{gene}_metrics.csv"),
        "summary_json": str(gene_outdir / f"otcfm_leave_one_{gene}_summary.json"),
    }


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
        keep_cols = ["gene_symbol", "isoform_id", "label_status", "n_cells", "response_score", "protein_aa_length"]
        target_rows.extend(holdouts.loc[:, [column for column in keep_cols if column in holdouts.columns]].to_dict(orient="records"))
        run = run_gene(args, gene, holdouts)
        run_rows.append(run)
        metrics_path = Path(str(run["metrics_csv"]))
        if metrics_path.exists():
            metrics = pd.read_csv(metrics_path)
            metrics["panel_gene"] = gene
            metrics["panel_returncode"] = run["returncode"]
            metrics["panel_log"] = run["log"]
            metric_frames.append(metrics)

    targets = pd.DataFrame(target_rows)
    runs = pd.DataFrame(run_rows)
    metrics = pd.concat(metric_frames, ignore_index=True) if metric_frames else pd.DataFrame()
    targets_path = outdir / "otcfm_leave_one_panel_targets.csv"
    runs_path = outdir / "otcfm_leave_one_panel_runs.csv"
    metrics_path = outdir / "otcfm_leave_one_panel_metrics.csv"
    report_path = outdir / "otcfm_leave_one_panel.md"
    targets.to_csv(targets_path, index=False)
    runs.to_csv(runs_path, index=False)
    metrics.to_csv(metrics_path, index=False)

    summary = {
        "genes": genes,
        "min_cells": int(args.min_cells),
        "max_per_label": int(args.max_per_label),
        "steps": int(args.steps),
        "max_train_rows": int(args.max_train_rows),
        "n_genes": int(len(genes)),
        "n_genes_with_targets": int(runs["metrics_csv"].notna().sum()) if not runs.empty else 0,
        "n_holdouts": int(metrics.shape[0]),
        "n_passed": int(metrics["passed"].sum()) if "passed" in metrics else 0,
        "n_failed": int((~metrics["passed"].astype(bool)).sum()) if "passed" in metrics and not metrics.empty else 0,
        "passed_all": bool(metrics["passed"].all()) if "passed" in metrics and not metrics.empty else False,
        "targets_csv": str(targets_path),
        "runs_csv": str(runs_path),
        "metrics_csv": str(metrics_path),
        "report_md": str(report_path),
    }
    write_json(outdir / "otcfm_leave_one_panel_summary.json", summary)

    lines = [
        "# OT-CFM Leave-One Isoform Panel",
        "",
        f"Status: {'PASS' if summary['passed_all'] else 'NEEDS DEBUG'}",
        f"Holdouts: {summary['n_passed']} / {summary['n_holdouts']} passed",
        f"Genes: {', '.join(genes)}",
        "",
        "## Targets",
        "",
    ]
    lines.extend(compact_markdown_table(targets, ["gene_symbol", "isoform_id", "label_status", "n_cells", "response_score", "protein_aa_length"]))
    lines.extend(["", "## Metrics", ""])
    metric_columns = [
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
    ]
    lines.extend(compact_markdown_table(metrics, [column for column in metric_columns if column in metrics.columns]))
    lines.extend(["", "## Runs", ""])
    lines.extend(compact_markdown_table(runs, ["gene", "returncode", "seconds", "log", "metrics_csv"]))
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"wrote {outdir / 'otcfm_leave_one_panel_summary.json'} {report_path}; passed_all={summary['passed_all']}")
    if not summary["passed_all"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
