#!/usr/bin/env python3
"""Compile HopTF smoke, panel, baseline, and mutation status into one Markdown report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overnight-dir", type=Path, required=True)
    parser.add_argument("--panel-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    if not path.exists():
        return {"status": "missing", "path": str(path)}
    return json.loads(path.read_text(encoding="utf-8"))


def markdown_table(df: pd.DataFrame, columns: list[str]) -> list[str]:
    if df.empty:
        return ["No rows."]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in df.iterrows():
        values = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                value = f"{value:.4g}"
            values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def main() -> None:
    args = parse_args()
    overnight = args.overnight_dir
    panel = args.panel_dir
    overnight_summary = load_json(overnight / "overnight_run_summary.json")
    mutant_report = load_json(overnight / "mutant_endpoint_predictions_report.json")
    controlled_summary = load_json(overnight / "controlled_sequence_baselines_summary.json")
    panel_summary = load_json(panel / "otcfm_leave_one_panel_summary.json")
    panel_aggregate_path = panel / "otcfm_leave_one_panel_aggregate.csv"
    panel_aggregate = pd.read_csv(panel_aggregate_path) if panel_aggregate_path.exists() else pd.DataFrame()

    lines = [
        "# HopTF Plan Execution Status",
        "",
        "## Smoke Pipeline",
        "",
        f"- Status: `{'SUCCESS' if overnight_summary.get('ok') else 'FAILED'}`",
        f"- Completed steps: `{len(overnight_summary.get('results', []))}`",
        f"- Key source: `{overnight_summary.get('key_source')}`",
        f"- Report: `{overnight / 'overnight_summary.md'}`",
        "",
        "## Calibrated Mixed-Response Panel",
        "",
        f"- Seed-level label-aware passes: `{panel_summary.get('n_label_aware_passed')} / {panel_summary.get('n_holdouts')}`",
        f"- Stable label-aware isoforms: `{panel_summary.get('n_stable_label_aware_passed')} / {panel_summary.get('n_isoforms')}`",
        f"- Failure categories: `{panel_summary.get('failure_categories')}`",
        f"- Report: `{panel / 'otcfm_leave_one_panel.md'}`",
        "",
        "### Panel Aggregate",
        "",
    ]
    aggregate_columns = [
        "gene_symbol",
        "isoform_id",
        "label_status",
        "n_cells",
        "response_score",
        "label_aware_pass_rate",
        "stable_label_aware_pass",
        "failure_category",
        "sibling_nonresponders",
        "control_standardized_endpoint_mse_fraction_of_baseline_mean",
        "mean_control_standardized_response_l2_ratio_mean",
    ]
    lines.extend(markdown_table(panel_aggregate, [col for col in aggregate_columns if col in panel_aggregate.columns]))

    controlled_metrics = pd.DataFrame(controlled_summary.get("metrics", []))
    lines.extend(["", "## Controlled Baselines", ""])
    if not controlled_metrics.empty:
        columns = [col for col in ["feature_set", "context", "mse", "improvement_vs_artifact_only"] if col in controlled_metrics.columns]
        lines.extend(markdown_table(controlled_metrics.head(20), columns))
    else:
        lines.append("No controlled baseline rows found.")

    lines.extend(
        [
            "",
            "## Mutation Endpoint Status",
            "",
            f"- Status: `{mutant_report.get('status')}`",
            f"- Reason: `{mutant_report.get('reason', '')}`",
            f"- Missing: `{mutant_report.get('missing', [])}`",
            "",
            "## Remaining Blockers",
            "",
            "- Mutant endpoint predictions require a mutant ESM-C embedding matrix and vocab.",
            "- SCARF/cell-level endpoints are not yet wired into this smoke pipeline.",
            "- `SOX5-1` has no same-gene nonresponder sibling support and remains an unsupported nonresponder holdout.",
            "- `IKZF3-5` remains a hard responder endpoint failure under the calibrated panel.",
        ]
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
