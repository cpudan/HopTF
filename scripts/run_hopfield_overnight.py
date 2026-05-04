#!/usr/bin/env python3
"""Run the overnight Hopfield/OT-CFM smoke pipeline and write a report."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=Path("tmp/hopfield_fitting"))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--fast", action="store_true", help="Use shorter training runs for debugging.")
    parser.add_argument(
        "--key-source",
        choices=("auto", "synthetic", "gene_pooled_npz"),
        default="auto",
        help="Use synthetic keys, real gene-pooled AlphaGenome keys, or real keys when the archive is present.",
    )
    parser.add_argument(
        "--alphagenome-npz",
        type=Path,
        default=Path("data/hf_min/external/AI4Genome/gene_pooled_embeddings.npz"),
        help="Gene-pooled AlphaGenome NPZ used when --key-source is gene_pooled_npz/auto.",
    )
    parser.add_argument("--holdout-gene", action="append", default=None, help="Repeatable gene for trained leave-one-isoform CFM checks.")
    parser.add_argument("--max-holdouts-per-gene", type=int, default=2)
    return parser.parse_args()


def run_step(name: str, command: list[str], log_dir: Path) -> dict[str, object]:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{name}.log"
    started = time.time()
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(command) + "\n\n")
        handle.flush()
        proc = subprocess.run(command, cwd=REPO_ROOT, stdout=handle, stderr=subprocess.STDOUT, text=True)
    elapsed = time.time() - started
    return {
        "name": name,
        "command": command,
        "returncode": int(proc.returncode),
        "seconds": elapsed,
        "log": str(log_path),
    }


def tail(path: Path, n_lines: int = 30) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-n_lines:])


def main() -> None:
    args = parse_args()
    outdir = args.outdir
    log_dir = outdir / "logs"
    outdir.mkdir(parents=True, exist_ok=True)
    steps_factor = 0.35 if args.fast else 1.0
    hopfield_steps = max(120, int(500 * steps_factor))
    otcfm_steps = max(180, int(700 * steps_factor))
    leave_steps = max(160, int(500 * steps_factor))
    key_source = args.key_source
    if key_source == "auto":
        key_source = "gene_pooled_npz" if args.alphagenome_npz.exists() else "synthetic"
    if key_source == "gene_pooled_npz":
        hopfield_steps = max(hopfield_steps, 2500)
    holdout_genes = args.holdout_gene or ["HNF4A"]

    if key_source == "gene_pooled_npz":
        prepare_key_command = [
            args.python,
            "scripts/prepare_alphagenome_keys.py",
            "--source",
            "gene_pooled_npz",
            "--input",
            str(args.alphagenome_npz),
            "--outdir",
            str(outdir),
        ]
        hopfield_real_extra = ["--lr", "0.004", "--beta", "25"]
    else:
        prepare_key_command = [
            args.python,
            "scripts/prepare_alphagenome_keys.py",
            "--source",
            "synthetic",
            "--synthetic-key-count",
            "256",
            "--synthetic-key-dim",
            "64",
            "--outdir",
            str(outdir),
        ]
        hopfield_real_extra = []

    commands: list[tuple[str, list[str]]] = [
        ("inspect_inputs", [args.python, "scripts/inspect_hopfield_inputs.py", "--outdir", str(outdir)]),
        (f"prepare_{key_source}_keys", prepare_key_command),
        ("build_latents", [args.python, "scripts/build_perturbation_latents.py", "--outdir", str(outdir)]),
        (
            "hopfield_synthetic",
            [args.python, "scripts/train_hopfield_projection.py", "--mode", "synthetic", "--steps", str(hopfield_steps), "--outdir", str(outdir)],
        ),
        (
            "hopfield_real_overfit",
            [
                args.python,
                "scripts/train_hopfield_projection.py",
                "--mode",
                "real-overfit",
                "--steps",
                str(hopfield_steps),
                "--max-real-rows",
                "64",
                "--outdir",
                str(outdir),
            ]
            + hopfield_real_extra,
        ),
        (
            "otcfm_synthetic",
            [
                args.python,
                "scripts/train_otcfm_sequence_conditioned.py",
                "--mode",
                "synthetic",
                "--steps",
                str(otcfm_steps),
                "--outdir",
                str(outdir),
            ],
        ),
        (
            "otcfm_real_overfit",
            [
                args.python,
                "scripts/train_otcfm_sequence_conditioned.py",
                "--mode",
                "real-overfit",
                "--steps",
                str(otcfm_steps),
                "--max-train-rows",
                "256",
                "--outdir",
                str(outdir),
            ],
        ),
        (
            "isoform_holdout_baselines",
            [
                args.python,
                "scripts/evaluate_isoform_holdouts.py",
                "--gene",
                "HNF4A",
                "--gene",
                "TP53",
                "--gene",
                "NFATC1",
                "--outdir",
                str(outdir),
            ],
        ),
    ]

    for gene in holdout_genes:
        commands.append(
            (
                f"otcfm_leave_one_{gene.lower()}",
                [
                args.python,
                "scripts/train_otcfm_sequence_conditioned.py",
                "--mode",
                "leave-one-isoform",
                "--holdout-gene",
                str(gene),
                "--max-holdouts",
                str(args.max_holdouts_per_gene),
                "--max-train-rows",
                "256",
                "--steps",
                str(leave_steps),
                "--outdir",
                str(outdir),
                ],
            )
        )

    commands.extend(
        [
            ("mutant_sequences", [args.python, "scripts/make_mutant_esmc_embeddings.py", "--outdir", str(outdir)]),
            (
                "mutant_endpoint_predictions",
                [
                    args.python,
                    "scripts/evaluate_mutant_endpoint_predictions.py",
                    "--outdir",
                    str(outdir),
                    "--mutant-metadata",
                    str(outdir / "mutant_sequences_metadata.csv"),
                    "--mutant-embedding-matrix",
                    str(outdir / "mutant_esmc_embeddings.npy"),
                    "--mutant-vocab",
                    str(outdir / "mutant_esmc_vocab.json"),
                    "--latents",
                    str(outdir / "perturbation_latents.npz"),
                    "--checkpoint",
                    str(outdir / "otcfm_real_overfit.pt"),
                ],
            ),
            (
                "sequence_endpoint_baselines",
                [
                    args.python,
                    "scripts/evaluate_sequence_endpoint_baselines.py",
                    "--outdir",
                    str(outdir),
                    "--latents",
                    str(outdir / "perturbation_latents.npz"),
                    "--hopfield-checkpoint",
                    str(outdir / "hopfield_real_overfit.pt"),
                    "--otcfm-metrics-dir",
                    str(outdir),
                ],
            ),
            (
                "controlled_sequence_baselines",
                [
                    args.python,
                    "scripts/evaluate_controlled_sequence_baselines.py",
                    "--outdir",
                    str(outdir),
                    "--latents",
                    str(outdir / "perturbation_latents.npz"),
                    "--hopfield-checkpoint",
                    str(outdir / "hopfield_real_overfit.pt"),
                ],
            ),
        ]
    )

    results = []
    for name, command in commands:
        result = run_step(name, command, log_dir)
        results.append(result)
        if result["returncode"] != 0:
            break

    summary_path = outdir / "overnight_run_summary.json"
    report_path = outdir / "overnight_summary.md"
    ok = all(result["returncode"] == 0 for result in results) and len(results) == len(commands)
    summary = {
        "ok": ok,
        "key_source": key_source,
        "alphagenome_npz": str(args.alphagenome_npz),
        "holdout_genes": holdout_genes,
        "results": results,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Hopfield/OT-CFM Overnight Smoke Run",
        "",
        f"Status: {'SUCCESS' if ok else 'FAILED'}",
        f"Completed steps: {len(results)} / {len(commands)}",
        f"Key source: `{key_source}`",
        f"Holdout genes: `{', '.join(holdout_genes)}`",
        "",
        "## Steps",
        "",
    ]
    for result in results:
        status = "PASS" if result["returncode"] == 0 else "FAIL"
        lines.append(f"- {status} `{result['name']}` in {result['seconds']:.1f}s, log `{result['log']}`")
    if not ok and results:
        failed = results[-1]
        lines.extend(["", f"## Failed Step: `{failed['name']}`", "", "```text", tail(Path(str(failed["log"]))), "```"])
    lines.extend(["", "## Key Outputs", ""])
    key_outputs = [
        "input_contract.json",
        "alphagenome_keys_report.json",
        "perturbation_latents_report.json",
        "hopfield_synthetic_metrics.json",
        "hopfield_real_overfit_metrics.json",
        "otcfm_synthetic_metrics.json",
        "otcfm_real_overfit_metrics.json",
        "isoform_holdout_baselines_summary.json",
        "otcfm_leave_one_HNF4A_summary.json",
        "sequence_endpoint_baselines_summary.json",
        "sequence_endpoint_baselines.md",
        "controlled_sequence_baselines_summary.json",
        "controlled_evaluation_report.md",
        "mutant_sequences_report.json",
        "mutant_endpoint_predictions_report.json",
        "mutant_endpoint_predictions.csv",
    ]
    for gene in holdout_genes:
        key_outputs.append(f"otcfm_leave_one_{gene}_summary.json")
    for output in dict.fromkeys(key_outputs):
        if (outdir / output).exists():
            lines.append(f"- `{outdir / output}`")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {summary_path} {report_path}; ok={ok}")
    if not ok:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
