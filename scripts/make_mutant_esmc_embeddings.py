#!/usr/bin/env python3
"""Build verified mutant protein FASTA/metadata for later ESM-C embedding."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from hopfield_fitting_common import DEFAULT_METADATA, DEFAULT_OUTDIR, ensure_dir, load_metadata, write_json

MUTATION_RE = re.compile(r"^(?P<wt>[A-Z])(?P<pos>[0-9]+)(?P<mut>[A-Z])$")
DEFAULT_PANEL = {
    "HNF4A": ["R85W", "S87N", "R89W", "T139I"],
    "TP53": ["R248Q", "R248W", "R273H", "R342Q"],
    "GATA2": ["R396Q", "R362Q"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--gene", action="append", default=None)
    parser.add_argument("--mutation", action="append", default=None, help="Mutation like R85W. Repeatable.")
    parser.add_argument("--min-cells", type=int, default=5)
    parser.add_argument("--require-label", action="append", default=["responder"], help="Allowed label_status values.")
    return parser.parse_args()


def parse_mutation(value: str) -> tuple[str, int, str]:
    match = MUTATION_RE.match(value.strip())
    if not match:
        raise ValueError(f"mutation must look like R85W, got {value!r}")
    return match.group("wt"), int(match.group("pos")), match.group("mut")


def fasta_record(name: str, sequence: str, width: int = 80) -> str:
    chunks = [sequence[index : index + width] for index in range(0, len(sequence), width)]
    return f">{name}\n" + "\n".join(chunks) + "\n"


def main() -> None:
    args = parse_args()
    outdir = ensure_dir(args.outdir)
    metadata = load_metadata(
        args.metadata,
        require_columns=["gene_symbol", "isoform_id", "isoform_embedding_id", "sequence", "n_cells", "label_status"],
    )
    metadata = metadata.loc[pd.to_numeric(metadata["n_cells"], errors="coerce").fillna(0) >= args.min_cells].copy()
    if args.require_label:
        metadata = metadata.loc[metadata["label_status"].astype(str).isin(set(args.require_label))].copy()

    if args.gene:
        panel = {gene: args.mutation or DEFAULT_PANEL.get(gene, []) for gene in args.gene}
    else:
        panel = DEFAULT_PANEL

    records = []
    fasta_chunks = []
    for gene, mutations in panel.items():
        gene_rows = metadata.loc[metadata["gene_symbol"].astype(str) == str(gene)].copy()
        for _, row in gene_rows.iterrows():
            sequence = str(row["sequence"])
            if not sequence or sequence == "nan":
                continue
            for mutation in mutations:
                wt, pos, mut = parse_mutation(mutation)
                zero = pos - 1
                status = "ok"
                reason = ""
                mutant_sequence = ""
                observed = ""
                if zero < 0 or zero >= len(sequence):
                    status = "skipped"
                    reason = f"position {pos} outside sequence length {len(sequence)}"
                else:
                    observed = sequence[zero]
                    if observed != wt:
                        status = "skipped"
                        reason = f"coordinate mismatch: expected {wt}{pos}, observed {observed}{pos}"
                    else:
                        mutant_sequence = sequence[:zero] + mut + sequence[zero + 1 :]
                        mutant_id = f"{row['isoform_embedding_id']}__{mutation}"
                        fasta_chunks.append(fasta_record(mutant_id, mutant_sequence))
                records.append(
                    {
                        "gene_symbol": gene,
                        "isoform_id": row["isoform_id"],
                        "isoform_embedding_id": row["isoform_embedding_id"],
                        "mutation": mutation,
                        "label_status": row["label_status"],
                        "n_cells": int(row["n_cells"]),
                        "protein_aa_length": len(sequence),
                        "expected_wt_residue": wt,
                        "position_1based": pos,
                        "observed_residue": observed,
                        "mutant_residue": mut,
                        "status": status,
                        "reason": reason,
                        "mutant_embedding_id": f"{row['isoform_embedding_id']}__{mutation}" if status == "ok" else "",
                    }
                )

    metadata_path = outdir / "mutant_sequences_metadata.csv"
    fasta_path = outdir / "mutant_sequences.fasta"
    report_path = outdir / "mutant_sequences_report.json"
    result = pd.DataFrame(records)
    result.to_csv(metadata_path, index=False)
    fasta_path.write_text("".join(fasta_chunks), encoding="utf-8")
    report = {
        "metadata": args.metadata,
        "metadata_path": metadata_path,
        "fasta_path": fasta_path,
        "n_candidate_rows": int(len(records)),
        "n_verified_mutants": int((result["status"] == "ok").sum()) if not result.empty else 0,
        "embedding_status": "not_generated",
        "embedding_reason": "No local ESM-C model snapshot was found; this script verifies coordinates and writes mutant FASTA for a later embedding run.",
    }
    write_json(report_path, report)
    print(f"wrote {metadata_path} {fasta_path} {report_path}; verified={report['n_verified_mutants']}")


if __name__ == "__main__":
    main()
