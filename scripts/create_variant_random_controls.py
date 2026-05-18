#!/usr/bin/env python3
"""Create matched random missense controls for a locked ClinVar variant panel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


STANDARD_AA = tuple("ACDEFGHIKLMNPQRSTVWY")
AA_CLASS = {
    "A": "small_hydrophobic",
    "V": "small_hydrophobic",
    "L": "hydrophobic",
    "I": "hydrophobic",
    "M": "hydrophobic",
    "F": "aromatic",
    "W": "aromatic",
    "Y": "aromatic",
    "S": "polar",
    "T": "polar",
    "N": "polar",
    "Q": "polar",
    "C": "polar",
    "K": "positive",
    "R": "positive",
    "H": "positive",
    "D": "negative",
    "E": "negative",
    "G": "special",
    "P": "special",
}
CLASS_TO_AA: dict[str, tuple[str, ...]] = {}
for aa, aa_class in AA_CLASS.items():
    CLASS_TO_AA.setdefault(aa_class, tuple())
    CLASS_TO_AA[aa_class] = (*CLASS_TO_AA[aa_class], aa)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--locked-panel", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--region-bins", type=int, default=5)
    return parser.parse_args()


def fasta_record(name: str, sequence: str, width: int = 80) -> str:
    return f">{name}\n" + "\n".join(sequence[index : index + width] for index in range(0, len(sequence), width)) + "\n"


def residue_class(aa: str) -> str:
    return AA_CLASS.get(str(aa), "other")


def choose_mutant_residue(rng: np.random.Generator, wt_residue: str, preferred_class: str) -> tuple[str, str]:
    candidates = [aa for aa in CLASS_TO_AA.get(preferred_class, STANDARD_AA) if aa != wt_residue]
    if not candidates:
        candidates = [aa for aa in STANDARD_AA if aa != wt_residue]
        preferred_class = "fallback_any_standard"
    return str(rng.choice(candidates)), preferred_class


def choose_position(
    rng: np.random.Generator,
    sequence: str,
    source_position: int,
    region_bins: int,
) -> tuple[int, str, int]:
    length = len(sequence)
    source_bin = min(region_bins - 1, int((source_position - 1) * region_bins / max(length, 1)))
    lower = int(source_bin * length / region_bins) + 1
    upper = int((source_bin + 1) * length / region_bins)
    standard_positions = [
        pos
        for pos in range(lower, upper + 1)
        if pos != source_position and sequence[pos - 1] in STANDARD_AA
    ]
    match_quality = "same_length_bin"
    if not standard_positions:
        standard_positions = [
            pos
            for pos, residue in enumerate(sequence, start=1)
            if pos != source_position and residue in STANDARD_AA
        ]
        match_quality = "same_isoform_any_position"
    if not standard_positions:
        raise ValueError("no standard amino-acid position available for random control")
    return int(rng.choice(standard_positions)), match_quality, source_bin


def make_controls(panel: pd.DataFrame, seed: int, region_bins: int) -> tuple[pd.DataFrame, dict[str, str]]:
    rng = np.random.default_rng(seed)
    records = []
    sequences = {}
    for _, row in panel.iterrows():
        if str(row.get("mapping_status", "")) != "ok":
            continue
        sequence = str(row["sequence"])
        source_position = int(row["position_1based"])
        position, match_quality, source_bin = choose_position(rng, sequence, source_position, region_bins)
        wt_residue = sequence[position - 1]
        preferred_class = residue_class(str(row["mut_aa"]))
        mutant_residue, used_class = choose_mutant_residue(rng, wt_residue, preferred_class)
        mutant_sequence = sequence[: position - 1] + mutant_residue + sequence[position:]
        mutation = f"{wt_residue}{position}{mutant_residue}"
        variation_id = str(row["VariationID"])
        source_mutation = str(row["mutation"])
        mutant_embedding_id = (
            f"{row['isoform_embedding_id']}__random_for_clinvar{variation_id}_{source_mutation}_to_{mutation}"
        )
        sequences[mutant_embedding_id] = mutant_sequence
        records.append(
            {
                "gene_symbol": row["GeneSymbol"],
                "isoform_id": row["isoform_id"],
                "isoform_embedding_id": row["isoform_embedding_id"],
                "mutation": mutation,
                "label_status": row["local_label_status"],
                "n_cells": row["n_cells"],
                "protein_aa_length": row["protein_aa_length"],
                "expected_wt_residue": wt_residue,
                "position_1based": position,
                "observed_residue": wt_residue,
                "mutant_residue": mutant_residue,
                "status": "ok",
                "reason": "",
                "mutant_embedding_id": mutant_embedding_id,
                "label_class": row["label_class"],
                "control_type": "matched_random",
                "matched_real_mutant_embedding_id": row["mutant_embedding_id"],
                "matched_real_clinvar_variation_id": row["VariationID"],
                "matched_real_clinvar_allele_id": row["#AlleleID"],
                "matched_real_mutation": row["mutation"],
                "matched_real_position_1based": row["position_1based"],
                "matched_real_mutant_residue": row["mut_aa"],
                "matched_real_mutant_residue_class": preferred_class,
                "random_mutant_residue_class": residue_class(mutant_residue),
                "aa_class_match_used": used_class,
                "position_match_quality": match_quality,
                "sequence_region_bin": source_bin,
                "clinical_significance": row["ClinicalSignificance"],
                "review_status": row["ReviewStatus"],
                "number_submitters": row["NumberSubmitters"],
            }
        )
    return pd.DataFrame(records), sequences


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    panel = pd.read_csv(args.locked_panel)
    controls, sequences = make_controls(panel, args.seed, args.region_bins)
    if controls.empty:
        raise ValueError("no random controls were generated")

    vocab = controls["mutant_embedding_id"].astype(str).tolist()
    fasta = "".join(fasta_record(record_id, sequences[record_id]) for record_id in vocab)

    controls.to_csv(args.outdir / "random_control_mutant_sequences_metadata.csv", index=False)
    (args.outdir / "random_control_mutant_sequences.fasta").write_text(fasta)
    (args.outdir / "random_control_mutant_esmc_vocab_input.json").write_text(json.dumps(vocab, indent=2) + "\n")
    (args.outdir / "random_control_mutant_esmc_sequences.json").write_text(json.dumps(sequences, indent=2) + "\n")

    report = {
        "status": "ok",
        "seed": int(args.seed),
        "region_bins": int(args.region_bins),
        "n_controls": int(len(controls)),
        "counts_by_matched_label_class": {
            str(k): int(v) for k, v in controls["label_class"].value_counts().sort_index().items()
        },
        "position_match_quality": {
            str(k): int(v) for k, v in controls["position_match_quality"].value_counts().sort_index().items()
        },
        "aa_class_match_used": {
            str(k): int(v) for k, v in controls["aa_class_match_used"].value_counts().sort_index().items()
        },
        "outputs": {
            "metadata": str(args.outdir / "random_control_mutant_sequences_metadata.csv"),
            "fasta": str(args.outdir / "random_control_mutant_sequences.fasta"),
            "vocab": str(args.outdir / "random_control_mutant_esmc_vocab_input.json"),
            "sequences": str(args.outdir / "random_control_mutant_esmc_sequences.json"),
        },
    }
    (args.outdir / "random_control_curation_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
