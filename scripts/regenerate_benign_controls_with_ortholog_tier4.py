#!/usr/bin/env python3
"""Regenerate HopTF 10-per-isoform controls with ortholog-supported tier 4.

Natural evidence is used first. Remaining slots are filled with the new
ortholog-supported tier 4 controls. If an isoform still has fewer than the
requested number of rows, the previous conservative fallback rows are retained
as explicitly labeled tier 5 legacy fallback controls so the table remains
exactly 10 variants per isoform.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd


REPO = Path(__file__).resolve().parents[1]
DATE_TAG = "2026-05-17"
DEFAULT_METADATA = REPO / "data/processed/linear_probe/tfatlas_subsample/PERTURBATION_METADATA_hard_local_subsample.csv"
DEFAULT_OLD_DIR = REPO / "data/processed/variant_controls/tf_isoform_benign_controls_10_per_isoform_20260517"
DEFAULT_ORTHO_DIR = REPO / "data/processed/variant_controls/tf_isoform_ortholog_supported_controls_20260517"
DEFAULT_OUTDIR = REPO / "data/processed/variant_controls/tf_isoform_benign_controls_10_per_isoform_ortholog_tier4_20260517"
BASE = "tf_isoform_benign_controls_10_per_isoform_ortholog_tier4_20260517"

AA1_TO_AA3 = {
    "A": "Ala",
    "R": "Arg",
    "N": "Asn",
    "D": "Asp",
    "C": "Cys",
    "Q": "Gln",
    "E": "Glu",
    "G": "Gly",
    "H": "His",
    "I": "Ile",
    "L": "Leu",
    "K": "Lys",
    "M": "Met",
    "F": "Phe",
    "P": "Pro",
    "S": "Ser",
    "T": "Thr",
    "W": "Trp",
    "Y": "Tyr",
    "V": "Val",
}
AA3_TO_AA1 = {v: k for k, v in AA1_TO_AA3.items()}
HGVS_P_RE = re.compile(r"^p\.([A-Z][a-z]{2})([0-9]+)([A-Z][a-z]{2})$")
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")
EXCLUDED_SYNTHETIC_WT = set("CHGPW")
CONSERVATIVE_ALTS = {
    "A": ["V", "S", "T"],
    "V": ["I", "L", "A"],
    "I": ["V", "L"],
    "L": ["I", "V", "M"],
    "M": ["L", "I"],
    "F": ["Y", "L"],
    "Y": ["F"],
    "S": ["T", "A", "N"],
    "T": ["S", "A"],
    "N": ["Q", "S"],
    "Q": ["N", "E"],
    "D": ["E", "N"],
    "E": ["D", "Q"],
    "K": ["R", "Q"],
    "R": ["K", "Q"],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    p.add_argument("--old-dir", type=Path, default=DEFAULT_OLD_DIR)
    p.add_argument("--ortholog-dir", type=Path, default=DEFAULT_ORTHO_DIR)
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    p.add_argument("--per-isoform", type=int, default=10)
    return p.parse_args()


def safe_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value)


def stable_fraction(text: str) -> float:
    import hashlib

    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) / float(16**16)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, low_memory=False)


def metadata_isoforms(path: Path) -> pd.DataFrame:
    meta = read_csv(path)
    if "is_control" in meta.columns:
        is_control = meta["is_control"].astype(str).str.lower().isin({"true", "1", "yes"})
        meta = meta.loc[~is_control].copy()
    required = [
        "isoform_embedding_id",
        "gene_symbol",
        "isoform_id",
        "label_status",
        "n_cells",
        "protein_aa_length",
        "sequence",
    ]
    missing = [c for c in required if c not in meta.columns]
    if missing:
        raise ValueError(f"metadata missing required columns: {missing}")
    return meta.drop_duplicates("isoform_embedding_id", keep="first").reset_index(drop=True)


def sort_candidates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col, default in [
        ("source_priority", 99),
        ("confidence_score", 0.0),
        ("selection_rank", 999999),
        ("aa_position", 999999),
        ("source_record_id", ""),
    ]:
        if col not in out.columns:
            out[col] = default
    out["_source_priority_sort"] = pd.to_numeric(out["source_priority"], errors="coerce").fillna(99)
    out["_confidence_sort"] = pd.to_numeric(out["confidence_score"], errors="coerce").fillna(0.0)
    out["_selection_rank_sort"] = pd.to_numeric(out["selection_rank"], errors="coerce").fillna(999999)
    out["_aa_position_sort"] = pd.to_numeric(out["aa_position"], errors="coerce").fillna(999999)
    out = out.sort_values(
        [
            "_source_priority_sort",
            "_confidence_sort",
            "_selection_rank_sort",
            "_aa_position_sort",
            "source_record_id",
        ],
        ascending=[True, False, True, True, True],
    )
    return out.drop(
        columns=[
            "_source_priority_sort",
            "_confidence_sort",
            "_selection_rank_sort",
            "_aa_position_sort",
        ],
        errors="ignore",
    )


def mutation_key(row: dict[str, Any]) -> tuple[str, str]:
    pos = safe_str(row.get("aa_position"))
    alt = safe_str(row.get("aa_alt")).upper()
    mut = safe_str(row.get("mutation") or row.get("mutation_shorthand")).upper()
    return (pos, alt or mut)


def mutate_sequence(seq: str, pos_1based: int, alt: str) -> str:
    return seq[: pos_1based - 1] + alt + seq[pos_1based:]


def aa1_to_aa3(aa: str) -> str:
    aa = safe_str(aa).strip().upper()
    return AA1_TO_AA3.get(aa, aa)


def make_hgvs_p(ref: str, pos: Any, alt: str) -> str:
    return f"p.{aa1_to_aa3(ref)}{int(float(pos))}{aa1_to_aa3(alt)}"


def parse_hgvs_p(value: str) -> tuple[str, int, str] | None:
    match = HGVS_P_RE.match(safe_str(value).strip())
    if not match:
        return None
    ref3, pos, alt3 = match.groups()
    return AA3_TO_AA1.get(ref3, "?"), int(pos), AA3_TO_AA1.get(alt3, "?")


def validate_rows(df: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        seq = safe_str(row.get("sequence"))
        mut_seq = safe_str(row.get("mutant_sequence"))
        ref = safe_str(row.get("aa_ref")).upper()
        alt = safe_str(row.get("aa_alt")).upper()
        status = "ok"
        try:
            pos = int(float(row.get("aa_position")))
        except Exception:
            pos = -1
            status = "bad_position"

        if status == "ok":
            if not seq or pos < 1 or pos > len(seq):
                status = "position_out_of_bounds"
            elif seq[pos - 1] != ref:
                status = f"reference_mismatch:{seq[pos - 1]}!={ref}"
            elif not mut_seq or len(mut_seq) != len(seq):
                status = "bad_mutant_sequence_length"
            elif mut_seq[pos - 1] != alt:
                status = f"mutant_mismatch:{mut_seq[pos - 1]}!={alt}"
            else:
                expected = seq[: pos - 1] + alt + seq[pos:]
                if mut_seq != expected:
                    status = "mutant_sequence_not_single_substitution"

        hgvs = safe_str(row.get("hgvs_p") or row.get("hgvs_protein"))
        parsed = parse_hgvs_p(hgvs)
        if status == "ok" and parsed is None:
            status = "bad_hgvs_p"
        elif status == "ok" and parsed != (ref, pos, alt):
            status = f"hgvs_mismatch:{parsed}!={(ref, pos, alt)}"

        records.append(
            {
                "variant_uid": row.get("variant_uid"),
                "hgvs_p": hgvs,
                "parsed_ref": parsed[0] if parsed else "",
                "parsed_pos": parsed[1] if parsed else "",
                "parsed_alt": parsed[2] if parsed else "",
                "validation_status": status,
            }
        )
    return pd.DataFrame(records)


def fasta_records(rows: pd.DataFrame) -> str:
    lines: list[str] = []
    for _, row in rows.iterrows():
        seq = safe_str(row["mutant_sequence"])
        lines.append(f">{row['variant_uid']}")
        for i in range(0, len(seq), 80):
            lines.append(seq[i : i + 80])
    return "\n".join(lines) + "\n"


def coerce_ortholog_rows(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "gene" not in out.columns and "gene_symbol" in out.columns:
        out["gene"] = out["gene_symbol"]
    if "isoform" not in out.columns and "isoform_id" in out.columns:
        out["isoform"] = out["isoform_id"]
    out["label"] = "putative_benign_control"
    out["label_tier"] = "Tier 4 ortholog-supported synthetic control"
    out["label_source"] = "ortholog residue support"
    out["evidence_source"] = "ensembl_compara_ortholog_alignment"
    out["evidence_tier"] = "Tier 4 ortholog-supported synthetic control"
    out["evidence_confidence"] = "synthetic_evolutionary_support"
    out["clinical_benign_flag"] = False
    out["population_tolerated_flag"] = False
    out["functional_neutral_flag"] = False
    out["synthetic_ortholog_flag"] = True
    out["synthetic_conservative_flag"] = False
    out["include_final"] = True
    out["source_priority"] = 8
    out["hgvs_p_source"] = "local_sequence"
    out["hgvs_parser_backend"] = "internal_mapping"
    out["mutation_shorthand"] = out["mutation"]
    out["hgvs_p"] = [
        make_hgvs_p(ref, pos, alt)
        for ref, pos, alt in zip(out["aa_ref"], out["aa_position"], out["aa_alt"])
    ]
    out["hgvs_protein"] = out["hgvs_p"]
    out["local_hgvs_p"] = out["isoform_embedding_id"].astype(str) + ":" + out["hgvs_p"].astype(str)
    out["source_variant_uid"] = out["variant_uid"]
    return sort_candidates(out)


def coerce_legacy_rows(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["label_tier"] = "Tier 5 legacy conservative fallback control"
    out["label_source"] = "legacy deterministic conservative substitution"
    out["evidence_source"] = "legacy_synthetic_conservative_sequence_rule"
    out["evidence_tier"] = "Tier 5 legacy conservative fallback control"
    out["evidence_confidence"] = "legacy_fallback_no_natural_or_ortholog_slot"
    out["synthetic_ortholog_flag"] = False
    out["synthetic_conservative_flag"] = True
    out["source_priority"] = 9
    out["synthetic_rule_version"] = "conservative_v1_20260517_legacy_after_ortholog"
    out["selection_notes"] = (
        "Legacy fallback retained only because this isoform had fewer than the "
        "requested number of natural plus ortholog-supported controls."
    )
    out["source_variant_uid"] = out["variant_uid"]
    return sort_candidates(out)


def mark_sensitive_windows(seq: str) -> set[int]:
    avoid: set[int] = set()
    if not seq:
        return avoid
    n = len(seq)
    for match in re.finditer(r"C.{2,4}C.{8,20}H.{2,8}H", seq):
        lo = max(1, match.start() + 1 - 2)
        hi = min(n, match.end() + 2)
        avoid.update(range(lo, hi + 1))
    for i in range(n):
        lo = max(0, i - 7)
        hi = min(n, i + 8)
        window = seq[lo:hi]
        if len(window) >= 10 and sum(aa in "KR" for aa in window) / len(window) >= 0.42:
            avoid.add(i + 1)
    for i in range(n - 21):
        if seq[i] == seq[i + 7] == seq[i + 14] == seq[i + 21] == "L":
            avoid.update(range(i + 1, i + 22))
    return avoid


def local_low_complexity(seq: str, pos_1based: int) -> bool:
    lo = max(0, pos_1based - 1 - 5)
    hi = min(len(seq), pos_1based - 1 + 6)
    window = seq[lo:hi]
    if len(window) < 6:
        return False
    return max(window.count(aa) for aa in set(window)) / len(window) >= 0.55


def supplemental_tier5_rows(
    iso_row: pd.Series,
    count_needed: int,
    existing_keys: set[tuple[str, str]],
) -> list[dict[str, Any]]:
    seq = safe_str(iso_row["sequence"])
    iso = safe_str(iso_row["isoform_embedding_id"])
    if not seq:
        return []
    avoid = mark_sensitive_windows(seq)
    candidates: list[tuple[int, float, int, str, str]] = []
    for pos, wt in enumerate(seq, start=1):
        if (safe_str(pos), "") in existing_keys:
            continue
        if wt not in STANDARD_AA or wt not in CONSERVATIVE_ALTS:
            continue
        strictness = 0
        if pos <= 5 or pos > len(seq) - 5 or local_low_complexity(seq, pos):
            strictness += 1
        if pos in avoid:
            strictness += 3
        if wt in EXCLUDED_SYNTHETIC_WT:
            strictness += 5
        if strictness >= 5:
            continue
        for alt in CONSERVATIVE_ALTS[wt]:
            key = (safe_str(pos), alt)
            if key in existing_keys:
                continue
            candidates.append((strictness, stable_fraction(f"{iso}|{pos}|{wt}>{alt}|tier5"), pos, wt, alt))
    candidates.sort(key=lambda x: (x[0], x[1], x[2], x[4]))

    rows: list[dict[str, Any]] = []
    used_positions: set[int] = set()
    for strictness, _, pos, wt, alt in candidates:
        if pos in used_positions:
            continue
        used_positions.add(pos)
        mut = f"{wt}{pos}{alt}"
        hgvs_p = make_hgvs_p(wt, pos, alt)
        rows.append(
            {
                "source_record_id": f"tier5_synthetic:{iso}:{mut}",
                "source_priority": 9,
                "label": "putative_benign_control",
                "label_tier": "Tier 5 legacy conservative fallback control",
                "label_source": "legacy deterministic conservative substitution",
                "evidence_source": "legacy_synthetic_conservative_sequence_rule",
                "evidence_tier": "Tier 5 legacy conservative fallback control",
                "evidence_confidence": "legacy_fallback_no_natural_or_ortholog_slot",
                "clinical_benign_flag": False,
                "population_tolerated_flag": False,
                "functional_neutral_flag": False,
                "synthetic_ortholog_flag": False,
                "synthetic_conservative_flag": True,
                "include_final": True,
                "exclusion_reason": "",
                "isoform_embedding_id": iso,
                "gene": iso_row.get("gene_symbol"),
                "isoform": iso_row.get("isoform_id"),
                "label_status": iso_row.get("label_status"),
                "n_cells": iso_row.get("n_cells"),
                "protein_aa_length": iso_row.get("protein_aa_length"),
                "sequence": seq,
                "mutation": mut,
                "mutation_shorthand": mut,
                "aa_position": pos,
                "aa_ref": wt,
                "aa_alt": alt,
                "hgvs_p": hgvs_p,
                "hgvs_protein": hgvs_p,
                "local_hgvs_p": f"{iso}:{hgvs_p}",
                "hgvs_p_source": "local_sequence",
                "hgvs_parser_backend": "internal_mapping",
                "mutant_sequence": mutate_sequence(seq, pos, alt),
                "selection_notes": (
                    "Tier 5 fallback generated only because natural and ortholog-supported "
                    "controls did not provide enough unique substitutions for this isoform."
                ),
                "source_rank_detail": f"synthetic_strictness={strictness}; supplemental_tier5=true",
                "confidence_score": 100 - 10 * strictness,
                "synthetic_rule_version": "conservative_v1_20260517_tier5_after_ortholog",
                "source_variant_uid": "",
            }
        )
        if len(rows) >= count_needed:
            break
    return rows


def align_columns(rows: pd.DataFrame, reference_columns: list[str]) -> pd.DataFrame:
    cols = list(reference_columns)
    for col in rows.columns:
        if col not in cols:
            cols.append(col)
    for col in cols:
        if col not in rows.columns:
            rows[col] = pd.NA
    if "sequence" in cols:
        cols = [c for c in cols if c not in {"sequence", "mutant_sequence"}] + ["sequence", "mutant_sequence"]
    return rows[cols]


def write_outputs(
    outdir: Path,
    final: pd.DataFrame,
    natural_selected: pd.DataFrame,
    all_natural: pd.DataFrame,
    iso_summary: pd.DataFrame,
    summary: dict[str, Any],
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    full_csv = outdir / f"{BASE}.csv"
    lite_csv = outdir / f"{BASE}.metadata_only.csv"
    natural_csv = outdir / f"{BASE}.natural_evidence_only.csv"
    natural_lite_csv = outdir / f"{BASE}.natural_evidence_only.metadata_only.csv"
    all_natural_csv = outdir / "tf_isoform_benign_controls_all_available_natural_20260517.csv"
    all_natural_lite_csv = outdir / "tf_isoform_benign_controls_all_available_natural_20260517.metadata_only.csv"
    iso_csv = outdir / f"{BASE}.isoform_summary.csv"
    validation_csv = outdir / f"{BASE}.hgvs_parse_validation.csv"
    fasta = outdir / f"{BASE}.fasta"
    vocab = outdir / f"{BASE}.vocab.json"
    seq_json = outdir / f"{BASE}.sequences.json"
    summary_json = outdir / f"{BASE}.summary.json"
    readme = outdir / "README.md"

    metadata_cols = [c for c in final.columns if c not in {"sequence", "mutant_sequence"}]
    final.to_csv(full_csv, index=False)
    final[metadata_cols].to_csv(lite_csv, index=False)
    natural_selected.to_csv(natural_csv, index=False)
    natural_selected[[c for c in natural_selected.columns if c not in {"sequence", "mutant_sequence"}]].to_csv(
        natural_lite_csv, index=False
    )
    all_natural.to_csv(all_natural_csv, index=False)
    all_natural[[c for c in all_natural.columns if c not in {"sequence", "mutant_sequence"}]].to_csv(
        all_natural_lite_csv, index=False
    )
    iso_summary.to_csv(iso_csv, index=False)
    validation = validate_rows(final)
    validation.to_csv(validation_csv, index=False)
    if not validation["validation_status"].eq("ok").all():
        failed = validation.loc[~validation["validation_status"].eq("ok")].head(20)
        raise RuntimeError("validation failed:\n" + failed.to_string(index=False))

    fasta.write_text(fasta_records(final), encoding="utf-8")
    vocab.write_text(json.dumps(final["variant_uid"].tolist(), indent=2) + "\n", encoding="utf-8")
    seq_json.write_text(
        json.dumps(dict(zip(final["variant_uid"], final["mutant_sequence"])), indent=2) + "\n",
        encoding="utf-8",
    )

    summary["outputs"] = {
        "full_csv": str(full_csv),
        "metadata_only_csv": str(lite_csv),
        "natural_evidence_only_csv": str(natural_csv),
        "natural_evidence_only_metadata_csv": str(natural_lite_csv),
        "all_available_natural_csv": str(all_natural_csv),
        "all_available_natural_metadata_csv": str(all_natural_lite_csv),
        "isoform_summary_csv": str(iso_csv),
        "hgvs_parse_validation_csv": str(validation_csv),
        "fasta": str(fasta),
        "vocab_json": str(vocab),
        "sequences_json": str(seq_json),
        "readme": str(readme),
    }
    summary["hgvs_parse_validation_status_counts"] = validation["validation_status"].value_counts(dropna=False).to_dict()
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    readme.write_text(
        "\n".join(
            [
                "# TF Isoform Benign/Control Variants, Ortholog Tier 4 Fill",
                "",
                f"Generated: {DATE_TAG}",
                "",
                "This directory supersedes the earlier 10-per-isoform table that used a purely conservative synthetic fallback.",
                "",
                "Selection order:",
                "",
                "1. Natural exact-mapped controls from ClinVar, gnomAD, and MaveDB.",
                "2. Tier 4 ortholog-supported synthetic controls.",
                "3. Tier 5 legacy conservative fallback controls only for slots still uncovered by the first two sources.",
                "",
                "The ortholog-supported rows are evolutionarily supported synthetic controls, not clinical benign variants.",
                "The legacy fallback rows are retained only to preserve the exact 10 variants per isoform contract.",
                "",
                "## Counts",
                "",
                f"- Isoforms: {summary['n_isoforms']}",
                f"- Variants per isoform: {summary['variants_per_isoform']}",
                f"- Total variants: {summary['n_variants']}",
                f"- Natural selected rows: {summary['natural_selected_rows']}",
                f"- Ortholog-supported selected rows: {summary['ortholog_supported_selected_rows']}",
                f"- Tier 5 legacy conservative fallback rows: {summary['legacy_conservative_fallback_rows']}",
                f"- Isoforms with no legacy fallback needed: {summary['isoforms_with_no_legacy_fallback_needed']}",
                f"- Isoforms with at least one ortholog-supported selected row: {summary['isoforms_with_ortholog_supported_selected_rows']}",
                f"- HGVS validation status counts: {summary['hgvs_parse_validation_status_counts']}",
                "",
                "## Files",
                "",
                f"- `{full_csv.name}`: full table including WT and mutant protein sequences.",
                f"- `{lite_csv.name}`: metadata-only table.",
                f"- `{natural_csv.name}`: natural rows selected into the capped table.",
                f"- `{all_natural_csv.name}`: all exact-mapped natural evidence rows before capping.",
                f"- `{iso_csv.name}`: per-isoform source mix summary.",
                f"- `{validation_csv.name}`: HGVS and sequence round-trip validation.",
                f"- `{fasta.name}`: mutant protein FASTA keyed by `variant_uid`.",
                f"- `{vocab.name}`: ordered `variant_uid` list.",
                f"- `{seq_json.name}`: `variant_uid -> mutant_sequence` mapping.",
                f"- `{summary_json.name}`: machine-readable run summary.",
                "",
                "## Caveat",
                "",
                "Do not pool these rows as one homogeneous benign label. Use `label_tier`, `evidence_source`,",
                "`synthetic_ortholog_flag`, and `synthetic_conservative_flag` to stratify analyses.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    old_full_path = args.old_dir / "tf_isoform_benign_controls_10_per_isoform_20260517.csv"
    all_natural_path = args.old_dir / "tf_isoform_benign_controls_all_available_natural_20260517.csv"
    ortholog_path = args.ortholog_dir / "tf_isoform_ortholog_supported_controls_20260517.csv"

    isoforms = metadata_isoforms(args.metadata)
    all_natural = sort_candidates(read_csv(all_natural_path))
    ortholog = coerce_ortholog_rows(read_csv(ortholog_path))
    old_full = read_csv(old_full_path)
    legacy = coerce_legacy_rows(
        old_full.loc[old_full.get("synthetic_conservative_flag", False).astype(str).str.lower().isin({"true", "1"})]
    )

    natural_by_iso = {k: v.copy() for k, v in all_natural.groupby("isoform_embedding_id", sort=False)}
    ortholog_by_iso = {k: v.copy() for k, v in ortholog.groupby("isoform_embedding_id", sort=False)}
    legacy_by_iso = {k: v.copy() for k, v in legacy.groupby("isoform_embedding_id", sort=False)}

    selected: list[dict[str, Any]] = []
    iso_summary: list[dict[str, Any]] = []
    for _, iso_row in isoforms.iterrows():
        iso = safe_str(iso_row["isoform_embedding_id"])
        chosen: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        source_counts = {"natural": 0, "ortholog": 0, "legacy": 0}

        def add_from(frame: pd.DataFrame | None, source: str) -> None:
            if frame is None or len(chosen) >= args.per_isoform:
                return
            for rec in frame.to_dict(orient="records"):
                if len(chosen) >= args.per_isoform:
                    break
                key = mutation_key(rec)
                if key in seen:
                    continue
                seen.add(key)
                chosen.append(rec)
                source_counts[source] += 1

        add_from(natural_by_iso.get(iso), "natural")
        add_from(ortholog_by_iso.get(iso), "ortholog")
        add_from(legacy_by_iso.get(iso), "legacy")
        if len(chosen) < args.per_isoform:
            generated = supplemental_tier5_rows(iso_row, args.per_isoform - len(chosen), seen)
            for rec in generated:
                key = mutation_key(rec)
                if key in seen:
                    continue
                seen.add(key)
                chosen.append(rec)
                source_counts["legacy"] += 1
        if len(chosen) != args.per_isoform:
            raise RuntimeError(f"{iso}: selected {len(chosen)} rows, expected {args.per_isoform}")

        for rank, rec in enumerate(chosen, start=1):
            rec = dict(rec)
            rec["selection_rank"] = rank
            rec["variant_uid"] = f"{iso}__benign_ctrl10_orthologtier4_{rank:02d}_{safe_str(rec['mutation'])}"
            selected.append(rec)

        iso_summary.append(
            {
                "isoform_embedding_id": iso,
                "gene": iso_row["gene_symbol"],
                "isoform": iso_row["isoform_id"],
                "label_status": iso_row["label_status"],
                "n_cells": iso_row["n_cells"],
                "protein_aa_length": iso_row["protein_aa_length"],
                "selected_variants": len(chosen),
                "natural_selected": source_counts["natural"],
                "ortholog_supported_selected": source_counts["ortholog"],
                "legacy_conservative_selected": source_counts["legacy"],
                "natural_candidates_available": int(len(natural_by_iso.get(iso, []))),
                "ortholog_supported_candidates_available": int(len(ortholog_by_iso.get(iso, []))),
                "legacy_candidates_available": int(len(legacy_by_iso.get(iso, []))),
            }
        )

    reference_cols = list(read_csv(old_full_path).columns)
    final = align_columns(pd.DataFrame(selected), reference_cols)
    final = final.sort_values(["isoform_embedding_id", "selection_rank"]).reset_index(drop=True)
    iso_summary_df = pd.DataFrame(iso_summary).sort_values("isoform_embedding_id")
    natural_selected = final.loc[
        ~final["synthetic_ortholog_flag"].astype(str).str.lower().isin({"true", "1"})
        & ~final["synthetic_conservative_flag"].astype(str).str.lower().isin({"true", "1"})
    ].copy()

    summary = {
        "status": "ok",
        "date": DATE_TAG,
        "question": "Can we provide 10 benign/control variants for every local HopTF TF isoform using ortholog-supported tier 4 rows before any legacy fallback?",
        "answer": "Yes. Natural evidence is used first, ortholog-supported tier 4 rows fill the next slots, and tier 5 legacy conservative fallback rows are retained only where needed to preserve exactly 10 variants per isoform.",
        "metadata": str(args.metadata),
        "source_files": {
            "all_available_natural": str(all_natural_path),
            "ortholog_supported_tier4": str(ortholog_path),
            "tier5_legacy_conservative_fallback": str(old_full_path),
        },
        "n_isoforms": int(isoforms["isoform_embedding_id"].nunique()),
        "variants_per_isoform": int(args.per_isoform),
        "n_variants": int(len(final)),
        "isoforms_with_exactly_requested_count": int((iso_summary_df["selected_variants"] == args.per_isoform).sum()),
        "natural_selected_rows": int(iso_summary_df["natural_selected"].sum()),
        "ortholog_supported_selected_rows": int(iso_summary_df["ortholog_supported_selected"].sum()),
        "legacy_conservative_fallback_rows": int(iso_summary_df["legacy_conservative_selected"].sum()),
        "isoforms_with_no_legacy_fallback_needed": int((iso_summary_df["legacy_conservative_selected"] == 0).sum()),
        "isoforms_with_ortholog_supported_selected_rows": int((iso_summary_df["ortholog_supported_selected"] > 0).sum()),
        "all_available_natural_rows": int(len(all_natural)),
        "all_available_natural_isoforms": int(all_natural["isoform_embedding_id"].nunique()),
        "ortholog_supported_available_rows": int(len(ortholog)),
        "ortholog_supported_available_isoforms": int(ortholog["isoform_embedding_id"].nunique()),
        "source_tier_counts": final["label_tier"].value_counts(dropna=False).to_dict(),
        "label_status_counts": final["label_status"].value_counts(dropna=False).to_dict(),
        "limitations": [
            "Ortholog-supported rows are synthetic evolutionary-support controls, not clinical benign labels.",
            "Tier 5 legacy conservative fallback rows remain only where natural and ortholog-supported sources do not provide enough rows for the exact 10-per-isoform contract.",
            "Use label_tier/evidence_source to stratify analyses; do not pool all rows as equivalent molecularly neutral variants.",
        ],
    }

    write_outputs(args.outdir, final, natural_selected, all_natural, iso_summary_df, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
