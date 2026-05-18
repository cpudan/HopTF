#!/usr/bin/env python3
"""Build ortholog-supported synthetic benign/control variants for HopTF.

This pipeline upgrades the Tier 4 fallback controls from purely conservative
substitutions to substitutions supported by amino acids observed at aligned
ortholog positions. It is intentionally conservative:

1. Map each local HopTF isoform sequence to the best same-gene Ensembl human
   protein.
2. Extract high-confidence Ensembl Compara orthologs for a fixed species panel.
3. Align local HopTF isoforms directly to ortholog proteins.
4. Nominate substitutions only when the ortholog residue aligns to the local
   HopTF residue, differs from human WT, is chemically plausible, and passes
   the existing sensitive-window filters.
5. If Pfam annotations are provided, require the ortholog protein domain
   architecture to match the mapped human Ensembl protein.

The output rows are not clinical benign labels. They are synthetic
evolutionary-support controls and must stay separate from natural ClinVar,
gnomAD, and MaveDB evidence tiers.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

try:
    from Bio.Align import PairwiseAligner
except Exception as exc:  # pragma: no cover - handled at runtime on cluster
    PairwiseAligner = None
    BIOPYTHON_IMPORT_ERROR = exc
else:
    BIOPYTHON_IMPORT_ERROR = None


REPO = Path(__file__).resolve().parents[1]
DEFAULT_METADATA = REPO / "data/processed/linear_probe/tfatlas_subsample/PERTURBATION_METADATA_hard_local_subsample.csv"
DEFAULT_COMPARA = Path("/gpfs/commons/groups/knowles_lab/data/cross_species/ensembl/ensembl_compara_115")
DEFAULT_PEP_DIR = Path(
    "/gpfs/commons/groups/knowles_lab/dmeyer/hoptf/ortholog_supported_controls_20260517/ensembl_release_115_pep"
)
DEFAULT_OUTDIR = REPO / "data/processed/variant_controls/tf_isoform_ortholog_supported_controls_20260517"
DEFAULT_PFAM = Path("/gpfs/commons/groups/knowles_lab/data/pfam_db/Pfam-A.hmm")
DATE_TAG = "2026-05-17"

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
FASTA_FIELD_RE = re.compile(r"([A-Za-z_]+):([^\s]+)")


DEFAULT_SPECIES = [
    "pan_troglodytes",
    "gorilla_gorilla",
    "pongo_abelii",
    "macaca_mulatta",
    "mus_musculus",
    "rattus_norvegicus",
    "canis_lupus_familiaris",
    "bos_taurus",
    "monodelphis_domestica",
    "gallus_gallus",
    "xenopus_tropicalis",
    "danio_rerio",
]


@dataclass
class ProteinRecord:
    protein_id: str
    protein_id_version: str
    gene_id: str
    gene_id_version: str
    transcript_id: str
    gene_symbol: str
    species: str
    description: str
    sequence: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    p.add_argument("--compara-dir", type=Path, default=DEFAULT_COMPARA)
    p.add_argument("--pep-dir", type=Path, default=DEFAULT_PEP_DIR)
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    p.add_argument("--pfam-domain-table", type=Path, default=None)
    p.add_argument("--species", nargs="*", default=DEFAULT_SPECIES)
    p.add_argument("--per-isoform", type=int, default=10)
    p.add_argument("--min-human-identity", type=float, default=0.80)
    p.add_argument("--min-human-coverage", type=float, default=0.70)
    p.add_argument("--min-ortholog-identity", type=float, default=0.25)
    p.add_argument("--min-support-species", type=int, default=1)
    p.add_argument("--max-isoforms", type=int, default=None, help="Smoke-test limit.")
    p.add_argument("--require-domain-architecture", action="store_true")
    p.add_argument("--reuse-ortholog-table", action="store_true")
    p.add_argument("--write-candidate-proteins-only", action="store_true")
    return p.parse_args()


def safe_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value)


def strip_version(stable_id: str) -> str:
    return safe_str(stable_id).split(".", 1)[0]


def open_text(path: Path):
    return gzip.open(path, "rt", encoding="utf-8", errors="replace") if path.suffix == ".gz" else path.open("rt", encoding="utf-8")


def stable_fraction(text: str) -> float:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) / float(16**16)


def mutate_sequence(seq: str, pos_1based: int, alt: str) -> str:
    return seq[: pos_1based - 1] + alt + seq[pos_1based:]


def make_hgvs_p(ref: str, pos: int, alt: str) -> str:
    return f"p.{AA1_TO_AA3[ref]}{pos}{AA1_TO_AA3[alt]}"


def parse_hgvs_p(hgvs_p: str) -> dict[str, Any]:
    match = HGVS_P_RE.match(safe_str(hgvs_p).strip())
    if not match:
        return {"status": "parse_failed", "ref": "", "pos": np.nan, "alt": ""}
    ref3, pos, alt3 = match.groups()
    return {"status": "ok", "ref": AA3_TO_AA1.get(ref3, ""), "pos": int(pos), "alt": AA3_TO_AA1.get(alt3, "")}


def parse_fasta_header(header: str, species: str, sequence: str) -> ProteinRecord:
    parts = header.split(None, 1)
    protein_id_version = parts[0]
    rest = parts[1] if len(parts) > 1 else ""
    fields = dict(FASTA_FIELD_RE.findall(rest))
    gene_id_version = fields.get("gene", "")
    transcript_id = fields.get("transcript", "")
    gene_symbol = fields.get("gene_symbol", "")
    return ProteinRecord(
        protein_id=strip_version(protein_id_version),
        protein_id_version=protein_id_version,
        gene_id=strip_version(gene_id_version),
        gene_id_version=gene_id_version,
        transcript_id=strip_version(transcript_id),
        gene_symbol=gene_symbol,
        species=species,
        description=rest,
        sequence=sequence.replace("*", "").upper(),
    )


def iter_fasta(path: Path, species: str) -> Iterable[ProteinRecord]:
    name = None
    chunks: list[str] = []
    with open_text(path) as handle:
        for raw in handle:
            line = raw.rstrip("\n")
            if line.startswith(">"):
                if name is not None:
                    yield parse_fasta_header(name, species, "".join(chunks))
                name = line[1:]
                chunks = []
            elif line:
                chunks.append(line.strip())
    if name is not None:
        yield parse_fasta_header(name, species, "".join(chunks))


def find_peptide_fastas(pep_dir: Path) -> dict[str, Path]:
    fastas: dict[str, Path] = {}
    for path in pep_dir.glob("*.pep.all.fa.gz"):
        species = path.name.split(".", 1)[0].lower()
        species = re.sub(r"([a-z])([A-Z])", r"\1_\2", species).lower()
        if path.name.startswith("Homo_sapiens"):
            species = "homo_sapiens"
        elif path.name.startswith("Pan_troglodytes"):
            species = "pan_troglodytes"
        elif path.name.startswith("Gorilla_gorilla"):
            species = "gorilla_gorilla"
        elif path.name.startswith("Pongo_abelii"):
            species = "pongo_abelii"
        elif path.name.startswith("Macaca_mulatta"):
            species = "macaca_mulatta"
        elif path.name.startswith("Mus_musculus"):
            species = "mus_musculus"
        elif path.name.startswith("Rattus_norvegicus"):
            species = "rattus_norvegicus"
        elif path.name.startswith("Canis_lupus_familiaris"):
            species = "canis_lupus_familiaris"
        elif path.name.startswith("Bos_taurus"):
            species = "bos_taurus"
        elif path.name.startswith("Monodelphis_domestica"):
            species = "monodelphis_domestica"
        elif path.name.startswith("Gallus_gallus"):
            species = "gallus_gallus"
        elif path.name.startswith("Xenopus_tropicalis"):
            species = "xenopus_tropicalis"
        elif path.name.startswith("Danio_rerio"):
            species = "danio_rerio"
        fastas[species] = path
    return fastas


def make_aligner() -> PairwiseAligner:
    if PairwiseAligner is None:
        raise RuntimeError(
            "Biopython is required for ortholog-supported alignment. "
            f"Original import error: {BIOPYTHON_IMPORT_ERROR!r}"
        )
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 2.0
    aligner.mismatch_score = -1.0
    aligner.open_gap_score = -8.0
    aligner.extend_gap_score = -0.5
    return aligner


def best_alignment(seq_a: str, seq_b: str, aligner: PairwiseAligner) -> tuple[dict[int, int], float, float]:
    aln = aligner.align(seq_a, seq_b)[0]
    mapping: dict[int, int] = {}
    matches = 0
    aligned_pairs = 0
    for (a0, a1), (b0, b1) in zip(aln.aligned[0], aln.aligned[1]):
        span = min(a1 - a0, b1 - b0)
        for offset in range(span):
            ai = int(a0 + offset)
            bi = int(b0 + offset)
            mapping[ai + 1] = bi + 1
            aligned_pairs += 1
            if seq_a[ai] == seq_b[bi]:
                matches += 1
    identity = matches / aligned_pairs if aligned_pairs else 0.0
    coverage = aligned_pairs / max(1, len(seq_a))
    return mapping, identity, coverage


def mark_sensitive_windows(seq: str) -> set[int]:
    avoid: set[int] = set()
    if not seq:
        return avoid
    for match in re.finditer(r"C.{2,4}C.{8,14}H.{3,6}H", seq):
        lo = max(1, match.start() - 4)
        hi = min(len(seq), match.end() + 4)
        avoid.update(range(lo, hi + 1))
    for i in range(0, max(0, len(seq) - 7)):
        window = seq[i : i + 8]
        if sum(aa in "KR" for aa in window) >= 5:
            avoid.update(range(i + 1, i + 9))
    for i in range(0, max(0, len(seq) - 21)):
        window = seq[i : i + 21]
        if sum(window[j] == "L" for j in range(0, 21, 7)) >= 3:
            avoid.update(range(i + 1, i + 22))
    return avoid


def local_low_complexity(seq: str, pos_1based: int) -> bool:
    lo = max(0, pos_1based - 8)
    hi = min(len(seq), pos_1based + 7)
    window = seq[lo:hi]
    if len(window) < 6:
        return False
    return max(window.count(aa) for aa in set(window)) / len(window) >= 0.55


def load_hoptf_metadata(path: Path, max_isoforms: int | None) -> pd.DataFrame:
    df = pd.read_csv(path)
    is_control = df["is_control"].map(lambda x: safe_str(x).strip().lower() in {"true", "1", "yes"})
    df = df[(~is_control) & df["sequence"].notna()].copy()
    if max_isoforms:
        df = df.head(max_isoforms).copy()
    return df


def load_proteins(pep_dir: Path, species: list[str]) -> tuple[dict[str, list[ProteinRecord]], dict[str, dict[str, list[ProteinRecord]]]]:
    fastas = find_peptide_fastas(pep_dir)
    needed = ["homo_sapiens"] + species
    missing = [sp for sp in needed if sp not in fastas]
    if missing:
        raise FileNotFoundError(f"Missing peptide FASTA files for species: {missing}; pep_dir={pep_dir}")

    by_species_gene: dict[str, dict[str, list[ProteinRecord]]] = {}
    human_by_symbol: dict[str, list[ProteinRecord]] = defaultdict(list)
    for sp in needed:
        records_by_gene: dict[str, list[ProteinRecord]] = defaultdict(list)
        for rec in iter_fasta(fastas[sp], sp):
            if not rec.sequence or not rec.gene_id:
                continue
            records_by_gene[rec.gene_id].append(rec)
            if sp == "homo_sapiens" and rec.gene_symbol:
                human_by_symbol[rec.gene_symbol].append(rec)
        by_species_gene[sp] = records_by_gene
    return human_by_symbol, by_species_gene


def choose_longest(records: list[ProteinRecord]) -> ProteinRecord | None:
    if not records:
        return None
    return sorted(records, key=lambda r: (len(r.sequence), r.protein_id), reverse=True)[0]


def map_isoforms_to_human(df: pd.DataFrame, human_by_symbol: dict[str, list[ProteinRecord]], aligner: PairwiseAligner, args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        seq = safe_str(row["sequence"]).upper()
        gene = safe_str(row["gene_symbol"])
        candidates = human_by_symbol.get(gene, [])
        best: dict[str, Any] | None = None
        for rec in candidates:
            if seq == rec.sequence:
                mapping = {i: i for i in range(1, len(seq) + 1)}
                identity = 1.0
                coverage = 1.0
            else:
                mapping, identity, coverage = best_alignment(seq, rec.sequence, aligner)
            score = (identity, coverage, len(mapping), len(rec.sequence))
            if best is None or score > best["score"]:
                best = {
                    "score": score,
                    "record": rec,
                    "identity": identity,
                    "coverage": coverage,
                    "mapped_positions": len(mapping),
                }
        status = "ok"
        rec = best["record"] if best else None
        if rec is None:
            status = "no_same_gene_ensembl_protein"
        elif best["identity"] < args.min_human_identity or best["coverage"] < args.min_human_coverage:
            status = "low_confidence_human_isoform_alignment"
        rows.append(
            {
                "isoform_embedding_id": row["isoform_embedding_id"],
                "gene_symbol": gene,
                "isoform_id": row.get("isoform_id", ""),
                "protein_aa_length": row.get("protein_aa_length", ""),
                "label_status": row.get("label_status", ""),
                "n_cells": row.get("n_cells", ""),
                "human_mapping_status": status,
                "human_ensembl_gene_id": rec.gene_id if rec else "",
                "human_ensembl_gene_id_version": rec.gene_id_version if rec else "",
                "human_ensembl_protein_id": rec.protein_id if rec else "",
                "human_ensembl_protein_id_version": rec.protein_id_version if rec else "",
                "human_ensembl_transcript_id": rec.transcript_id if rec else "",
                "human_alignment_identity": best["identity"] if best else 0.0,
                "human_alignment_coverage": best["coverage"] if best else 0.0,
                "human_mapped_positions": best["mapped_positions"] if best else 0,
            }
        )
    return pd.DataFrame(rows)


def load_genome_db_ids(compara_dir: Path, species: list[str]) -> dict[str, int]:
    ids: dict[str, int] = {}
    with open_text(compara_dir / "genome_db.txt.gz") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3 and parts[2] in species + ["homo_sapiens"]:
                ids[parts[2]] = int(parts[0])
    return ids


def build_ortholog_table(mapping: pd.DataFrame, compara_dir: Path, species: list[str], out_path: Path) -> pd.DataFrame:
    genome_db = load_genome_db_ids(compara_dir, species)
    selected_genome_ids = set(genome_db.values())
    human_genome_id = genome_db["homo_sapiens"]
    wanted_human_genes = set(mapping.loc[mapping["human_mapping_status"] == "ok", "human_ensembl_gene_id"].dropna().map(strip_version))

    gene_members: dict[int, dict[str, Any]] = {}
    human_member_ids: set[int] = set()
    print(f"Loading selected gene members for {len(wanted_human_genes)} human genes and {len(species)} ortholog species...", flush=True)
    with open_text(compara_dir / "gene_member.txt.gz") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for parts in reader:
            if len(parts) < 14:
                continue
            gene_member_id = int(parts[0])
            stable_id = strip_version(parts[1])
            genome_id = int(parts[5])
            if genome_id not in selected_genome_ids:
                continue
            display_label = parts[13] if len(parts) > 13 else ""
            if genome_id == human_genome_id and stable_id not in wanted_human_genes:
                continue
            gene_members[gene_member_id] = {
                "gene_member_id": gene_member_id,
                "gene_stable_id": stable_id,
                "genome_db_id": genome_id,
                "display_label": display_label,
            }
            if genome_id == human_genome_id:
                human_member_ids.add(gene_member_id)
    print(f"Loaded {len(gene_members)} selected gene members; human members={len(human_member_ids)}", flush=True)

    homology_ids: set[int] = set()
    human_homology_members: dict[int, list[dict[str, Any]]] = defaultdict(list)
    print("Pass 1 over homology_member.txt.gz: finding homologies involving mapped human TF genes...", flush=True)
    with open_text(compara_dir / "homology_member.txt.gz") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for parts in reader:
            if len(parts) < 7:
                continue
            gene_member_id = int(parts[1])
            if gene_member_id not in human_member_ids:
                continue
            homology_id = int(parts[0])
            homology_ids.add(homology_id)
            human_homology_members[homology_id].append(
                {
                    "human_gene_member_id": gene_member_id,
                    "human_seq_member_id": parts[2],
                    "human_cigar_line": parts[3],
                    "human_perc_cov": parts[4],
                    "human_perc_id": parts[5],
                    "human_perc_pos": parts[6],
                }
            )
    print(f"Candidate homology ids: {len(homology_ids)}", flush=True)

    homology_meta: dict[int, dict[str, Any]] = {}
    allowed_descriptions = {"ortholog_one2one", "ortholog_one2many", "ortholog_many2many"}
    print("Loading homology metadata...", flush=True)
    with open_text(compara_dir / "homology.txt.gz") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for parts in reader:
            if len(parts) < 15:
                continue
            homology_id = int(parts[0])
            if homology_id not in homology_ids:
                continue
            description = parts[2]
            if description not in allowed_descriptions:
                continue
            high_conf = parts[3]
            if high_conf != "1":
                continue
            homology_meta[homology_id] = {
                "method_link_species_set_id": parts[1],
                "description": description,
                "is_high_confidence": high_conf,
                "goc_score": parts[4],
                "wga_coverage": parts[5],
            }
    keep_homology_ids = set(homology_meta)
    print(f"High-confidence ortholog homologies retained: {len(keep_homology_ids)}", flush=True)

    ortholog_members: dict[int, list[dict[str, Any]]] = defaultdict(list)
    print("Pass 2 over homology_member.txt.gz: collecting selected ortholog species members...", flush=True)
    with open_text(compara_dir / "homology_member.txt.gz") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for parts in reader:
            if len(parts) < 7:
                continue
            homology_id = int(parts[0])
            if homology_id not in keep_homology_ids:
                continue
            gene_member_id = int(parts[1])
            member = gene_members.get(gene_member_id)
            if not member or member["genome_db_id"] == human_genome_id:
                continue
            ortholog_members[homology_id].append(
                {
                    "ortholog_gene_member_id": gene_member_id,
                    "ortholog_seq_member_id": parts[2],
                    "ortholog_cigar_line": parts[3],
                    "ortholog_perc_cov": parts[4],
                    "ortholog_perc_id": parts[5],
                    "ortholog_perc_pos": parts[6],
                }
            )

    genome_id_to_species = {v: k for k, v in genome_db.items()}
    rows: list[dict[str, Any]] = []
    for hid in keep_homology_ids:
        meta = homology_meta[hid]
        for human in human_homology_members.get(hid, []):
            h_member = gene_members.get(human["human_gene_member_id"])
            if not h_member:
                continue
            for orth in ortholog_members.get(hid, []):
                o_member = gene_members.get(orth["ortholog_gene_member_id"])
                if not o_member:
                    continue
                rows.append(
                    {
                        "homology_id": hid,
                        **meta,
                        **human,
                        **orth,
                        "human_gene_stable_id": h_member["gene_stable_id"],
                        "human_gene_symbol": h_member["display_label"],
                        "ortholog_gene_stable_id": o_member["gene_stable_id"],
                        "ortholog_gene_symbol": o_member["display_label"],
                        "ortholog_species": genome_id_to_species[o_member["genome_db_id"]],
                        "ortholog_genome_db_id": o_member["genome_db_id"],
                    }
                )
    result = pd.DataFrame(rows).drop_duplicates()
    result.to_csv(out_path, sep="\t", index=False)
    return result


def load_domain_architectures(path: Path | None) -> dict[str, tuple[str, ...]]:
    if path is None or not path.exists():
        return {}
    df = pd.read_csv(path, sep="\t")
    if df.empty:
        return {}
    required = {"protein_id", "pfam_acc", "ali_start"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Pfam domain table missing columns: {sorted(missing)}")
    df = df.sort_values(["protein_id", "ali_start", "ali_end" if "ali_end" in df.columns else "ali_start"])
    return {
        protein_id: tuple(group["pfam_acc"].astype(str).tolist())
        for protein_id, group in df.groupby("protein_id", sort=False)
    }


def write_candidate_protein_fasta(
    mapping: pd.DataFrame,
    orthologs: pd.DataFrame,
    by_species_gene: dict[str, dict[str, list[ProteinRecord]]],
    out_path: Path,
) -> pd.DataFrame:
    records: dict[str, ProteinRecord] = {}
    for _, row in mapping[mapping["human_mapping_status"] == "ok"].iterrows():
        gene = row["human_ensembl_gene_id"]
        protein_id = row["human_ensembl_protein_id"]
        for rec in by_species_gene["homo_sapiens"].get(gene, []):
            if rec.protein_id == protein_id:
                records[f"homo_sapiens|{rec.protein_id}"] = rec
                break
    for _, row in orthologs.iterrows():
        sp = row["ortholog_species"]
        gene = row["ortholog_gene_stable_id"]
        rec = choose_longest(by_species_gene.get(sp, {}).get(gene, []))
        if rec:
            records[f"{sp}|{rec.protein_id}"] = rec

    rows: list[dict[str, Any]] = []
    with out_path.open("w", encoding="utf-8") as handle:
        for key, rec in sorted(records.items()):
            handle.write(f">{key} gene:{rec.gene_id} gene_symbol:{rec.gene_symbol} transcript:{rec.transcript_id}\n")
            for i in range(0, len(rec.sequence), 80):
                handle.write(rec.sequence[i : i + 80] + "\n")
            rows.append(
                {
                    "protein_key": key,
                    "species": rec.species,
                    "protein_id": rec.protein_id,
                    "gene_id": rec.gene_id,
                    "gene_symbol": rec.gene_symbol,
                    "sequence_length": len(rec.sequence),
                }
            )
    return pd.DataFrame(rows)


def support_rows_for_isoform(
    iso_row: pd.Series,
    mapping_row: pd.Series,
    orthologs: pd.DataFrame,
    by_species_gene: dict[str, dict[str, list[ProteinRecord]]],
    domain_arch: dict[str, tuple[str, ...]],
    args: argparse.Namespace,
    aligner: PairwiseAligner,
) -> list[dict[str, Any]]:
    seq = safe_str(iso_row["sequence"]).upper()
    avoid = mark_sensitive_windows(seq)
    human_gene = mapping_row["human_ensembl_gene_id"]
    human_protein_id = mapping_row["human_ensembl_protein_id"]
    human_domain = domain_arch.get(human_protein_id)
    gene_orthologs = orthologs[orthologs["human_gene_stable_id"] == human_gene]
    support: dict[tuple[int, str, str], dict[str, Any]] = {}

    for _, orth in gene_orthologs.iterrows():
        sp = orth["ortholog_species"]
        rec = choose_longest(by_species_gene.get(sp, {}).get(orth["ortholog_gene_stable_id"], []))
        if rec is None:
            continue
        if args.require_domain_architecture:
            orth_domain = domain_arch.get(rec.protein_id)
            if not human_domain or not orth_domain or human_domain != orth_domain:
                continue
        mapping, identity, coverage = best_alignment(seq, rec.sequence, aligner)
        if identity < args.min_ortholog_identity:
            continue
        for pos, orth_pos in mapping.items():
            wt = seq[pos - 1]
            if wt not in STANDARD_AA or wt in EXCLUDED_SYNTHETIC_WT:
                continue
            if pos <= 5 or pos > len(seq) - 5 or pos in avoid or local_low_complexity(seq, pos):
                continue
            if orth_pos < 1 or orth_pos > len(rec.sequence):
                continue
            alt = rec.sequence[orth_pos - 1]
            if alt == wt or alt not in STANDARD_AA:
                continue
            if alt not in CONSERVATIVE_ALTS.get(wt, []):
                continue
            key = (pos, wt, alt)
            entry = support.setdefault(
                key,
                {
                    "aa_position": pos,
                    "aa_ref": wt,
                    "aa_alt": alt,
                    "support_species": [],
                    "support_gene_ids": [],
                    "support_protein_ids": [],
                    "ortholog_alignment_identities": [],
                    "ortholog_alignment_coverages": [],
                    "domain_architecture_match": bool(not args.require_domain_architecture or human_domain),
                    "human_pfam_architecture": ";".join(human_domain or ()),
                },
            )
            entry["support_species"].append(sp)
            entry["support_gene_ids"].append(rec.gene_id)
            entry["support_protein_ids"].append(rec.protein_id)
            entry["ortholog_alignment_identities"].append(identity)
            entry["ortholog_alignment_coverages"].append(coverage)

    rows: list[dict[str, Any]] = []
    for entry in support.values():
        species = sorted(set(entry["support_species"]))
        if len(species) < args.min_support_species:
            continue
        rows.append(
            {
                **entry,
                "support_species_count": len(species),
                "support_species": ";".join(species),
                "support_gene_ids": ";".join(sorted(set(entry["support_gene_ids"]))),
                "support_protein_ids": ";".join(sorted(set(entry["support_protein_ids"]))),
                "mean_ortholog_alignment_identity": float(np.mean(entry["ortholog_alignment_identities"])),
                "mean_ortholog_alignment_coverage": float(np.mean(entry["ortholog_alignment_coverages"])),
            }
        )
    rows.sort(
        key=lambda r: (
            -r["support_species_count"],
            -r["mean_ortholog_alignment_identity"],
            stable_fraction(f"{iso_row['isoform_embedding_id']}|{r['aa_position']}|{r['aa_ref']}>{r['aa_alt']}"),
        )
    )
    return rows


def add_hgvs_and_sequences(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = rows.copy()
    rows["mutation_shorthand"] = [
        f"{ref}{int(pos)}{alt}" for ref, pos, alt in zip(rows["aa_ref"], rows["aa_position"], rows["aa_alt"])
    ]
    rows["mutation"] = rows["mutation_shorthand"]
    rows["hgvs_p"] = [make_hgvs_p(ref, int(pos), alt) for ref, pos, alt in zip(rows["aa_ref"], rows["aa_position"], rows["aa_alt"])]
    rows["hgvs_protein"] = rows["hgvs_p"]
    rows["local_hgvs_p"] = rows["isoform_embedding_id"].astype(str) + ":" + rows["hgvs_p"].astype(str)
    rows["mutant_sequence"] = [mutate_sequence(seq, int(pos), alt) for seq, pos, alt in zip(rows["sequence"], rows["aa_position"], rows["aa_alt"])]

    validation_rows: list[dict[str, Any]] = []
    for _, row in rows.iterrows():
        parsed = parse_hgvs_p(row["hgvs_p"])
        status = parsed["status"]
        ref = parsed["ref"]
        pos = parsed["pos"]
        alt = parsed["alt"]
        ok = (
            status == "ok"
            and ref == row["aa_ref"]
            and alt == row["aa_alt"]
            and int(pos) == int(row["aa_position"])
            and row["sequence"][int(pos) - 1] == ref
            and row["mutant_sequence"][int(pos) - 1] == alt
        )
        validation_rows.append(
            {
                "variant_uid": row["variant_uid"],
                "hgvs_p": row["hgvs_p"],
                "parsed_ref": ref,
                "parsed_pos": pos,
                "parsed_alt": alt,
                "validation_status": "ok" if ok else "failed",
            }
        )
    return rows, pd.DataFrame(validation_rows)


def write_fasta(rows: pd.DataFrame, path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for _, row in rows.iterrows():
            handle.write(f">{row['variant_uid']}\n")
            seq = row["mutant_sequence"]
            for i in range(0, len(seq), 80):
                handle.write(seq[i : i + 80] + "\n")


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    metadata = load_hoptf_metadata(args.metadata, args.max_isoforms)
    aligner = make_aligner()
    human_by_symbol, by_species_gene = load_proteins(args.pep_dir, args.species)

    mapping_path = args.outdir / "ortholog_supported_isoform_human_mapping.csv"
    mapping = map_isoforms_to_human(metadata, human_by_symbol, aligner, args)
    mapping.to_csv(mapping_path, index=False)

    ortholog_path = args.outdir / "ortholog_supported_compara_ortholog_pairs.tsv"
    if args.reuse_ortholog_table and ortholog_path.exists():
        orthologs = pd.read_csv(ortholog_path, sep="\t")
    else:
        orthologs = build_ortholog_table(mapping, args.compara_dir, args.species, ortholog_path)

    candidate_fasta = args.outdir / "ortholog_supported_candidate_proteins.fasta"
    candidate_proteins = write_candidate_protein_fasta(mapping, orthologs, by_species_gene, candidate_fasta)
    candidate_proteins.to_csv(args.outdir / "ortholog_supported_candidate_proteins.csv", index=False)
    if args.write_candidate_proteins_only:
        report = {
            "status": "candidate_proteins_only",
            "date": DATE_TAG,
            "candidate_protein_fasta": str(candidate_fasta),
            "candidate_proteins": int(len(candidate_proteins)),
            "mapped_isoforms": int((mapping["human_mapping_status"] == "ok").sum()),
            "ortholog_pairs": int(len(orthologs)),
            "next_step": "Run hmmscan on candidate_proteins.fasta, parse domtblout, then rerun without --write-candidate-proteins-only.",
        }
        (args.outdir / "ortholog_supported_pipeline_report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        return

    domain_arch = load_domain_architectures(args.pfam_domain_table)
    rows: list[dict[str, Any]] = []
    mapping_by_iso = mapping.set_index("isoform_embedding_id", drop=False)
    for _, iso_row in metadata.iterrows():
        iso = iso_row["isoform_embedding_id"]
        if iso not in mapping_by_iso.index:
            continue
        map_row = mapping_by_iso.loc[iso]
        if map_row["human_mapping_status"] != "ok":
            continue
        support_rows = support_rows_for_isoform(iso_row, map_row, orthologs, by_species_gene, domain_arch, args, aligner)
        for rank, support in enumerate(support_rows[: args.per_isoform], start=1):
            mut = f"{support['aa_ref']}{support['aa_position']}{support['aa_alt']}"
            rows.append(
                {
                    "source_record_id": f"ortholog_supported:{iso}:{mut}",
                    "source_priority": 8,
                    "label": "putative_benign_control",
                    "label_tier": "Tier 4 ortholog-supported synthetic control",
                    "label_source": "ortholog residue support",
                    "evidence_source": "ensembl_compara_ortholog_alignment",
                    "evidence_tier": "Tier 4 ortholog-supported synthetic control",
                    "evidence_confidence": "synthetic_evolutionary_support",
                    "clinical_benign_flag": False,
                    "population_tolerated_flag": False,
                    "functional_neutral_flag": False,
                    "synthetic_ortholog_flag": True,
                    "synthetic_conservative_flag": False,
                    "include_final": True,
                    "exclusion_reason": "",
                    "selection_rank": rank,
                    "isoform_embedding_id": iso,
                    "gene": iso_row["gene_symbol"],
                    "gene_symbol": iso_row["gene_symbol"],
                    "isoform": iso_row.get("isoform_id", ""),
                    "isoform_id": iso_row.get("isoform_id", ""),
                    "label_status": iso_row.get("label_status", ""),
                    "n_cells": iso_row.get("n_cells", ""),
                    "protein_aa_length": iso_row.get("protein_aa_length", ""),
                    "sequence": iso_row["sequence"],
                    "human_ensembl_gene_id": map_row["human_ensembl_gene_id"],
                    "human_ensembl_protein_id": map_row["human_ensembl_protein_id"],
                    "human_alignment_identity": map_row["human_alignment_identity"],
                    "human_alignment_coverage": map_row["human_alignment_coverage"],
                    "selection_notes": (
                        "Synthetic substitution where the alternate residue is observed at the aligned position in "
                        "an Ensembl Compara ortholog, passes conservative chemistry and sensitive-window filters, "
                        "and is kept separate from natural benign/control evidence."
                    ),
                    "synthetic_rule_version": "ortholog_supported_v1_20260517",
                    **support,
                }
            )

    final = pd.DataFrame(rows)
    if final.empty:
        raise RuntimeError("No ortholog-supported synthetic controls were generated.")
    final["variant_uid"] = [
        f"ortholog_supported_{i:06d}_{iso}_{mut}"
        for i, (iso, mut) in enumerate(zip(final["isoform_embedding_id"], final["aa_ref"] + final["aa_position"].astype(str) + final["aa_alt"]))
    ]
    final, validation = add_hgvs_and_sequences(final)

    full_csv = args.outdir / "tf_isoform_ortholog_supported_controls_20260517.csv"
    metadata_csv = args.outdir / "tf_isoform_ortholog_supported_controls_20260517.metadata_only.csv"
    validation_csv = args.outdir / "tf_isoform_ortholog_supported_controls_20260517.hgvs_parse_validation.csv"
    fasta = args.outdir / "tf_isoform_ortholog_supported_controls_20260517.fasta"
    vocab = args.outdir / "tf_isoform_ortholog_supported_controls_20260517.vocab.json"
    seq_json = args.outdir / "tf_isoform_ortholog_supported_controls_20260517.sequences.json"
    summary_json = args.outdir / "tf_isoform_ortholog_supported_controls_20260517.summary.json"

    final.to_csv(full_csv, index=False)
    final.drop(columns=["sequence", "mutant_sequence"]).to_csv(metadata_csv, index=False)
    validation.to_csv(validation_csv, index=False)
    write_fasta(final, fasta)
    vocab.write_text(json.dumps(final["variant_uid"].tolist(), indent=2) + "\n")
    seq_json.write_text(json.dumps(dict(zip(final["variant_uid"], final["mutant_sequence"])), indent=2) + "\n")

    per_iso_counts = final.groupby("isoform_embedding_id").size()
    summary = {
        "status": "ok",
        "date": DATE_TAG,
        "n_isoforms_input": int(metadata["isoform_embedding_id"].nunique()),
        "n_isoforms_mapped_to_human_ensembl": int((mapping["human_mapping_status"] == "ok").sum()),
        "n_isoforms_with_ortholog_supported_controls": int(per_iso_counts.shape[0]),
        "n_isoforms_with_requested_count": int((per_iso_counts >= args.per_isoform).sum()),
        "per_isoform_requested": int(args.per_isoform),
        "n_variants": int(len(final)),
        "min_variants_per_represented_isoform": int(per_iso_counts.min()),
        "max_variants_per_represented_isoform": int(per_iso_counts.max()),
        "mean_support_species_count": float(final["support_species_count"].mean()),
        "min_support_species": int(args.min_support_species),
        "require_domain_architecture": bool(args.require_domain_architecture),
        "pfam_domain_table": str(args.pfam_domain_table) if args.pfam_domain_table else None,
        "outputs": {
            "full_csv": str(full_csv),
            "metadata_only_csv": str(metadata_csv),
            "hgvs_parse_validation_csv": str(validation_csv),
            "fasta": str(fasta),
            "vocab_json": str(vocab),
            "sequences_json": str(seq_json),
            "isoform_human_mapping_csv": str(mapping_path),
            "ortholog_pairs_tsv": str(ortholog_path),
            "candidate_proteins_fasta": str(candidate_fasta),
        },
        "limitations": [
            "Rows are synthetic evolutionary-support controls, not clinical benign labels.",
            "Amino-acid support comes from Ensembl Compara ortholog proteins and pairwise alignment to the local HopTF isoform.",
            "Known TF contact/interface exclusion is currently represented by conservative sequence-window heuristics unless external curated contact annotations are added.",
        ],
    }
    summary_json.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
