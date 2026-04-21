#!/usr/bin/env python3
"""Build a clean raw-counts AnnData object for SCP3357, with TF construct metadata."""

from __future__ import annotations

import argparse
import re
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy.io import mmread
from scipy.sparse import csr_matrix

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCP3357_DIR = REPO_ROOT / "data" / "raw" / "scp3357"
DEFAULT_OUTPUT_H5AD = REPO_ROOT / "data" / "processed" / "scp3357" / "SCP3357_raw_counts.h5ad"
DEFAULT_MORF_XLSX = REPO_ROOT / "data" / "reference" / "200102_tf_orf_library.xlsx"
MORF_XLSX_URL = (
    "https://media.addgene.org/cms/filer_public/5e/22/"
    "5e22c6a5-d186-4c54-95c0-8314db54bfbe/200102_tf_orf_library.xlsx"
)
XLSX_NS = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
CELL_REF_RE = re.compile(r"([A-Z]+)")

CODON_TABLE = {
    "TTT": "F",
    "TTC": "F",
    "TTA": "L",
    "TTG": "L",
    "TCT": "S",
    "TCC": "S",
    "TCA": "S",
    "TCG": "S",
    "TAT": "Y",
    "TAC": "Y",
    "TAA": "*",
    "TAG": "*",
    "TGT": "C",
    "TGC": "C",
    "TGA": "*",
    "TGG": "W",
    "CTT": "L",
    "CTC": "L",
    "CTA": "L",
    "CTG": "L",
    "CCT": "P",
    "CCC": "P",
    "CCA": "P",
    "CCG": "P",
    "CAT": "H",
    "CAC": "H",
    "CAA": "Q",
    "CAG": "Q",
    "CGT": "R",
    "CGC": "R",
    "CGA": "R",
    "CGG": "R",
    "ATT": "I",
    "ATC": "I",
    "ATA": "I",
    "ATG": "M",
    "ACT": "T",
    "ACC": "T",
    "ACA": "T",
    "ACG": "T",
    "AAT": "N",
    "AAC": "N",
    "AAA": "K",
    "AAG": "K",
    "AGT": "S",
    "AGC": "S",
    "AGA": "R",
    "AGG": "R",
    "GTT": "V",
    "GTC": "V",
    "GTA": "V",
    "GTG": "V",
    "GCT": "A",
    "GCC": "A",
    "GCA": "A",
    "GCG": "A",
    "GAT": "D",
    "GAC": "D",
    "GAA": "E",
    "GAG": "E",
    "GGT": "G",
    "GGC": "G",
    "GGA": "G",
    "GGG": "G",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert the SCP3357 raw counts bundle into a clean .h5ad file."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_SCP3357_DIR,
        help="Path to the downloaded SCP3357 directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_H5AD,
        help="Destination .h5ad path.",
    )
    parser.add_argument(
        "--morf-xlsx",
        type=Path,
        default=DEFAULT_MORF_XLSX,
        help="Local path for the Addgene MORF construct workbook.",
    )
    parser.add_argument(
        "--fasta-output",
        type=Path,
        default=None,
        help="Protein FASTA output path. Defaults next to the .h5ad file.",
    )
    return parser.parse_args()


def find_unique_file(root: Path, pattern: str) -> Path:
    matches = sorted(root.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No files matched {pattern!r} under {root}")
    if len(matches) > 1:
        raise RuntimeError(f"Expected one match for {pattern!r}, found {len(matches)}: {matches}")
    return matches[0]


def load_obs_table(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path, sep="\t")
    if "NAME" not in table.columns:
        raise ValueError(f"Expected NAME column in {path}")
    table = table[table["NAME"] != "TYPE"].copy()
    table = table.set_index("NAME")
    table.index.name = None
    return table


def load_counts(input_dir: Path) -> tuple[csr_matrix, pd.Index, pd.Index]:
    counts_dir = find_unique_file(input_dir, "expression/*/counts_matrix.mtx").parent
    matrix = mmread(counts_dir / "counts_matrix.mtx").tocsr().transpose().tocsr()
    matrix = matrix.astype(np.int32, copy=False)

    barcodes = pd.read_csv(
        counts_dir / "counts_barcodes.tsv",
        sep="\t",
        header=None,
        names=["cell_barcode"],
    )["cell_barcode"]
    genes = pd.read_csv(
        counts_dir / "counts_features.tsv",
        sep="\t",
        header=None,
        names=["gene_symbol"],
    )["gene_symbol"]

    return matrix, pd.Index(barcodes), pd.Index(genes, name="gene_symbol")


def derive_perturbation_gene(label: str) -> str:
    if label.startswith("BFP."):
        return "BFP"
    if label.startswith("None."):
        return "None"
    return label.split(".", 1)[0]


def derive_perturbation_class(label: str) -> str:
    if label.startswith("BFP."):
        return "control_bfp"
    if label.startswith("None."):
        return "control_none"
    return "tf"


def translate_orf_to_protein(orf_nt: str) -> str:
    orf_nt = orf_nt.upper()
    aa = []
    for i in range(0, len(orf_nt) - 2, 3):
        aa_code = CODON_TABLE.get(orf_nt[i : i + 3], "X")
        if aa_code == "*":
            break
        aa.append(aa_code)
    return "".join(aa)


def col_letters_to_index(col_letters: str) -> int:
    index = 0
    for char in col_letters:
        index = index * 26 + (ord(char) - ord("A") + 1)
    return index - 1


def parse_xlsx_shared_strings(zf: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in zf.namelist():
        return []

    root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
    shared_strings: list[str] = []
    for item in root.findall("a:si", XLSX_NS):
        shared_strings.append("".join(node.text or "" for node in item.iter("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t")))
    return shared_strings


def parse_xlsx_cell_value(cell: ET.Element, shared_strings: list[str]) -> str:
    cell_type = cell.attrib.get("t")
    value_node = cell.find("a:v", XLSX_NS)
    if value_node is None:
        return ""
    raw_value = value_node.text or ""
    if cell_type == "s":
        return shared_strings[int(raw_value)]
    return raw_value


def read_single_sheet_xlsx(path: Path) -> list[dict[str, str]]:
    with zipfile.ZipFile(path) as zf:
        shared_strings = parse_xlsx_shared_strings(zf)
        sheet_root = ET.fromstring(zf.read("xl/worksheets/sheet1.xml"))

    raw_rows: list[dict[int, str]] = []
    max_col = -1
    for row in sheet_root.findall(".//a:sheetData/a:row", XLSX_NS):
        row_values: dict[int, str] = {}
        for cell in row.findall("a:c", XLSX_NS):
            cell_ref = cell.attrib.get("r", "")
            match = CELL_REF_RE.match(cell_ref)
            if match is None:
                continue
            col_index = col_letters_to_index(match.group(1))
            row_values[col_index] = parse_xlsx_cell_value(cell, shared_strings)
            max_col = max(max_col, col_index)
        raw_rows.append(row_values)

    if not raw_rows:
        raise ValueError(f"No rows found in {path}")

    headers = [raw_rows[0].get(i, "") for i in range(max_col + 1)]
    records: list[dict[str, str]] = []
    for raw_row in raw_rows[1:]:
        if not raw_row:
            continue
        record = {headers[i]: raw_row.get(i, "") for i in range(len(headers))}
        records.append(record)
    return records


def ensure_morf_xlsx(path: Path) -> Path:
    if path.exists():
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        MORF_XLSX_URL,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    with urllib.request.urlopen(request) as response, path.open("wb") as handle:
        handle.write(response.read())
    return path


def build_tf_sequence_tables(
    perturbation_labels: list[str],
    morf_xlsx_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    morf_records = pd.DataFrame(read_single_sheet_xlsx(morf_xlsx_path))
    needed_cols = {
        "Name",
        "RefSeq Gene Name",
        "RefSeq and Gencode ID",
        "Barcode Sequence",
        "ORF sequence",
    }
    missing_cols = needed_cols.difference(morf_records.columns)
    if missing_cols:
        raise ValueError(f"MORF workbook missing columns: {sorted(missing_cols)}")

    label_to_gene = {label: derive_perturbation_gene(label) for label in perturbation_labels}
    gene_to_labels: dict[str, list[str]] = {}
    for label, gene in label_to_gene.items():
        gene_to_labels.setdefault(gene, []).append(label)
    for labels in gene_to_labels.values():
        labels.sort()

    tf_genes = sorted(gene for gene in gene_to_labels if gene not in {"BFP", "None"})
    construct_rows = morf_records[morf_records["RefSeq Gene Name"].isin(tf_genes)].copy()
    construct_rows["construct_id"] = construct_rows["Name"].astype(str).str.strip()
    construct_rows["gene_symbol"] = construct_rows["RefSeq Gene Name"].astype(str).str.strip()
    construct_rows["refseq_gencode_ids"] = construct_rows["RefSeq and Gencode ID"].astype(str).str.strip()
    construct_rows["morf_barcode"] = construct_rows["Barcode Sequence"].astype(str).str.strip()
    construct_rows["orf_nt_sequence"] = construct_rows["ORF sequence"].astype(str).str.strip().str.upper()
    construct_rows["orf_nt_length"] = construct_rows["orf_nt_sequence"].str.len().astype(int)
    construct_rows["protein_aa_sequence"] = construct_rows["orf_nt_sequence"].map(translate_orf_to_protein)
    construct_rows["protein_aa_length"] = construct_rows["protein_aa_sequence"].str.len().astype(int)

    construct_table_records: list[dict[str, object]] = []
    perturbation_map_records: list[dict[str, object]] = []

    ordered_constructs = construct_rows.sort_values(["gene_symbol", "construct_id"]).reset_index(drop=True)
    for gene, gene_df in ordered_constructs.groupby("gene_symbol", sort=True):
        labels = gene_to_labels[gene]
        construct_count = len(gene_df)
        label_count = len(labels)
        is_exact = construct_count == 1 and label_count == 1
        if is_exact:
            mapping_status = "exact_single_candidate"
            mapping_note = "Only one MORF construct matched this perturbation gene."
        else:
            mapping_status = "unresolved_multi_candidate_within_gene"
            mapping_note = (
                f"{construct_count} MORF constructs matched gene {gene}, while SCP3357 contains "
                f"{label_count} label(s) for that gene. The SCP3357 bundle does not expose MORF "
                "construct IDs or dial-out barcodes, so the label-to-construct assignment remains unresolved."
            )

        for rank_within_gene, (_, row) in enumerate(gene_df.iterrows(), start=1):
            construct_table_records.append(
                {
                    "construct_id": row["construct_id"],
                    "gene_symbol": gene,
                    "candidate_rank_within_gene": rank_within_gene,
                    "refseq_gencode_ids": row["refseq_gencode_ids"],
                    "morf_barcode": row["morf_barcode"],
                    "orf_nt_length": int(row["orf_nt_length"]),
                    "protein_aa_length": int(row["protein_aa_length"]),
                    "orf_nt_sequence": row["orf_nt_sequence"],
                    "protein_aa_sequence": row["protein_aa_sequence"],
                    "dataset_perturbation_ids": "|".join(labels),
                    "mapping_status": mapping_status,
                    "mapping_note": mapping_note,
                }
            )

    construct_table = pd.DataFrame(construct_table_records).sort_values(
        ["gene_symbol", "candidate_rank_within_gene", "construct_id"]
    ).reset_index(drop=True)

    by_gene = {
        gene: group.sort_values(["candidate_rank_within_gene", "construct_id"]).reset_index(drop=True)
        for gene, group in construct_table.groupby("gene_symbol", sort=True)
    }

    for label in sorted(perturbation_labels):
        gene = label_to_gene[label]
        perturbation_class = derive_perturbation_class(label)
        is_control = perturbation_class != "tf"
        if is_control:
            perturbation_map_records.append(
                {
                    "perturbation_id": label,
                    "gene_symbol": gene,
                    "perturbation_class": perturbation_class,
                    "is_control": True,
                    "has_candidate_sequence": False,
                    "candidate_construct_count": 0,
                    "candidate_construct_ids": "",
                    "candidate_protein_aa_lengths": "",
                    "mapping_status": "control_no_candidate_sequence",
                    "mapping_note": "Control perturbation; no MORF TF construct sequence is attached.",
                }
            )
            continue

        candidates = by_gene[gene]
        is_exact = len(candidates) == 1
        perturbation_map_records.append(
            {
                "perturbation_id": label,
                "gene_symbol": gene,
                "perturbation_class": perturbation_class,
                "is_control": False,
                "has_candidate_sequence": True,
                "candidate_construct_count": int(len(candidates)),
                "candidate_construct_ids": "|".join(candidates["construct_id"].tolist()),
                "candidate_protein_aa_lengths": "|".join(str(v) for v in candidates["protein_aa_length"].tolist()),
                "mapping_status": (
                    "exact_single_candidate"
                    if is_exact
                    else "unresolved_multi_candidate_within_gene"
                ),
                "mapping_note": (
                    "Only one MORF construct matched this perturbation gene."
                    if is_exact
                    else (
                        f"Gene {gene} has multiple candidate MORF constructs and SCP3357 does not expose "
                        "construct-level identifiers, so this label is linked to the candidate set rather "
                        "than a single exact sequence."
                    )
                ),
            }
        )

    perturbation_map = pd.DataFrame(perturbation_map_records).sort_values("perturbation_id").reset_index(drop=True)
    return construct_table, perturbation_map


def wrap_fasta_sequence(sequence: str, width: int = 60) -> str:
    return "\n".join(sequence[i : i + width] for i in range(0, len(sequence), width))


def write_candidate_fasta(construct_table: pd.DataFrame, fasta_output: Path) -> None:
    fasta_output.parent.mkdir(parents=True, exist_ok=True)
    with fasta_output.open("w", encoding="utf-8") as handle:
        for row in construct_table.itertuples(index=False):
            header = (
                f">{row.construct_id} "
                f"gene={row.gene_symbol} "
                f"rank_within_gene={row.candidate_rank_within_gene} "
                f"aa_length={row.protein_aa_length} "
                f"ref_ids={row.refseq_gencode_ids.replace(',', ';')} "
                f"dataset_labels={row.dataset_perturbation_ids.replace('|', ';')} "
                f"mapping_status={row.mapping_status}"
            )
            handle.write(header + "\n")
            handle.write(wrap_fasta_sequence(row.protein_aa_sequence) + "\n")


def table_to_uns_dict(table: pd.DataFrame) -> dict[str, list[object]]:
    uns_table: dict[str, list[object]] = {}
    for column in table.columns:
        series = table[column]
        if pd.api.types.is_bool_dtype(series):
            uns_table[column] = [bool(v) for v in series.tolist()]
        elif pd.api.types.is_integer_dtype(series):
            uns_table[column] = [int(v) for v in series.tolist()]
        else:
            uns_table[column] = [str(v) for v in series.tolist()]
    return uns_table


def build_anndata(
    input_dir: Path,
    morf_xlsx_path: Path,
    fasta_output: Path,
) -> ad.AnnData:
    matrix, barcodes, genes = load_counts(input_dir)

    obs = load_obs_table(input_dir / "metadata" / "metadata.tsv")
    umap_table = load_obs_table(input_dir / "cluster" / "cluster.txt")

    if not barcodes.equals(obs.index):
        raise ValueError("Counts barcodes do not match metadata row order.")
    if not barcodes.equals(umap_table.index):
        raise ValueError("Counts barcodes do not match cluster row order.")
    if list(umap_table.columns[:2]) != ["X", "Y"]:
        raise ValueError("Expected UMAP columns X and Y in cluster.txt.")
    if not genes.is_unique:
        raise ValueError("Gene symbols are not unique; please make var_names unique before writing.")

    obs["perturbation_id"] = obs["tf_isoform_unique"].astype(str)
    perturbation_labels = sorted(obs["perturbation_id"].unique())
    construct_table, perturbation_map = build_tf_sequence_tables(perturbation_labels, morf_xlsx_path)
    write_candidate_fasta(construct_table, fasta_output)

    perturbation_lookup = perturbation_map.set_index("perturbation_id")
    obs["perturbation_gene_symbol"] = obs["perturbation_id"].map(perturbation_lookup["gene_symbol"])
    obs["perturbation_is_control"] = obs["perturbation_id"].map(perturbation_lookup["is_control"]).astype(bool)
    obs["perturbation_has_candidate_sequence"] = (
        obs["perturbation_id"].map(perturbation_lookup["has_candidate_sequence"]).astype(bool)
    )
    obs["perturbation_candidate_construct_count"] = (
        obs["perturbation_id"].map(perturbation_lookup["candidate_construct_count"]).astype(np.int16)
    )
    obs["perturbation_sequence_mapping_status"] = obs["perturbation_id"].map(perturbation_lookup["mapping_status"])

    for column in obs.columns:
        if column in {
            "perturbation_is_control",
            "perturbation_has_candidate_sequence",
            "perturbation_candidate_construct_count",
        }:
            continue
        obs[column] = obs[column].astype("category")

    var = pd.DataFrame(index=genes)
    var["gene_short_name"] = genes.to_numpy()
    var["feature_name"] = genes.to_numpy()

    adata = ad.AnnData(X=matrix, obs=obs, var=var)
    adata.layers["counts"] = matrix.copy()
    adata.obsm["X_umap"] = umap_table.loc[barcodes, ["X", "Y"]].to_numpy(dtype=np.float32)
    adata.uns["dataset_id"] = "SCP3357"
    adata.uns["source"] = "Single Cell Portal"
    adata.uns["matrix_kind"] = "raw_counts"
    adata.uns["count_matrix_shape"] = list(matrix.shape)
    adata.uns["obs_source_file"] = "metadata/metadata.tsv"
    adata.uns["umap_source_file"] = "cluster/cluster.txt"
    adata.uns["tf_construct_candidates"] = table_to_uns_dict(construct_table)
    adata.uns["tf_perturbation_map"] = table_to_uns_dict(perturbation_map)
    adata.uns["tf_sequence_resources"] = {
        "obs_lookup_key": "perturbation_id",
        "original_obs_column": "tf_isoform_unique",
        "construct_table_key": "tf_construct_candidates",
        "perturbation_map_key": "tf_perturbation_map",
        "fasta_file": fasta_output.name,
        "sequence_space": "amino_acid",
        "construct_source": "Addgene MORF Collection",
        "construct_source_url": MORF_XLSX_URL,
        "construct_source_workbook": morf_xlsx_path.name,
        "mapping_caveat": (
            "Single-candidate genes are exact at the gene level. Multi-candidate genes retain an "
            "unresolved candidate set because SCP3357 does not expose MORF construct IDs or dial-out barcodes."
        ),
    }

    return adata


def sanitize_adata_strings_for_h5ad(adata: ad.AnnData) -> None:
    """
    Normalize obs/var indexes into plain Python-object strings before writing
    H5AD files.

    AnnData 0.11 gates StringArray writes behind a compatibility flag because
    older readers may not understand that encoding. Converting the axes indexes
    avoids the common `_index` write failure while preserving nullable string
    columns as-is.
    """

    def sanitize_index(index: pd.Index) -> pd.Index:
        values = index.astype("string").to_numpy(dtype=object, na_value=None)
        return pd.Index(values, name=index.name)

    adata.obs.index = sanitize_index(adata.obs.index)
    adata.var.index = sanitize_index(adata.var.index)


def main() -> None:
    args = parse_args()
    fasta_output = (
        args.fasta_output
        if args.fasta_output is not None
        else args.output.with_name("SCP3357_tf_construct_candidates_aa.fasta")
    )
    morf_xlsx_path = ensure_morf_xlsx(args.morf_xlsx)
    adata = build_anndata(args.input_dir, morf_xlsx_path, fasta_output)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sanitize_adata_strings_for_h5ad(adata)
    ad.settings.allow_write_nullable_strings = True
    adata.write_h5ad(args.output, compression="gzip")
    print(f"Wrote {args.output}")
    print(f"Wrote {fasta_output}")
    print(f"Shape: {adata.n_obs} cells x {adata.n_vars} genes")
    print(f"obs columns: {list(adata.obs.columns)}")
    print(f"uns keys: {list(adata.uns.keys())}")


if __name__ == "__main__":
    main()
