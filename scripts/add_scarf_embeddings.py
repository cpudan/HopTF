#!/usr/bin/env python3
"""Add RNA-only SCARF embeddings to an AnnData file."""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from transformers import BertConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scarf import TotalModel_downstream

DEFAULT_CHECKPOINT_DIR = REPO_ROOT / "data" / "reference" / "scarf" / "weights"
DEFAULT_PRIOR_DATA_DIR = REPO_ROOT / "data" / "reference" / "scarf" / "prior_data"
DEFAULT_COUNTS_LAYER = "counts"
DEFAULT_GENE_ID_KEY = "gene_ids"
DEFAULT_OBSM_KEY = "X_scarf"
GENE_VERSION_RE = re.compile(r"\.\d+$")
SPECIES_TO_TOKEN = {"hg38": 0, "mm10": 1}
MYGENE_SPECIES = {"hg38": "human", "mm10": "mouse"}
ALLOWED_UNEXPECTED_KEY_PREFIXES = (
    "loss_R2R.",
    "encoder_rna.cls4value_rec.",
    "encoder_rna.cls4clsGeneAlign.",
    "encoder_rna.kl_enc.",
    "encoder_atac.",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-h5ad", type=Path, required=True, help="Input AnnData file.")
    parser.add_argument("--output-h5ad", type=Path, required=True, help="Output AnnData file with SCARF embeddings.")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=DEFAULT_CHECKPOINT_DIR,
        help="Directory containing the extracted SCARF weights files.",
    )
    parser.add_argument(
        "--prior-data-dir",
        type=Path,
        default=DEFAULT_PRIOR_DATA_DIR,
        help="Directory containing the extracted SCARF RNA token dictionary and median file.",
    )
    parser.add_argument(
        "--counts-layer",
        type=str,
        default=DEFAULT_COUNTS_LAYER,
        help="Layer containing raw counts.",
    )
    parser.add_argument(
        "--gene-id-key",
        type=str,
        default=DEFAULT_GENE_ID_KEY,
        help="`adata.var` column holding Ensembl gene IDs. Falls back to `var_names` when absent.",
    )
    parser.add_argument(
        "--species",
        type=str,
        default="hg38",
        choices=sorted(SPECIES_TO_TOKEN),
        help="Species token expected by SCARF.",
    )
    parser.add_argument(
        "--obsm-key",
        type=str,
        default=DEFAULT_OBSM_KEY,
        help="Destination `obsm` key for the embedding matrix.",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Inference batch size.")
    parser.add_argument(
        "--gene-map-cache",
        type=Path,
        default=REPO_ROOT / "data" / "reference" / "scarf" / "gene_symbol_to_ensembl.json",
        help="JSON cache for symbol-to-Ensembl lookups when `gene_ids` are absent.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Torch device to use. Defaults to `cuda` when available, otherwise `cpu`.",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def load_state_dict_from_index(checkpoint_dir: Path) -> dict[str, torch.Tensor]:
    index_file = checkpoint_dir / "pytorch_model.bin.index.json"
    with index_file.open() as handle:
        index_data = json.load(handle)

    state_dict: dict[str, torch.Tensor] = {}
    for shard_name in sorted(set(index_data["weight_map"].values())):
        shard_path = checkpoint_dir / shard_name
        checkpoint = torch.load(shard_path, map_location="cpu")
        state_dict.update(checkpoint)
    return state_dict


def load_counts_matrix(adata: ad.AnnData, counts_layer: str):
    if counts_layer == "X":
        matrix = adata.X
    elif counts_layer in adata.layers:
        matrix = adata.layers[counts_layer]
    else:
        raise KeyError(
            f"Counts layer {counts_layer!r} was not found. Available layers: {sorted(adata.layers.keys())}"
        )

    if sparse.issparse(matrix):
        return matrix.tocsr().astype(np.float32)
    return np.asarray(matrix, dtype=np.float32)


def resolve_gene_ids(adata: ad.AnnData, gene_id_key: str) -> pd.Series:
    if gene_id_key in adata.var.columns:
        gene_ids = adata.var[gene_id_key].astype(str)
    else:
        gene_ids = pd.Series(adata.var_names.astype(str), index=adata.var_names, name=gene_id_key)

    gene_ids = gene_ids.str.replace(GENE_VERSION_RE, "", regex=True)
    return gene_ids


def normalize_ensembl_candidates(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, dict):
        value = [value]
    candidates: list[str] = []
    if isinstance(value, list):
        for entry in value:
            if isinstance(entry, dict) and "gene" in entry:
                gene_id = str(entry["gene"]).strip()
                if gene_id:
                    candidates.append(GENE_VERSION_RE.sub("", gene_id))
    return sorted(dict.fromkeys(candidates))


def load_gene_map_cache(path: Path) -> dict[str, list[str]]:
    if not path.exists():
        return {}
    with path.open() as handle:
        raw_cache = json.load(handle)
    return {str(k): [str(vv) for vv in v] for k, v in raw_cache.items()}


def save_gene_map_cache(path: Path, cache: dict[str, list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(cache, handle, indent=2, sort_keys=True)


def query_mygene_symbol_map(symbols: list[str], species: str) -> dict[str, list[str]]:
    results: dict[str, list[str]] = {}
    chunk_size = 1000
    for start in range(0, len(symbols), chunk_size):
        chunk = symbols[start : start + chunk_size]
        body = urllib.parse.urlencode(
            {
                "q": ",".join(chunk),
                "scopes": "symbol",
                "fields": "symbol,ensembl.gene",
                "species": species,
            }
        ).encode()
        request = urllib.request.Request("https://mygene.info/v3/query", data=body)
        with urllib.request.urlopen(request) as response:
            payload = json.load(response)
        for record in payload:
            query = str(record.get("query", "")).strip()
            if not query:
                continue
            results[query] = normalize_ensembl_candidates(record.get("ensembl"))
    return results


def resolve_gene_ids_from_symbols(
    adata: ad.AnnData,
    token_dictionary: dict[str, int],
    species: str,
    cache_path: Path,
    gene_id_key: str,
) -> pd.Series:
    symbols = pd.Series(adata.var_names.astype(str), index=adata.var_names, name="gene_symbol")
    unique_symbols = sorted(set(symbols))
    cache = load_gene_map_cache(cache_path)
    missing_symbols = [symbol for symbol in unique_symbols if symbol not in cache]
    if missing_symbols:
        cache.update(query_mygene_symbol_map(missing_symbols, species=MYGENE_SPECIES[species]))
        save_gene_map_cache(cache_path, cache)

    resolved_gene_ids: list[str] = []
    candidate_counts: list[int] = []
    mapping_status: list[str] = []
    for symbol in symbols:
        candidates = [candidate for candidate in cache.get(symbol, []) if candidate in token_dictionary]
        if len(candidates) == 1:
            resolved_gene_ids.append(candidates[0])
            candidate_counts.append(1)
            mapping_status.append("mapped_unique_token_dictionary_match")
        elif len(candidates) > 1:
            resolved_gene_ids.append(candidates[0])
            candidate_counts.append(len(candidates))
            mapping_status.append("mapped_ambiguous_token_dictionary_match")
        else:
            resolved_gene_ids.append("")
            candidate_counts.append(0)
            mapping_status.append("unmapped")

    adata.var[gene_id_key] = resolved_gene_ids
    adata.var[f"{gene_id_key}_candidate_count"] = np.array(candidate_counts, dtype=np.int16)
    adata.var[f"{gene_id_key}_mapping_status"] = pd.Categorical(mapping_status)
    return adata.var[gene_id_key].astype(str)


def normalize_with_scarf_recipe(
    counts_matrix,
    gene_ids: pd.Series,
    token_dictionary: dict[str, int],
    gene_medians: dict[str, float],
):
    keep_mask = gene_ids.isin(token_dictionary).to_numpy()
    if not keep_mask.any():
        raise ValueError(
            "No genes from the AnnData object matched the SCARF token dictionary. "
            "SCARF expects Ensembl-style gene IDs."
        )

    filtered_gene_ids = gene_ids[keep_mask].to_numpy()
    token_ids = np.array([token_dictionary[gene_id] for gene_id in filtered_gene_ids], dtype=np.int32)
    median_values = np.array([gene_medians.get(gene_id, np.nan) for gene_id in filtered_gene_ids], dtype=np.float32)
    inv_medians = np.divide(
        1.0,
        median_values,
        out=np.zeros_like(median_values, dtype=np.float32),
        where=np.isfinite(median_values) & (median_values > 0),
    )

    if sparse.issparse(counts_matrix):
        normalized = counts_matrix[:, keep_mask].tocsr(copy=True)
        totals = np.asarray(normalized.sum(axis=1)).ravel()
        scale = np.divide(
            1e4,
            totals,
            out=np.zeros_like(totals, dtype=np.float32),
            where=totals > 0,
        )
        normalized = normalized.multiply(scale[:, None])
        normalized.data = np.log1p(normalized.data)
        normalized = normalized.multiply(inv_medians)
        normalized.eliminate_zeros()
        return normalized.tocsr(), filtered_gene_ids, token_ids

    normalized = counts_matrix[:, keep_mask].copy()
    totals = normalized.sum(axis=1, keepdims=True)
    scale = np.divide(
        1e4,
        totals,
        out=np.zeros_like(totals, dtype=np.float32),
        where=totals > 0,
    )
    normalized *= scale
    np.log1p(normalized, out=normalized)
    normalized *= inv_medians
    np.nan_to_num(normalized, copy=False)
    return normalized, filtered_gene_ids, token_ids


def rank_rna_inputs(
    normalized_matrix,
    token_ids: np.ndarray,
    cell_names: list[str],
) -> list[dict[str, object]]:
    ranked_cells: list[dict[str, object]] = []

    if sparse.issparse(normalized_matrix):
        matrix = normalized_matrix.tocsr()
        for row_idx, cell_name in enumerate(cell_names):
            row = matrix.getrow(row_idx)
            if row.nnz:
                order = np.argsort(-row.data, kind="stable")
                ranked_token_ids = token_ids[row.indices][order].astype(np.int32, copy=False)
                ranked_values = row.data[order].astype(np.float32, copy=False)
            else:
                ranked_token_ids = np.empty(0, dtype=np.int32)
                ranked_values = np.empty(0, dtype=np.float32)
            ranked_cells.append(
                {
                    "cell_name": cell_name,
                    "rna_gene_ids": ranked_token_ids,
                    "rna_gene_values": ranked_values,
                }
            )
        return ranked_cells

    for row_idx, cell_name in enumerate(cell_names):
        row = normalized_matrix[row_idx]
        nonzero_idx = np.flatnonzero(row)
        if nonzero_idx.size:
            order = np.argsort(-row[nonzero_idx], kind="stable")
            selected_idx = nonzero_idx[order]
            ranked_token_ids = token_ids[selected_idx].astype(np.int32, copy=False)
            ranked_values = row[selected_idx].astype(np.float32, copy=False)
        else:
            ranked_token_ids = np.empty(0, dtype=np.int32)
            ranked_values = np.empty(0, dtype=np.float32)
        ranked_cells.append(
            {
                "cell_name": cell_name,
                "rna_gene_ids": ranked_token_ids,
                "rna_gene_values": ranked_values,
            }
        )
    return ranked_cells


def iter_batches(items: list[dict[str, object]], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def collate_batch(
    batch: list[dict[str, object]],
    max_length: int,
    species_token: int,
    device: torch.device,
) -> dict[str, object]:
    batch_size = len(batch)
    gene_ids = np.zeros((batch_size, max_length), dtype=np.int64)
    values = np.zeros((batch_size, max_length), dtype=np.float32)
    attention_mask = np.zeros((batch_size, max_length), dtype=np.int64)
    cell_names: list[str] = []

    for idx, item in enumerate(batch):
        current_gene_ids = item["rna_gene_ids"][:max_length]
        current_values = item["rna_gene_values"][:max_length]
        length = len(current_gene_ids)
        if length:
            gene_ids[idx, :length] = current_gene_ids
            values[idx, :length] = current_values
            attention_mask[idx, :length] = 1
        cell_names.append(str(item["cell_name"]))

    return {
        "rna_gene_ids": torch.as_tensor(gene_ids, device=device),
        "rna_gene_values": torch.as_tensor(values, device=device),
        "rna_attention_mask": torch.as_tensor(attention_mask, device=device),
        "species": torch.full((batch_size,), species_token, dtype=torch.long, device=device),
        "modality": torch.zeros((batch_size,), dtype=torch.long, device=device),
        "cell_name": cell_names,
    }


def compute_embeddings(
    ranked_cells: list[dict[str, object]],
    checkpoint_dir: Path,
    batch_size: int,
    species_token: int,
    device: torch.device,
) -> tuple[np.ndarray, BertConfig]:
    with (checkpoint_dir / "config.json").open() as handle:
        config = BertConfig(**json.load(handle))

    state_dict = load_state_dict_from_index(checkpoint_dir)
    model = TotalModel_downstream(config, priors={})
    incompat = model.load_state_dict(state_dict, strict=False)
    if incompat.missing_keys:
        raise RuntimeError(f"Missing keys while loading SCARF checkpoint: {incompat.missing_keys}")
    unexpected_keys = [
        key
        for key in incompat.unexpected_keys
        if not key.startswith(ALLOWED_UNEXPECTED_KEY_PREFIXES)
    ]
    if unexpected_keys:
        raise RuntimeError(f"Unexpected keys while loading SCARF checkpoint: {unexpected_keys}")

    model.to(device)
    model.eval()
    max_length = int(config.rna_model_cfg["rna_max_input_size"])

    embeddings: list[np.ndarray] = []
    for batch in iter_batches(ranked_cells, batch_size):
        inputs = collate_batch(batch, max_length=max_length, species_token=species_token, device=device)
        with torch.no_grad():
            result = model.get_rna_embeddings(**inputs)
        for cell_name in inputs["cell_name"]:
            embeddings.append(result[cell_name].detach().cpu().numpy().astype(np.float32, copy=False))

    return np.vstack(embeddings), config


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    if args.species not in SPECIES_TO_TOKEN:
        raise ValueError(f"Unsupported species {args.species!r}. Expected one of {sorted(SPECIES_TO_TOKEN)}.")

    token_dict_path = args.prior_data_dir / "hm_ENSG2token_dict.pickle"
    median_path = args.prior_data_dir / f"RNA_nonzero_median_10W.{args.species}.pickle"
    if not token_dict_path.exists():
        raise FileNotFoundError(f"Missing SCARF token dictionary: {token_dict_path}")
    if not median_path.exists():
        raise FileNotFoundError(
            f"Missing SCARF RNA median file: {median_path}. "
            "The Zenodo 17205044 model_files.zip archive includes the checkpoint weights and "
            "hm_ENSG2token_dict.pickle, but it does not include RNA_nonzero_median_10W.hg38.pickle."
        )
    if device.type != "cuda":
        raise RuntimeError(
            "End-to-end SCARF embedding inference currently requires CUDA in this repo. "
            "The upstream Mamba2 runtime still depends on GPU-oriented Triton scan kernels."
        )

    adata = ad.read_h5ad(args.input_h5ad)
    counts_matrix = load_counts_matrix(adata, args.counts_layer)
    token_dictionary = pd.read_pickle(token_dict_path)
    gene_medians = pd.read_pickle(median_path)
    if args.gene_id_key in adata.var.columns:
        gene_ids = resolve_gene_ids(adata, args.gene_id_key)
    else:
        gene_ids = resolve_gene_ids_from_symbols(
            adata,
            token_dictionary=token_dictionary,
            species=args.species,
            cache_path=args.gene_map_cache,
            gene_id_key=args.gene_id_key,
        )

    normalized_matrix, filtered_gene_ids, token_ids = normalize_with_scarf_recipe(
        counts_matrix,
        gene_ids=gene_ids,
        token_dictionary=token_dictionary,
        gene_medians=gene_medians,
    )
    ranked_cells = rank_rna_inputs(
        normalized_matrix,
        token_ids=token_ids,
        cell_names=adata.obs_names.astype(str).tolist(),
    )
    embeddings, config = compute_embeddings(
        ranked_cells,
        checkpoint_dir=args.checkpoint_dir,
        batch_size=args.batch_size,
        species_token=SPECIES_TO_TOKEN[args.species],
        device=device,
    )

    adata.obsm[args.obsm_key] = embeddings
    adata.uns["scarf_embedding"] = {
        "obsm_key": args.obsm_key,
        "checkpoint_dir": str(args.checkpoint_dir),
        "prior_data_dir": str(args.prior_data_dir),
        "counts_layer": args.counts_layer,
        "gene_id_key": args.gene_id_key,
        "gene_map_cache": str(args.gene_map_cache),
        "species": args.species,
        "device": str(device),
        "num_model_genes_used": int(len(filtered_gene_ids)),
        "embedding_dim": int(config.mlp_out_size),
    }

    args.output_h5ad.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(args.output_h5ad, compression="gzip")
    print(f"Wrote {args.output_h5ad}")
    print(f"Stored embeddings in adata.obsm[{args.obsm_key!r}]")
    print(f"Used {len(filtered_gene_ids)} SCARF-mapped genes")


if __name__ == "__main__":
    main()
