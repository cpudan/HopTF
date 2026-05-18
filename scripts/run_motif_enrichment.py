#!/usr/bin/env python3
"""Test whether top Hopfield-retrieved loci have higher cognate motif scores."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from hopfield_fitting_common import DEFAULT_ESM, DEFAULT_METADATA, DEFAULT_VOCAB, ensure_dir, load_metadata, load_vocab, write_json


BASE_TO_CODE = np.full(256, 4, dtype=np.uint8)
for _base, _code in {"A": 0, "C": 1, "G": 2, "T": 3, "a": 0, "c": 1, "g": 2, "t": 3}.items():
    BASE_TO_CODE[ord(_base)] = _code


@dataclass(frozen=True)
class Motif:
    matrix_id: str
    name: str
    species: str
    pwm: np.ndarray
    pssm: np.ndarray
    rc_pssm: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--embedding-matrix", type=Path, default=DEFAULT_ESM)
    parser.add_argument("--vocab", type=Path, default=DEFAULT_VOCAB)
    parser.add_argument("--keys", type=Path, required=True)
    parser.add_argument("--key-metadata", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--genome-fasta", type=Path, required=True)
    parser.add_argument("--jaspar-meme", type=Path, required=True)
    parser.add_argument("--jaspar-metadata", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--top-k", default="100,500")
    parser.add_argument("--window-radius", type=int, default=500)
    parser.add_argument("--min-cells", type=int, default=5)
    parser.add_argument("--min-top-windows", type=int, default=40)
    parser.add_argument("--background-per-top", type=float, default=1.0)
    parser.add_argument("--motif-threshold-quantile", type=float, default=0.95)
    parser.add_argument("--gc-bins", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-tfs", type=int, default=None)
    parser.add_argument("--max-keys", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260516)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def parse_number_list(value: str, *, cast=int) -> list:
    values = [cast(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError(f"empty comma-separated list: {value!r}")
    return values


def choose_device(value: str):
    import torch

    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def load_rows_and_embeddings(args: argparse.Namespace) -> tuple[pd.DataFrame, np.ndarray]:
    metadata = load_metadata(
        args.metadata,
        require_columns=["perturbation_id", "isoform_embedding_id", "gene_symbol", "n_cells"],
    )
    metadata = metadata.loc[pd.to_numeric(metadata["n_cells"], errors="coerce").fillna(0) >= args.min_cells].copy()
    metadata = metadata.sort_values(["gene_symbol", "perturbation_id"]).reset_index(drop=True)
    vocab = load_vocab(args.vocab)
    vocab_index = {value: index for index, value in enumerate(vocab)}
    keep = metadata["isoform_embedding_id"].astype(str).isin(vocab_index)
    rows = metadata.loc[keep].copy().reset_index(drop=True)
    if rows.empty:
        raise ValueError("no metadata rows matched the protein embedding vocabulary")
    embedding_matrix = np.load(args.embedding_matrix, mmap_mode="r")
    indices = np.asarray([vocab_index[str(value)] for value in rows["isoform_embedding_id"]], dtype=np.int64)
    embeddings = np.asarray(embedding_matrix[indices], dtype=np.float32)
    return rows, embeddings


def load_query_projection(checkpoint: Path, embeddings: np.ndarray, device) -> tuple[np.ndarray, dict[str, object]]:
    import torch
    import torch.nn.functional as F

    from train_hopfield_projection import make_model

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = make_model(int(payload["esm_dim"]), int(payload["key_dim"]), float(payload.get("beta", 1.0)))
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    model.eval()
    queries = []
    with torch.no_grad():
        for start in range(0, embeddings.shape[0], 512):
            chunk = torch.as_tensor(embeddings[start : start + 512], dtype=torch.float32, device=device)
            q = F.normalize(model.query(chunk), dim=-1)
            queries.append(q.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(queries, axis=0), {
        "checkpoint": str(checkpoint),
        "checkpoint_beta": float(payload.get("beta", float("nan"))),
        "esm_dim": int(payload["esm_dim"]),
        "key_dim": int(payload["key_dim"]),
        "value_dim": int(payload.get("value_dim", -1)),
        "mode": str(payload.get("mode", "")),
    }


def load_keys(path: Path, *, max_keys: int | None) -> np.ndarray:
    loaded = np.load(path, mmap_mode="r")
    if loaded.ndim != 2:
        raise ValueError(f"key matrix must be 2D, got {loaded.shape}")
    if max_keys is not None:
        loaded = loaded[: int(max_keys)]
    keys = np.asarray(loaded, dtype=np.float32)
    norms = np.linalg.norm(keys, axis=1, keepdims=True)
    return keys / np.maximum(norms, 1.0e-8)


def load_jaspar_motifs(meme_path: Path, metadata_path: Path) -> dict[str, list[Motif]]:
    metadata = pd.read_csv(metadata_path, sep="\t")
    meta_by_id = metadata.set_index("matrix_id").to_dict(orient="index")
    motifs: dict[str, list[Motif]] = {}
    current_id: str | None = None
    current_name: str | None = None
    matrix: list[list[float]] | None = None
    reading_matrix = False

    def finish() -> None:
        nonlocal current_id, current_name, matrix, reading_matrix
        if current_id is None or not matrix:
            return
        pwm = np.asarray(matrix, dtype=np.float32)
        if pwm.ndim != 2 or pwm.shape[1] != 4:
            return
        pwm = pwm / np.maximum(pwm.sum(axis=1, keepdims=True), 1.0e-8)
        pssm = np.log2((pwm + 1.0e-4) / 0.25).astype(np.float32)
        rc_pssm = pssm[::-1, [3, 2, 1, 0]].copy()
        meta = meta_by_id.get(current_id, {})
        name = str(meta.get("name") or current_name or current_id)
        species = str(meta.get("species") or "")
        motif = Motif(
            matrix_id=current_id,
            name=name,
            species=species,
            pwm=pwm,
            pssm=pssm,
            rc_pssm=rc_pssm,
        )
        motifs.setdefault(name.upper(), []).append(motif)

    with meme_path.open(encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                reading_matrix = False
                continue
            if line.startswith("MOTIF "):
                finish()
                parts = line.split(maxsplit=2)
                current_id = parts[1]
                current_name = parts[2] if len(parts) > 2 else current_id
                matrix = []
                reading_matrix = False
                continue
            if line.startswith("letter-probability matrix"):
                reading_matrix = True
                continue
            if reading_matrix:
                parts = line.split()
                if len(parts) < 4:
                    reading_matrix = False
                    continue
                try:
                    values = [float(item) for item in parts[:4]]
                except ValueError:
                    reading_matrix = False
                    continue
                matrix.append(values)
        finish()
    return motifs


def prefer_human_motifs(motifs: list[Motif]) -> list[Motif]:
    human = [motif for motif in motifs if motif.species == "Homo sapiens"]
    return human or motifs


def encode_sequence(seq: str) -> np.ndarray:
    raw = np.frombuffer(seq.encode("ascii", errors="ignore"), dtype=np.uint8)
    return BASE_TO_CODE[raw]


def build_key_windows(key_metadata: pd.DataFrame, genome_fasta: Path, radius: int) -> tuple[pd.DataFrame, np.ndarray]:
    from pyfaidx import Fasta

    genome = Fasta(str(genome_fasta), as_raw=True, sequence_always_upper=True)
    window_len = 2 * int(radius)
    encoded = np.full((key_metadata.shape[0], window_len), 4, dtype=np.uint8)
    records = []
    genome_chroms = set(genome.keys())
    for idx, row in key_metadata.reset_index(drop=True).iterrows():
        chrom = str(row["chrom"])
        start = int(row["starts"])
        end = int(row["ends"])
        strand = str(row.get("strands", "+"))
        tss = end if strand == "-" else start
        win_start = max(0, tss - radius)
        win_end = win_start + window_len
        valid = chrom in genome_chroms
        seq = ""
        if valid:
            try:
                seq = str(genome[chrom][win_start:win_end])
                valid = len(seq) == window_len
            except Exception:
                valid = False
        if valid:
            encoded[idx, :] = encode_sequence(seq)
            valid_mask = encoded[idx] < 4
            gc = float(np.isin(encoded[idx, valid_mask], [1, 2]).mean()) if valid_mask.any() else float("nan")
        else:
            gc = float("nan")
        records.append(
            {
                "key_index": int(idx),
                "key_id": row.get("key_id", ""),
                "gene_symbol": row.get("gene_symbols", ""),
                "chrom": chrom,
                "strand": strand,
                "tss": int(tss),
                "window_start": int(win_start),
                "window_end": int(win_end),
                "gc_fraction": gc,
                "valid_window": bool(valid),
            }
        )
    return pd.DataFrame(records), encoded


def add_gc_bins(windows: pd.DataFrame, n_bins: int) -> pd.DataFrame:
    out = windows.copy()
    out["gc_bin"] = -1
    valid = out["valid_window"] & np.isfinite(pd.to_numeric(out["gc_fraction"], errors="coerce"))
    if valid.any():
        ranks = out.loc[valid, "gc_fraction"].rank(method="first")
        labels = pd.qcut(ranks, q=min(int(n_bins), int(valid.sum())), labels=False, duplicates="drop")
        out.loc[valid, "gc_bin"] = labels.astype(int).to_numpy()
    return out


def top_loci(queries: np.ndarray, keys: np.ndarray, *, top_k: int, batch_size: int, device) -> np.ndarray:
    import torch

    q_t = torch.as_tensor(queries, dtype=torch.float32, device=device)
    k_t = torch.as_tensor(keys, dtype=torch.float32, device=device)
    chunks = []
    with torch.no_grad():
        for start in range(0, q_t.shape[0], int(batch_size)):
            end = min(start + int(batch_size), q_t.shape[0])
            logits = q_t[start:end] @ k_t.T
            _, idx = torch.topk(logits, k=int(top_k), dim=1)
            chunks.append(idx.detach().cpu().numpy().astype(np.int32))
    return np.concatenate(chunks, axis=0)


def sample_background(
    *,
    top_idx: np.ndarray,
    windows: pd.DataFrame,
    valid_indices_by_bin: dict[int, np.ndarray],
    size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    top_set = set(int(value) for value in top_idx)
    valid_top = windows.loc[top_idx, "valid_window"].to_numpy(dtype=bool)
    top_valid_idx = top_idx[valid_top]
    if top_valid_idx.size == 0:
        return np.asarray([], dtype=np.int32)
    top_bins = windows.loc[top_valid_idx, "gc_bin"].to_numpy(dtype=int)
    out: list[int] = []
    all_bins = sorted(valid_indices_by_bin)
    for gc_bin in top_bins:
        candidates = valid_indices_by_bin.get(int(gc_bin), np.asarray([], dtype=np.int32))
        if candidates.size < 2:
            nearest = sorted(all_bins, key=lambda value: abs(value - int(gc_bin)))[:3]
            candidates = np.concatenate([valid_indices_by_bin[value] for value in nearest])
        candidates = np.asarray([int(value) for value in candidates if int(value) not in top_set], dtype=np.int32)
        if candidates.size == 0:
            continue
        draw_n = max(1, int(math.ceil(float(size) / max(1, top_valid_idx.size))))
        replace = candidates.size < draw_n
        out.extend(int(value) for value in rng.choice(candidates, size=draw_n, replace=replace))
        if len(out) >= size:
            break
    return np.asarray(out[:size], dtype=np.int32)


def score_encoded_sequences(encoded_subset: np.ndarray, motifs: list[Motif]) -> np.ndarray:
    n_seq, seq_len = encoded_subset.shape
    best = np.full(n_seq, -np.inf, dtype=np.float32)
    valid = encoded_subset < 4
    for motif in motifs:
        for pssm in (motif.pssm, motif.rc_pssm):
            width = int(pssm.shape[0])
            if width > seq_len:
                continue
            n_pos = seq_len - width + 1
            scores = np.zeros((n_seq, n_pos), dtype=np.float32)
            ok = np.ones((n_seq, n_pos), dtype=bool)
            for offset in range(width):
                codes = encoded_subset[:, offset : offset + n_pos]
                scores += pssm[offset, np.minimum(codes, 3)]
                ok &= valid[:, offset : offset + n_pos]
            scores[~ok] = -np.inf
            best = np.maximum(best, scores.max(axis=1))
    return best


def bh_fdr(pvalues: pd.Series) -> pd.Series:
    values = pd.to_numeric(pvalues, errors="coerce").to_numpy(dtype=float)
    out = np.full(values.shape, np.nan, dtype=float)
    mask = np.isfinite(values)
    if not mask.any():
        return pd.Series(out, index=pvalues.index)
    finite = values[mask]
    order = np.argsort(finite)
    ranked = finite[order]
    n = float(ranked.size)
    q = ranked * n / (np.arange(ranked.size, dtype=float) + 1.0)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)
    restored = np.empty_like(q)
    restored[order] = q
    out[np.where(mask)[0]] = restored
    return pd.Series(out, index=pvalues.index)


def analyze_motif_enrichment(
    *,
    rows: pd.DataFrame,
    top_idx_matrix: np.ndarray,
    windows: pd.DataFrame,
    encoded_windows: np.ndarray,
    motifs_by_name: dict[str, list[Motif]],
    top_ks: list[int],
    min_top_windows: int,
    background_per_top: float,
    threshold_quantile: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid_indices_by_bin = {
        int(gc_bin): group["key_index"].to_numpy(dtype=np.int32)
        for gc_bin, group in windows.loc[windows["valid_window"] & (windows["gc_bin"] >= 0)].groupby("gc_bin")
    }
    records: list[dict[str, object]] = []
    top_locus_records: list[dict[str, object]] = []
    score_cache: dict[tuple[str, tuple[int, ...]], np.ndarray] = {}

    for row_idx, row in rows.iterrows():
        gene = str(row["gene_symbol"]).upper()
        motifs = prefer_human_motifs(motifs_by_name[gene])
        motif_ids = ",".join(motif.matrix_id for motif in motifs)
        motif_species = ",".join(sorted(set(motif.species for motif in motifs if motif.species)))
        for top_k in top_ks:
            raw_top = top_idx_matrix[row_idx, : int(top_k)]
            top_valid = raw_top[windows.loc[raw_top, "valid_window"].to_numpy(dtype=bool)]
            if top_valid.size < int(min_top_windows):
                records.append(
                    {
                        "row_index": int(row_idx),
                        "perturbation_id": row.get("perturbation_id", ""),
                        "isoform_embedding_id": row.get("isoform_embedding_id", ""),
                        "gene_symbol": row.get("gene_symbol", ""),
                        "isoform_id": row.get("isoform_id", ""),
                        "label_status": row.get("label_status", ""),
                        "n_cells": row.get("n_cells", np.nan),
                        "response_score": row.get("response_score", np.nan),
                        "top_k": int(top_k),
                        "status": "too_few_valid_top_windows",
                        "n_top_tested": int(top_valid.size),
                        "motif_ids": motif_ids,
                        "motif_species": motif_species,
                    }
                )
                continue
            n_bg = int(math.ceil(float(top_valid.size) * float(background_per_top)))
            rng = np.random.default_rng(int(seed) + int(row_idx) * 1009 + int(top_k) * 9173)
            bg_idx = sample_background(
                top_idx=raw_top,
                windows=windows,
                valid_indices_by_bin=valid_indices_by_bin,
                size=n_bg,
                rng=rng,
            )
            if bg_idx.size < int(min_top_windows):
                status = "too_few_background_windows"
            else:
                status = "ok"
            top_key = (gene, tuple(int(value) for value in top_valid))
            bg_key = (gene, tuple(int(value) for value in bg_idx))
            if top_key not in score_cache:
                score_cache[top_key] = score_encoded_sequences(encoded_windows[top_valid], motifs)
            if bg_key not in score_cache:
                score_cache[bg_key] = score_encoded_sequences(encoded_windows[bg_idx], motifs)
            top_scores = score_cache[top_key]
            bg_scores = score_cache[bg_key]
            finite_top_scores = top_scores[np.isfinite(top_scores)]
            finite_bg_scores = bg_scores[np.isfinite(bg_scores)]
            if status == "ok" and (finite_top_scores.size < int(min_top_windows) or finite_bg_scores.size < int(min_top_windows)):
                status = "too_few_finite_scores"
            if status == "ok":
                threshold = float(np.quantile(finite_bg_scores, float(threshold_quantile)))
                top_hits = int(np.sum(top_scores >= threshold))
                bg_hits = int(np.sum(bg_scores >= threshold))
                top_nonhits = int(top_scores.size - top_hits)
                bg_nonhits = int(bg_scores.size - bg_hits)
                fisher = fisher_exact([[top_hits, top_nonhits], [bg_hits, bg_nonhits]], alternative="greater")
                try:
                    mwu = mannwhitneyu(finite_top_scores, finite_bg_scores, alternative="greater")
                    mwu_stat = float(mwu.statistic)
                    mwu_p = float(mwu.pvalue)
                except Exception:
                    mwu_stat = float("nan")
                    mwu_p = float("nan")
                odds_ratio = float(fisher.statistic) if np.isfinite(float(fisher.statistic)) else float("inf")
                fisher_p = float(fisher.pvalue)
                top_hit_fraction = float(top_hits / max(1, top_scores.size))
                bg_hit_fraction = float(bg_hits / max(1, bg_scores.size))
            else:
                threshold = float("nan")
                top_hits = bg_hits = top_nonhits = bg_nonhits = 0
                odds_ratio = fisher_p = top_hit_fraction = bg_hit_fraction = mwu_stat = mwu_p = float("nan")
            records.append(
                {
                    "row_index": int(row_idx),
                    "perturbation_id": row.get("perturbation_id", ""),
                    "isoform_embedding_id": row.get("isoform_embedding_id", ""),
                    "gene_symbol": row.get("gene_symbol", ""),
                    "isoform_id": row.get("isoform_id", ""),
                    "label_status": row.get("label_status", ""),
                    "n_cells": row.get("n_cells", np.nan),
                    "response_score": row.get("response_score", np.nan),
                    "top_k": int(top_k),
                    "status": status,
                    "n_top_tested": int(top_valid.size),
                    "n_background_tested": int(bg_idx.size),
                    "motif_ids": motif_ids,
                    "motif_species": motif_species,
                    "n_motifs_scored": int(len(motifs)),
                    "background_threshold_quantile": float(threshold_quantile),
                    "background_score_threshold": threshold,
                    "top_median_score": float(np.median(finite_top_scores)) if finite_top_scores.size else float("nan"),
                    "background_median_score": float(np.median(finite_bg_scores)) if finite_bg_scores.size else float("nan"),
                    "score_delta_median": float(np.median(finite_top_scores) - np.median(finite_bg_scores)) if finite_top_scores.size and finite_bg_scores.size else float("nan"),
                    "top_hit_fraction": top_hit_fraction,
                    "background_hit_fraction": bg_hit_fraction,
                    "top_hits": top_hits,
                    "top_nonhits": top_nonhits,
                    "background_hits": bg_hits,
                    "background_nonhits": bg_nonhits,
                    "odds_ratio": odds_ratio,
                    "fisher_pvalue_greater": fisher_p,
                    "mannwhitneyu_statistic": mwu_stat,
                    "mannwhitneyu_pvalue_greater": mwu_p,
                }
            )
            for rank, key_index in enumerate(top_valid[: min(25, top_valid.size)], start=1):
                win = windows.iloc[int(key_index)]
                top_locus_records.append(
                    {
                        "row_index": int(row_idx),
                        "top_k": int(top_k),
                        "rank": int(rank),
                        "key_index": int(key_index),
                        "retrieved_gene_symbol": win["gene_symbol"],
                        "chrom": win["chrom"],
                        "window_start": int(win["window_start"]),
                        "window_end": int(win["window_end"]),
                        "gc_fraction": float(win["gc_fraction"]),
                    }
                )
    result = pd.DataFrame(records)
    ok = result["status"].eq("ok") if "status" in result else pd.Series(False, index=result.index)
    result["fisher_fdr"] = np.nan
    result["mannwhitneyu_fdr"] = np.nan
    for top_k, group_idx in result.loc[ok].groupby("top_k").groups.items():
        idx = list(group_idx)
        result.loc[idx, "fisher_fdr"] = bh_fdr(result.loc[idx, "fisher_pvalue_greater"]).to_numpy()
        result.loc[idx, "mannwhitneyu_fdr"] = bh_fdr(result.loc[idx, "mannwhitneyu_pvalue_greater"]).to_numpy()
    return result, pd.DataFrame(top_locus_records)


def save_plots(results: pd.DataFrame, outdir: Path, *, dpi: int) -> None:
    import matplotlib.pyplot as plt

    ok = results.loc[results["status"].eq("ok")].copy()
    if ok.empty:
        return
    odds = pd.to_numeric(ok["odds_ratio"], errors="coerce").replace([np.inf, -np.inf, 0], np.nan)
    ok["log2_odds_ratio"] = np.log2(odds)
    top_ks = sorted(ok["top_k"].dropna().unique())

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    data = [ok.loc[ok["top_k"].eq(k), "log2_odds_ratio"].dropna().to_numpy() for k in top_ks]
    ax.boxplot(data, tick_labels=[str(int(k)) for k in top_ks], showfliers=False)
    ax.axhline(0, color="#444444", linewidth=1, linestyle="--")
    ax.set_xlabel("Number of top-ranked loci tested")
    ax.set_ylabel("log2 odds ratio for high motif-score windows")
    ax.set_title("Cognate motif scores in top-ranked retrieved loci")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "motif_enrichment_log2_odds_by_topk.png", dpi=dpi, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)

    summary = []
    for top_k, group in ok.groupby("top_k"):
        summary.append(
            {
                "top_k": int(top_k),
                "fraction_fdr_0_10": float((pd.to_numeric(group["fisher_fdr"], errors="coerce") <= 0.10).mean()),
                "fraction_nominal": float((pd.to_numeric(group["fisher_pvalue_greater"], errors="coerce") < 0.05).mean()),
                "n": int(group.shape[0]),
            }
        )
    summary_df = pd.DataFrame(summary).sort_values("top_k")
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    x = np.arange(summary_df.shape[0])
    ax.bar(x - 0.18, summary_df["fraction_nominal"], width=0.36, color="#60A5FA", label="Nominal p < 0.05")
    ax.bar(x + 0.18, summary_df["fraction_fdr_0_10"], width=0.36, color="#2563EB", label="FDR <= 0.10")
    ax.set_xticks(x)
    ax.set_xticklabels(summary_df["top_k"].astype(str))
    ax.set_ylim(0, max(0.05, min(1.0, float(summary_df[["fraction_nominal", "fraction_fdr_0_10"]].max().max()) * 1.25)))
    ax.set_xlabel("Number of top-ranked loci tested")
    ax.set_ylabel("Fraction of TF rows")
    ax.set_title("TF rows with higher cognate motif scores than matched background")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "motif_enrichment_significant_fraction_by_topk.png", dpi=dpi, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)

    best = ok.loc[ok["top_k"].eq(min(top_ks))].copy()
    best = best.sort_values(["fisher_fdr", "fisher_pvalue_greater", "odds_ratio"], ascending=[True, True, False]).head(25)
    if not best.empty:
        fig, ax = plt.subplots(figsize=(8.2, max(4.8, 0.24 * best.shape[0] + 1.4)))
        y = np.arange(best.shape[0])
        labels = best["gene_symbol"].astype(str) + " (" + best["isoform_id"].astype(str) + ")"
        odds = pd.to_numeric(best["odds_ratio"], errors="coerce").replace([np.inf, -np.inf, 0], np.nan)
        x = np.log2(odds).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        ax.barh(y, x, color="#2563EB")
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.axvline(0, color="#444444", linewidth=1, linestyle="--")
        ax.set_xlabel("log2 odds ratio")
        ax.set_title(f"Top nominal cognate motif enrichments, top-{int(min(top_ks))} loci")
        ax.grid(axis="x", alpha=0.25)
        fig.tight_layout()
        fig.savefig(outdir / "motif_enrichment_top_rows.png", dpi=dpi, bbox_inches="tight", pad_inches=0.08)
        plt.close(fig)


def write_report(out: Path, summary: dict[str, object], results: pd.DataFrame) -> None:
    ok = results.loc[results["status"].eq("ok")].copy()
    lines = [
        "# Motif Enrichment Analysis",
        "",
        "Question: do top-ranked Hopfield-retrieved loci have higher cognate JASPAR motif scores than matched background loci?",
        "",
        "## Inputs",
        "",
        f"- TF rows with exact JASPAR motif names tested: `{summary['n_rows_tested']}`",
        f"- Unique TF genes tested: `{summary['n_genes_tested']}`",
        f"- AlphaGenome key rows: `{summary['n_keys']}`",
        f"- TSS-centered window size: `{summary['window_bp']}` bp",
        f"- Top-k cutoffs: `{summary['top_k']}`",
        f"- Motif source: `{summary['jaspar_meme']}`",
        "",
        "## Method",
        "",
        "For each TF row with an exact JASPAR motif-name match, the analysis ranked AlphaGenome loci by the Hopfield query-key dot product. It then scanned 1 kb TSS-centered windows for the TF's exact JASPAR motif and compared top-ranked loci with GC-matched background loci. A high motif-score window is defined relative to the matched background distribution for that TF row.",
        "",
        "This analysis tests whether retrieved loci are motif-plausible regulatory contexts. It does not prove binding, and it does not use the word attention as a synonym for binding.",
        "",
        "## Main Result",
        "",
    ]
    if ok.empty:
        lines.append("No rows passed filtering.")
    else:
        grouped = []
        for top_k, group in ok.groupby("top_k"):
            grouped.append(
                {
                    "top_k": int(top_k),
                    "rows": int(group.shape[0]),
                    "unique_genes": int(group["gene_symbol"].nunique()),
                    "median_odds_ratio": float(pd.to_numeric(group["odds_ratio"], errors="coerce").replace([np.inf, -np.inf], np.nan).median()),
                    "median_top_hit_fraction": float(pd.to_numeric(group["top_hit_fraction"], errors="coerce").median()),
                    "median_background_hit_fraction": float(pd.to_numeric(group["background_hit_fraction"], errors="coerce").median()),
                    "nominal_p_lt_0_05": int((pd.to_numeric(group["fisher_pvalue_greater"], errors="coerce") < 0.05).sum()),
                    "fdr_le_0_10": int((pd.to_numeric(group["fisher_fdr"], errors="coerce") <= 0.10).sum()),
                }
            )
        lines.extend(["| Top loci | Rows | Genes | Median odds ratio | Median top hit fraction | Median background hit fraction | Nominal p<0.05 | FDR<=0.10 |", "|---:|---:|---:|---:|---:|---:|---:|---:|"])
        for row in grouped:
            lines.append(
                f"| {row['top_k']} | {row['rows']} | {row['unique_genes']} | {row['median_odds_ratio']:.3g} | "
                f"{row['median_top_hit_fraction']:.3f} | {row['median_background_hit_fraction']:.3f} | "
                f"{row['nominal_p_lt_0_05']} | {row['fdr_le_0_10']} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            str(summary["headline_interpretation"]),
            "",
            "Safe wording: the result is evidence about motif-score enrichment in TSS-centered windows of top-ranked retrieved loci. It should not be described as direct TF binding without external binding or accessibility validation.",
        ]
    )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    outdir = ensure_dir(args.outdir)
    top_ks = sorted(set(parse_number_list(args.top_k, cast=int)))
    max_top_k = max(top_ks)
    device = choose_device(args.device)

    motifs_by_name = load_jaspar_motifs(args.jaspar_meme, args.jaspar_metadata)
    all_rows, all_embeddings = load_rows_and_embeddings(args)
    all_rows["gene_symbol_upper"] = all_rows["gene_symbol"].astype(str).str.upper()
    keep = all_rows["gene_symbol_upper"].isin(motifs_by_name).to_numpy()
    rows = all_rows.loc[keep].copy().reset_index(drop=True)
    embeddings = all_embeddings[keep]
    if args.max_tfs is not None:
        keep_n = int(args.max_tfs)
        rows = rows.head(keep_n).copy().reset_index(drop=True)
        embeddings = embeddings[:keep_n]
    if rows.empty:
        raise ValueError("no TF rows had exact JASPAR motif-name matches")

    keys = load_keys(args.keys, max_keys=args.max_keys)
    key_metadata = pd.read_csv(args.key_metadata)
    if args.max_keys is not None:
        key_metadata = key_metadata.head(int(args.max_keys)).copy()
    if key_metadata.shape[0] != keys.shape[0]:
        raise ValueError(f"key metadata rows {key_metadata.shape[0]} != key rows {keys.shape[0]}")

    queries, checkpoint_info = load_query_projection(args.checkpoint, embeddings, device)
    if queries.shape[1] != keys.shape[1]:
        raise ValueError(f"query dim {queries.shape[1]} does not match key dim {keys.shape[1]}")
    top_idx_matrix = top_loci(queries, keys, top_k=max_top_k, batch_size=args.batch_size, device=device)

    windows, encoded_windows = build_key_windows(key_metadata, args.genome_fasta, args.window_radius)
    windows = add_gc_bins(windows, args.gc_bins)
    windows.to_csv(outdir / "motif_key_tss_windows.tsv", sep="\t", index=False)

    results, top_loci_df = analyze_motif_enrichment(
        rows=rows,
        top_idx_matrix=top_idx_matrix,
        windows=windows,
        encoded_windows=encoded_windows,
        motifs_by_name=motifs_by_name,
        top_ks=top_ks,
        min_top_windows=args.min_top_windows,
        background_per_top=args.background_per_top,
        threshold_quantile=args.motif_threshold_quantile,
        seed=args.seed,
    )
    results.to_csv(outdir / "motif_enrichment_per_tf.csv", index=False)
    top_loci_df.to_csv(outdir / "motif_enrichment_top_loci_examples.csv", index=False)
    save_plots(results, outdir, dpi=args.dpi)

    ok = results.loc[results["status"].eq("ok")].copy()
    if ok.empty:
        headline = "No motif-enrichment rows passed filtering, so this experiment remains blocked."
    else:
        best_top_k = min(top_ks)
        best = ok.loc[ok["top_k"].eq(best_top_k)].copy()
        n_fdr = int((pd.to_numeric(best["fisher_fdr"], errors="coerce") <= 0.10).sum())
        n_nominal = int((pd.to_numeric(best["fisher_pvalue_greater"], errors="coerce") < 0.05).sum())
        median_or = float(pd.to_numeric(best["odds_ratio"], errors="coerce").replace([np.inf, -np.inf], np.nan).median())
        headline = (
            f"Using top-{best_top_k} TSS-centered windows, {n_nominal} TF rows were nominally enriched "
            f"for their exact JASPAR motif and {n_fdr} passed FDR <= 0.10. The median odds ratio was {median_or:.3g}."
        )
    summary = {
        **checkpoint_info,
        "metadata": str(args.metadata),
        "embedding_matrix": str(args.embedding_matrix),
        "vocab": str(args.vocab),
        "keys": str(args.keys),
        "key_metadata": str(args.key_metadata),
        "genome_fasta": str(args.genome_fasta),
        "jaspar_meme": str(args.jaspar_meme),
        "jaspar_metadata": str(args.jaspar_metadata),
        "outdir": str(outdir),
        "device": str(device),
        "n_rows_tested": int(rows.shape[0]),
        "n_genes_tested": int(rows["gene_symbol"].nunique()),
        "n_keys": int(keys.shape[0]),
        "valid_tss_windows": int(windows["valid_window"].sum()),
        "window_bp": int(2 * args.window_radius),
        "top_k": top_ks,
        "min_cells": int(args.min_cells),
        "background_per_top": float(args.background_per_top),
        "motif_threshold_quantile": float(args.motif_threshold_quantile),
        "gc_bins": int(args.gc_bins),
        "result_csv": str(outdir / "motif_enrichment_per_tf.csv"),
        "top_loci_examples_csv": str(outdir / "motif_enrichment_top_loci_examples.csv"),
        "headline_interpretation": headline,
    }
    summary_by_top_k = []
    for top_k, group in ok.groupby("top_k"):
        summary_by_top_k.append(
            {
                "top_k": int(top_k),
                "rows": int(group.shape[0]),
                "unique_genes": int(group["gene_symbol"].nunique()),
                "median_odds_ratio": float(pd.to_numeric(group["odds_ratio"], errors="coerce").replace([np.inf, -np.inf], np.nan).median()),
                "median_top_hit_fraction": float(pd.to_numeric(group["top_hit_fraction"], errors="coerce").median()),
                "median_background_hit_fraction": float(pd.to_numeric(group["background_hit_fraction"], errors="coerce").median()),
                "nominal_p_lt_0_05": int((pd.to_numeric(group["fisher_pvalue_greater"], errors="coerce") < 0.05).sum()),
                "fdr_le_0_10": int((pd.to_numeric(group["fisher_fdr"], errors="coerce") <= 0.10).sum()),
            }
        )
    summary["summary_by_top_k"] = summary_by_top_k
    write_json(outdir / "motif_enrichment_summary.json", summary)
    write_report(outdir / "motif_enrichment_report.md", summary, results)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
