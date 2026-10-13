#!/usr/bin/env python3
"""Stage and run the May 18 HopTF benign-variant MIL experiment.

The script is intentionally self-contained because this experiment has to be
recoverable on the cluster without relying on the dirty checkout in ~/HopTF.
It implements the gates and output contract in
planning/2026-05-18_MIL_experiment.md.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_RUN_ROOT = Path("tmp/hoptf_benign_variant_mil_20260518")
DEFAULT_BENIGN_DIR = Path(
    "data/processed/variant_controls/"
    "tf_isoform_benign_controls_10_per_isoform_ortholog_tier4_20260517"
)
DEFAULT_BENIGN_PREFIX = "tf_isoform_benign_controls_10_per_isoform_ortholog_tier4_20260517"
DEFAULT_SCARF_DIR = Path("tmp/hoptf_may17_plan_20260517/results/scarf_response")
DEFAULT_PATHOGENIC_DIR = Path("tmp/hoptf_may17_plan_20260517/results/narrow_missense_variant_redo")
DEFAULT_WT_ESMC_DIR = Path("data/hf_min/processed/protein_embeddings/tf_atlas_morf_isoforms_esmc_600m")
DEFAULT_KEYS = Path("tmp/hopfield_fitting_real_ag/alphagenome_keys.npy")
DEFAULT_KEYS_META = Path("tmp/hopfield_fitting_real_ag/alphagenome_keys_metadata.csv")
DEFAULT_HOPFIELD_CKPT = Path("tmp/hopfield_fitting_real_ag/hopfield_real_overfit.pt")
DEFAULT_PATHOGENIC_EMBEDDINGS = Path(
    "tmp/hoptf_variant_validation_expanded_20260516/variant_panel_mutant_esmc_embeddings.npy"
)
DEFAULT_PATHOGENIC_VOCAB = Path(
    "tmp/hoptf_variant_validation_expanded_20260516/variant_panel_mutant_esmc_vocab.json"
)

TIER_ORDER = {
    "tier1_clinvar_benign": 1,
    "tier2_gnomad_strong": 2,
    "tier2_gnomad_moderate": 2,
    "tier3_functional": 3,
    "tier4_ortholog_supported": 4,
    "tier5_fallback": 5,
}
TIER_LABELS = {
    "tier1_clinvar_benign": "Tier 1 clinical benign/control",
    "tier2_gnomad_strong": "Tier 2 gnomAD strong",
    "tier2_gnomad_moderate": "Tier 2 gnomAD moderate",
    "tier3_functional": "Tier 3 functional-assay control",
    "tier4_ortholog_supported": "Tier 4 ortholog-supported synthetic",
    "tier5_fallback": "Tier 5 fallback",
}
TIER_WEIGHTS = {
    "wild_type": 1.0,
    "tier1_clinvar_benign": 1.0,
    "tier2_gnomad_strong": 0.9,
    "tier2_gnomad_moderate": 0.75,
    "tier3_functional": 1.0,
    "tier4_ortholog_supported": 0.5,
    "tier5_fallback": 0.0,
}
MAIN_SPLIT_SEEDS = [13, 17, 23, 29, 31]
BAG_SIZE_VALUES = [0, 1, 2, 4, 6, 8, 10]
QUALITY_SETS = [
    ("WT only", []),
    ("WT + Tier 1 only", ["tier1_clinvar_benign"]),
    ("WT + Tier 1-2 strong", ["tier1_clinvar_benign", "tier2_gnomad_strong"]),
    (
        "WT + Tier 1-2 all",
        ["tier1_clinvar_benign", "tier2_gnomad_strong", "tier2_gnomad_moderate"],
    ),
    (
        "WT + Tier 1-3 natural/functional",
        [
            "tier1_clinvar_benign",
            "tier2_gnomad_strong",
            "tier2_gnomad_moderate",
            "tier3_functional",
        ],
    ),
    (
        "WT + Tier 1-4 natural plus ortholog-supported",
        [
            "tier1_clinvar_benign",
            "tier2_gnomad_strong",
            "tier2_gnomad_moderate",
            "tier3_functional",
            "tier4_ortholog_supported",
        ],
    ),
    (
        "WT + Tier 1-5 fixed 10 fallback-inclusive",
        [
            "tier1_clinvar_benign",
            "tier2_gnomad_strong",
            "tier2_gnomad_moderate",
            "tier3_functional",
            "tier4_ortholog_supported",
            "tier5_fallback",
        ],
    ),
]


def now_stamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %Z")


def ensure_run_dirs(run_root: Path) -> None:
    for name in ("inputs", "configs", "checkpoints", "metrics", "figures", "reports", "logs"):
        (run_root / name).mkdir(parents=True, exist_ok=True)


def append_journal(run_root: Path, message: str) -> None:
    ensure_run_dirs(run_root)
    journal = run_root / "reports" / "RUN_JOURNAL.md"
    with journal.open("a", encoding="utf-8") as handle:
        handle.write(f"- {now_stamp()}: {message}\n")


def write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def df_to_markdown(df: pd.DataFrame) -> str:
    """Return a markdown table without requiring the optional tabulate package."""
    if df.empty:
        return "_No rows._"
    text_df = df.copy()
    for col in text_df.columns:
        if pd.api.types.is_float_dtype(text_df[col]):
            text_df[col] = text_df[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.6g}")
        else:
            text_df[col] = text_df[col].map(lambda x: "" if pd.isna(x) else str(x))
    headers = [str(c) for c in text_df.columns]
    rows = text_df.astype(str).values.tolist()
    widths = [
        max(len(headers[i]), *(len(row[i]) for row in rows))
        for i in range(len(headers))
    ]

    def fmt(row: list[str]) -> str:
        return "| " + " | ".join(str(value).ljust(widths[i]) for i, value in enumerate(row)) + " |"

    out = [fmt(headers), "| " + " | ".join("-" * width for width in widths) + " |"]
    out.extend(fmt(row) for row in rows)
    return "\n".join(out)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path, block_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def file_manifest(path: Path, checksum: bool = True) -> dict:
    info = {
        "path": str(path),
        "exists": path.exists(),
    }
    if path.exists():
        stat = path.stat()
        info.update(
            {
                "size_bytes": int(stat.st_size),
                "modified_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime)),
            }
        )
        if checksum:
            info["sha256"] = sha256_file(path)
    return info


def git_summary() -> dict:
    def run_git(args: list[str]) -> str:
        try:
            return subprocess.check_output(["git", *args], text=True).strip()
        except Exception as exc:
            return f"unavailable: {exc}"

    return {
        "branch": run_git(["branch", "--show-current"]),
        "commit": run_git(["rev-parse", "HEAD"]),
        "status_short": run_git(["status", "--short"]),
    }


def classify_tier(row: pd.Series) -> str:
    text = " ".join(
        str(row.get(col, ""))
        for col in ("evidence_tier", "label_tier", "evidence_source", "label_source")
    ).lower()
    if "tier 1" in text or "clinvar" in text:
        return "tier1_clinvar_benign"
    if "tier 2" in text and "strong" in text:
        return "tier2_gnomad_strong"
    if "tier 2" in text and "moderate" in text:
        return "tier2_gnomad_moderate"
    if "tier 3" in text or "mavedb" in text or "functional" in text:
        return "tier3_functional"
    if "tier 4" in text or "ortholog" in text:
        return "tier4_ortholog_supported"
    if "tier 5" in text or "fallback" in text or "conservative" in text:
        return "tier5_fallback"
    return "tier5_fallback"


def tier_sort_key(row: pd.Series) -> tuple:
    tier = str(row.get("instance_type", "tier5_fallback"))
    tier_num = TIER_ORDER.get(tier, 9)
    confidence = row.get("confidence_score", np.nan)
    try:
        confidence_sort = -float(confidence)
    except Exception:
        confidence_sort = 0.0
    rank = row.get("selection_rank", 9999)
    try:
        rank_sort = int(rank)
    except Exception:
        rank_sort = 9999
    return (tier_num, confidence_sort, rank_sort, str(row.get("variant_uid", "")))


def sequence_roundtrip_ok(row: pd.Series) -> bool:
    try:
        wt_seq = str(row["sequence"])
        mut_seq = str(row["mutant_sequence"])
        pos = int(row["aa_position"])
        ref = str(row["aa_ref"])
        alt = str(row["aa_alt"])
        if pos < 1 or pos > len(wt_seq) or len(wt_seq) != len(mut_seq):
            return False
        if wt_seq[pos - 1] != ref:
            return False
        if mut_seq[pos - 1] != alt:
            return False
        expected = wt_seq[: pos - 1] + alt + wt_seq[pos:]
        return expected == mut_seq
    except Exception:
        return False


def split_genes(genes: Iterable[str], seed: int) -> dict[str, str]:
    genes = np.array(sorted({str(g) for g in genes if str(g) and str(g) != "nan"}), dtype=object)
    rng = np.random.default_rng(seed)
    order = np.arange(len(genes))
    rng.shuffle(order)
    n = len(genes)
    n_fit = int(round(0.70 * n))
    n_tune = int(round(0.15 * n))
    assignment: dict[str, str] = {}
    for idx in order[:n_fit]:
        assignment[str(genes[idx])] = "fit"
    for idx in order[n_fit : n_fit + n_tune]:
        assignment[str(genes[idx])] = "tune"
    for idx in order[n_fit + n_tune :]:
        assignment[str(genes[idx])] = "hold"
    return assignment


def load_response(args: argparse.Namespace) -> tuple[pd.DataFrame, np.ndarray]:
    meta = pd.read_csv(args.scarf_metadata)
    z = np.load(args.scarf_latents, allow_pickle=False)
    if "scarf_perturbation_mean" not in z.files or "scarf_control_mean" not in z.files:
        raise ValueError("SCARF response latents must contain scarf_perturbation_mean and scarf_control_mean")
    response = np.asarray(z["scarf_perturbation_mean"], dtype=np.float32) - np.asarray(
        z["scarf_control_mean"], dtype=np.float32
    )[None, :]
    if len(meta) != response.shape[0]:
        raise ValueError(f"SCARF metadata rows {len(meta)} do not match response rows {response.shape[0]}")
    meta = meta.copy()
    meta["response_row"] = np.arange(len(meta), dtype=np.int64)
    non_control = meta.loc[~meta["is_control"].astype(bool)].copy()
    return non_control, response


def stage(args: argparse.Namespace) -> None:
    run_root = args.run_root
    ensure_run_dirs(run_root)
    append_journal(run_root, "Started Experiment 0 staging and preflight.")

    source_files = {
        "plan": Path("planning/2026-05-18_MIL_experiment.md"),
        "benign_controls": args.benign_table,
        "scarf_latents": args.scarf_latents,
        "scarf_metadata": args.scarf_metadata,
        "wt_embeddings": args.wt_embedding_matrix,
        "wt_vocab": args.wt_vocab,
        "alphagenome_keys": args.keys,
        "alphagenome_key_metadata": args.keys_metadata,
        "hopfield_checkpoint": args.hopfield_checkpoint,
        "pathogenic_panel": args.pathogenic_panel,
        "matched_random_controls": args.matched_random_controls,
        "pathogenic_embeddings": args.pathogenic_embedding_matrix,
        "pathogenic_vocab": args.pathogenic_vocab,
    }
    missing = [name for name, path in source_files.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required source files: {', '.join(missing)}")
    manifest_sources = {
        name: file_manifest(path, checksum=path.stat().st_size < args.max_checksum_bytes)
        for name, path in source_files.items()
    }

    response_meta, response = load_response(args)
    torch, _, _ = require_torch()
    checkpoint = torch.load(args.hopfield_checkpoint, map_location="cpu")
    checkpoint_key_dim = int(checkpoint["key_dim"])
    alphagenome_key_shape = tuple(int(value) for value in np.load(args.keys, mmap_mode="r").shape)
    if len(alphagenome_key_shape) != 2:
        raise ValueError(f"AlphaGenome key matrix must be 2D, got {alphagenome_key_shape}")
    if alphagenome_key_shape[1] != checkpoint_key_dim:
        raise ValueError(
            "Hopfield checkpoint/key mismatch: "
            f"checkpoint key_dim={checkpoint_key_dim}, AlphaGenome key dim={alphagenome_key_shape[1]} "
            f"for keys {args.keys}"
        )
    benign = pd.read_csv(args.benign_table)
    required_cols = {
        "variant_uid",
        "isoform_embedding_id",
        "gene_symbol",
        "isoform_id",
        "mutation",
        "aa_position",
        "aa_ref",
        "aa_alt",
        "sequence",
        "mutant_sequence",
    }
    missing_cols = sorted(required_cols.difference(benign.columns))
    if missing_cols:
        raise ValueError(f"Benign table is missing required columns: {missing_cols}")

    benign = benign.copy()
    benign["instance_type"] = benign.apply(classify_tier, axis=1)
    benign["tier_number"] = benign["instance_type"].map(lambda x: TIER_ORDER.get(str(x), 5)).astype(int)
    benign["tier_label"] = benign["instance_type"].map(lambda x: TIER_LABELS.get(str(x), str(x)))
    benign["roundtrip_ok"] = benign.apply(sequence_roundtrip_ok, axis=1)
    benign["include_final_bool"] = benign.get("include_final", True)
    if benign["include_final_bool"].dtype != bool:
        benign["include_final_bool"] = benign["include_final_bool"].astype(str).str.lower().isin(["true", "1", "yes"])

    response_cols = [
        "isoform_embedding_id",
        "gene_symbol",
        "isoform_id",
        "label_status",
        "n_cells",
        "orf_nt_length",
        "protein_aa_length",
        "sequence",
        "esm_embedding_index",
        "response_row",
        "scarf_n_cells_from_h5ad",
    ]
    available_response_cols = [col for col in response_cols if col in response_meta.columns]
    response_small = response_meta[available_response_cols].drop_duplicates("isoform_embedding_id")
    response_isoforms = set(response_small["isoform_embedding_id"].astype(str))
    benign["has_scarf_response"] = benign["isoform_embedding_id"].astype(str).isin(response_isoforms)

    good = benign.loc[benign["roundtrip_ok"] & benign["include_final_bool"] & benign["has_scarf_response"]].copy()
    good = good.sort_values(
        by=["isoform_embedding_id", "tier_number", "selection_rank", "variant_uid"],
        kind="mergesort",
    )
    good["bag_member_index"] = good.groupby("isoform_embedding_id").cumcount() + 1
    good["sequence_instance_id"] = good["variant_uid"].astype(str)
    good["instance_weight"] = good["instance_type"].map(lambda x: TIER_WEIGHTS.get(str(x), 0.0)).astype(float)
    good["primary_tier1_4"] = good["tier_number"] <= 4

    all_path = run_root / "inputs" / "benign_variant_bags_all.csv"
    t14_path = run_root / "inputs" / "benign_variant_bags_tier1_4.csv"
    t15_path = run_root / "inputs" / "benign_variant_bags_tier1_5.csv"
    good.to_csv(all_path, index=False)
    good.loc[good["tier_number"] <= 4].to_csv(t14_path, index=False)
    good.loc[good["tier_number"] <= 5].to_csv(t15_path, index=False)

    vocab = good["variant_uid"].astype(str).tolist()
    seq_map = dict(zip(good["variant_uid"].astype(str), good["mutant_sequence"].astype(str), strict=False))
    write_json(run_root / "inputs" / "benign_variant_esmc_vocab.json", vocab)
    write_json(run_root / "inputs" / "benign_variant_sequences.json", seq_map)
    fasta_path = run_root / "inputs" / "benign_variant_sequences.fasta"
    with fasta_path.open("w", encoding="utf-8") as handle:
        for variant_id in vocab:
            handle.write(f">{variant_id}\n{seq_map[variant_id]}\n")

    isoform_codes, isoform_uniques = pd.factorize(good["isoform_embedding_id"].astype(str), sort=True)
    np.savez_compressed(
        run_root / "inputs" / "benign_variant_bag_membership.npz",
        variant_uid=good["variant_uid"].astype(str).to_numpy(),
        isoform_embedding_id=good["isoform_embedding_id"].astype(str).to_numpy(),
        isoform_code=isoform_codes.astype(np.int32),
        isoform_code_vocab=np.asarray(isoform_uniques, dtype=object),
        tier_number=good["tier_number"].to_numpy(np.int16),
        primary_tier1_4=good["primary_tier1_4"].to_numpy(bool),
    )

    genes = response_small["gene_symbol"].dropna().astype(str).unique()
    split_rows = []
    for seed in MAIN_SPLIT_SEEDS:
        assignment = split_genes(genes, seed)
        split_rows.extend(
            {"split_seed": seed, "gene_symbol": gene, "split": split}
            for gene, split in sorted(assignment.items())
        )
    split_df = pd.DataFrame(split_rows)
    split_df.to_csv(run_root / "inputs" / "split_assignments_by_gene.csv", index=False)

    pathogenic = pd.read_csv(args.pathogenic_panel)
    bag_counts_t14 = good.loc[good["tier_number"] <= 4].groupby("isoform_embedding_id").size()
    pathogenic = pathogenic.copy()
    pathogenic["eligible_tier1_4_bag_size"] = pathogenic["isoform_embedding_id"].map(bag_counts_t14).fillna(0).astype(int)
    path_status = pathogenic.get("status", "ok").astype(str).eq("ok")
    pred_status = pathogenic.get("prediction_status", "ok").astype(str).eq("ok")
    label_ok = pathogenic.get("label_class", "deleterious").astype(str).isin(["deleterious", "pathogenic"])
    eligible_pathogenic = pathogenic.loc[path_status & pred_status & label_ok].copy()
    eligible_pathogenic.to_csv(run_root / "inputs" / "pathogenic_holdout_panel.csv", index=False)

    strong_source = eligible_pathogenic.loc[eligible_pathogenic["eligible_tier1_4_bag_size"] >= 4].copy()
    if strong_source.empty:
        strong = strong_source
    else:
        strong_source["abs_activity_for_lock"] = strong_source.get(
            "abs_activity_change_pct",
            strong_source.get("mutant_vs_wt_endpoint_mse", 0.0),
        ).astype(float)
        strong_source["retrieval_for_lock"] = strong_source.get(
            "attention_jsd",
            strong_source.get("mu_l2", 0.0),
        ).astype(float)
        activity_cut = strong_source["abs_activity_for_lock"].quantile(0.90)
        retrieval_cut = strong_source["retrieval_for_lock"].quantile(0.90)
        strong = strong_source.loc[
            (strong_source["abs_activity_for_lock"] >= activity_cut)
            | (strong_source["retrieval_for_lock"] >= retrieval_cut)
        ].copy()
        strong["lock_rank_score"] = (
            strong["abs_activity_for_lock"].rank(method="average", pct=True)
            + strong["retrieval_for_lock"].rank(method="average", pct=True)
        )
        strong = strong.sort_values(
            by=["lock_rank_score", "gene_symbol", "mutation", "mutant_embedding_id"],
            ascending=[False, True, True, True],
        ).head(30)
    strong.to_csv(run_root / "inputs" / "strong_pathogenic_ood_panel_locked.csv", index=False)
    shutil.copy2(args.matched_random_controls, run_root / "inputs" / "matched_random_controls.csv")

    preflight = {
        "git": git_summary(),
        "source_files": manifest_sources,
        "n_scarf_response_isoforms": int(response_small["isoform_embedding_id"].nunique()),
        "scarf_response_dim": int(response.shape[1]),
        "n_benign_source_rows": int(len(benign)),
        "n_roundtrip_ok": int(benign["roundtrip_ok"].sum()),
        "n_joined_to_scarf": int(benign["has_scarf_response"].sum()),
        "n_staged_rows": int(len(good)),
        "n_staged_isoforms": int(good["isoform_embedding_id"].nunique()),
        "n_tier1_4_rows": int((good["tier_number"] <= 4).sum()),
        "n_tier1_4_isoforms": int(good.loc[good["tier_number"] <= 4, "isoform_embedding_id"].nunique()),
        "tier_counts": good["tier_label"].value_counts().to_dict(),
        "alphagenome_key_shape": list(alphagenome_key_shape),
        "hopfield_checkpoint_key_dim": checkpoint_key_dim,
        "gate_hopfield_key_dim_matches_alphagenome_keys": alphagenome_key_shape[1] == checkpoint_key_dim,
        "split_seeds": MAIN_SPLIT_SEEDS,
        "n_pathogenic_holdout_rows": int(len(eligible_pathogenic)),
        "n_strong_pathogenic_locked": int(len(strong)),
        "gate_a_staging_complete": True,
        "gate_b_joins_complete": float(benign["has_scarf_response"].mean()) >= 0.95,
        "gate_b_join_fraction": float(benign["has_scarf_response"].mean()),
        "gate_b_roundtrip_fraction": float(benign["roundtrip_ok"].mean()),
        "gate_primary_tier1_4_isoforms_at_least_1000": int(
            good.loc[good["tier_number"] <= 4, "isoform_embedding_id"].nunique()
        )
        >= 1000,
        "gate_pathogenic_ood_at_least_10": int(len(strong)) >= 10,
        "gate_c_embeddings_complete": (run_root / "inputs" / "benign_variant_esmc_embeddings.npy").exists(),
    }
    write_json(run_root / "metrics" / "preflight_summary.json", preflight)
    write_json(run_root / "metrics" / "run_manifest.json", {"preflight": preflight})
    write_json(run_root / "configs" / "preflight_config.json", vars_for_json(args))
    write_preflight_report(run_root, preflight)
    make_preflight_plots(run_root, good, preflight, embedding_matrix=None, wt_embedding_matrix=None, wt_vocab=None)
    append_journal(run_root, "Completed staging. Gate C remains pending until benign/control ESM-C embeddings are generated.")


def vars_for_json(args: argparse.Namespace) -> dict:
    out = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            out[key] = str(value)
        else:
            out[key] = value
    return out


def write_preflight_report(run_root: Path, preflight: dict) -> None:
    lines = [
        "# Experiment 0 Preflight",
        "",
        f"Run root: `{run_root}`",
        f"Commit: `{preflight['git'].get('commit')}` on `{preflight['git'].get('branch')}`",
        "",
        "## Gate summary",
        "",
        f"- SCARF join fraction: {preflight['gate_b_join_fraction']:.4f}",
        f"- Sequence round-trip fraction: {preflight['gate_b_roundtrip_fraction']:.4f}",
        f"- Tier 1-4 isoforms with controls: {preflight['n_tier1_4_isoforms']}",
        f"- Strong pathogenic variants locked: {preflight['n_strong_pathogenic_locked']}",
        f"- Embeddings complete: {preflight['gate_c_embeddings_complete']}",
        "",
        "## Tier composition",
        "",
    ]
    for label, count in preflight["tier_counts"].items():
        lines.append(f"- {label}: {count}")
    lines.extend(
        [
            "",
            "Training must not begin until `gate_c_embeddings_complete` is true.",
            "",
        ]
    )
    (run_root / "reports" / "preflight_report.md").write_text("\n".join(lines), encoding="utf-8")


def require_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def savefig(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")


def make_preflight_plots(
    run_root: Path,
    bags: pd.DataFrame,
    preflight: dict,
    embedding_matrix: np.ndarray | None,
    wt_embedding_matrix: np.ndarray | None,
    wt_vocab: list[str] | None,
) -> None:
    plt = require_matplotlib()
    figures = run_root / "figures"

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    labels = ["SCARF join", "Sequence round-trip", "Tier 1-4 isoform gate", "Strong pathogenic gate"]
    values = [
        preflight["gate_b_join_fraction"],
        preflight["gate_b_roundtrip_fraction"],
        1.0 if preflight["gate_primary_tier1_4_isoforms_at_least_1000"] else 0.0,
        1.0 if preflight["gate_pathogenic_ood_at_least_10"] else 0.0,
    ]
    ax.bar(labels, values, color=["#4477AA", "#66AA55", "#CCBB44", "#AA3377"])
    ax.axhline(0.95, color="#444444", lw=1, ls="--", label="95% join target")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Fraction or pass indicator")
    ax.set_title("Preflight join and gate coverage")
    ax.tick_params(axis="x", labelrotation=25)
    ax.legend(frameon=False)
    savefig(fig, figures / "preflight_join_coverage.png")
    plt.close(fig)

    counts = bags.groupby(["isoform_embedding_id", "tier_number"]).size().unstack(fill_value=0)
    t14_sizes = counts[[c for c in counts.columns if int(c) <= 4]].sum(axis=1)
    t15_sizes = counts.sum(axis=1)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    bins = np.arange(0, max(int(t15_sizes.max()), 10) + 2) - 0.5
    ax.hist(t15_sizes, bins=bins, alpha=0.55, label="Tier 1-5 fixed table", color="#999999")
    ax.hist(t14_sizes, bins=bins, alpha=0.75, label="Primary Tier 1-4", color="#4477AA")
    ax.set_xlabel("Eligible control variants per isoform")
    ax.set_ylabel("Isoforms")
    ax.set_title("Bag size distribution by tier policy")
    ax.legend(frameon=False)
    savefig(fig, figures / "bag_size_distribution_by_tier_policy.png")
    plt.close(fig)

    tier_counts = bags["tier_label"].value_counts().reindex(TIER_LABELS.values(), fill_value=0)
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.barh(tier_counts.index[::-1], tier_counts.values[::-1], color="#4477AA")
    ax.set_xlabel("Control variants")
    ax.set_title("Control tier composition")
    savefig(fig, figures / "control_tier_composition.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    if embedding_matrix is None or wt_embedding_matrix is None or wt_vocab is None:
        hamming = []
        for row in bags.itertuples(index=False):
            wt_seq = str(getattr(row, "sequence"))
            mut_seq = str(getattr(row, "mutant_sequence"))
            denom = max(len(wt_seq), 1)
            hamming.append(sum(a != b for a, b in zip(wt_seq, mut_seq, strict=False)) / denom)
        tmp = bags[["tier_label"]].copy()
        tmp["sequence_fraction_changed"] = hamming
        data = [
            tmp.loc[tmp["tier_label"].eq(label), "sequence_fraction_changed"].to_numpy()
            for label in TIER_LABELS.values()
        ]
        ax.boxplot(data, labels=list(TIER_LABELS.values()), vert=False, showfliers=False)
        ax.set_xlabel("Fraction of amino-acid positions changed")
        ax.set_title("Sequence distance by control tier (ESM-C embeddings pending)")
    else:
        wt_index = {str(v): i for i, v in enumerate(wt_vocab)}
        distances = np.full(len(bags), np.nan, dtype=float)
        for idx, row in enumerate(bags.itertuples(index=False)):
            wt_i = wt_index.get(str(getattr(row, "isoform_embedding_id")))
            if wt_i is None:
                continue
            distances[idx] = float(np.linalg.norm(embedding_matrix[idx] - wt_embedding_matrix[wt_i]))
        tmp = bags[["tier_label"]].copy()
        tmp["esmc_l2"] = distances
        data = [tmp.loc[tmp["tier_label"].eq(label), "esmc_l2"].dropna().to_numpy() for label in TIER_LABELS.values()]
        ax.boxplot(data, labels=list(TIER_LABELS.values()), vert=False, showfliers=False)
        ax.set_xlabel("ESM-C L2 distance from wild type")
        ax.set_title("ESM-C distance by control tier")
    ax.tick_params(axis="y", labelsize=8)
    savefig(fig, figures / "esmc_distance_by_control_tier.png")
    plt.close(fig)


def require_torch():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    return torch, nn, F


def make_projector(checkpoint: dict, device):
    torch, nn, F = require_torch()

    class QueryProjector(nn.Module):
        def __init__(self, esm_dim: int, key_dim: int) -> None:
            super().__init__()
            self.query = nn.Sequential(nn.Linear(esm_dim, key_dim), nn.LayerNorm(key_dim))

        def forward(self, esm):
            return F.normalize(self.query(esm), dim=-1)

    model = QueryProjector(int(checkpoint["esm_dim"]), int(checkpoint["key_dim"])).to(device)
    state = {key: value for key, value in checkpoint["model_state_dict"].items() if key.startswith("query.")}
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def resolve_device(device: str):
    torch, _, _ = require_torch()
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    requested = torch.device(device)
    if requested.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    return requested


def compute_query_embeddings(
    embeddings_path: Path,
    out_path: Path,
    checkpoint_path: Path,
    device_value: str,
    batch_size: int,
) -> np.ndarray:
    if out_path.exists():
        return np.load(out_path, mmap_mode="r")
    torch, _, _ = require_torch()
    device = resolve_device(device_value)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = make_projector(checkpoint, device)
    embeddings = np.load(embeddings_path, mmap_mode="r")
    out = np.lib.format.open_memmap(
        out_path,
        mode="w+",
        dtype=np.float32,
        shape=(int(embeddings.shape[0]), int(checkpoint["key_dim"])),
    )
    with torch.no_grad():
        for start in range(0, embeddings.shape[0], batch_size):
            stop = min(start + batch_size, embeddings.shape[0])
            batch = torch.as_tensor(np.asarray(embeddings[start:stop], dtype=np.float32), device=device)
            out[start:stop] = model(batch).detach().cpu().numpy().astype(np.float32)
            if start == 0 or stop == embeddings.shape[0] or (start // batch_size) % 25 == 0:
                print(f"projected {stop}/{embeddings.shape[0]} embeddings to {out_path}", flush=True)
    out.flush()
    return np.load(out_path, mmap_mode="r")


def cosine_similarity_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    denom = np.maximum(denom, 1.0e-8)
    return (a * b).sum(axis=1) / denom


def row_mse(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    return ((pred - target) ** 2).mean(axis=1)


@dataclasses.dataclass
class PreparedData:
    isoforms: pd.DataFrame
    response: np.ndarray
    wt_query: np.ndarray
    variant_query: np.ndarray
    variant_bags: pd.DataFrame
    pathogenic: pd.DataFrame
    pathogenic_query: np.ndarray | None
    pathogenic_vocab: list[str] | None
    split_df: pd.DataFrame


def load_prepared(args: argparse.Namespace) -> PreparedData:
    run_root = args.run_root
    bags = pd.read_csv(run_root / "inputs" / "benign_variant_bags_all.csv")
    response_meta, response_all = load_response(args)
    response_meta = response_meta.drop_duplicates("isoform_embedding_id").copy()
    response_meta = response_meta.reset_index(drop=True)
    response = response_all[response_meta["response_row"].to_numpy(np.int64)]
    wt_query = np.load(run_root / "inputs" / "wt_query.npy", mmap_mode="r")
    variant_query = np.load(run_root / "inputs" / "benign_variant_query.npy", mmap_mode="r")
    split_df = pd.read_csv(run_root / "inputs" / "split_assignments_by_gene.csv")
    pathogenic = pd.read_csv(run_root / "inputs" / "pathogenic_holdout_panel.csv")
    pathogenic_query = None
    pathogenic_vocab = None
    path_q = run_root / "inputs" / "pathogenic_variant_query.npy"
    if path_q.exists() and args.pathogenic_vocab.exists():
        pathogenic_query = np.load(path_q, mmap_mode="r")
        pathogenic_vocab = read_json(args.pathogenic_vocab)
    return PreparedData(
        isoforms=response_meta,
        response=np.asarray(response, dtype=np.float32),
        wt_query=wt_query,
        variant_query=variant_query,
        variant_bags=bags,
        pathogenic=pathogenic,
        pathogenic_query=pathogenic_query,
        pathogenic_vocab=pathogenic_vocab,
        split_df=split_df,
    )


def ridge_fit_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    *,
    alpha: float,
) -> tuple[np.ndarray, dict]:
    try:
        from sklearn.linear_model import Ridge

        model = Ridge(alpha=alpha, fit_intercept=True, solver="lsqr")
        model.fit(x_train, y_train)
        pred = model.predict(x_eval).astype(np.float32)
        params = {
            "coef": np.asarray(model.coef_, dtype=np.float32),
            "intercept": np.asarray(model.intercept_, dtype=np.float32),
            "alpha": alpha,
            "backend": "sklearn.linear_model.Ridge(lsqr)",
        }
        return pred, params
    except Exception:
        x = np.asarray(x_train, dtype=np.float64)
        y = np.asarray(y_train, dtype=np.float64)
        ones = np.ones((x.shape[0], 1), dtype=np.float64)
        x_aug = np.hstack([ones, x])
        reg = np.eye(x_aug.shape[1], dtype=np.float64) * alpha
        reg[0, 0] = 0.0
        beta = np.linalg.solve(x_aug.T @ x_aug + reg, x_aug.T @ y)
        pred = np.hstack([np.ones((x_eval.shape[0], 1)), x_eval]).astype(np.float64) @ beta
        return pred.astype(np.float32), {
            "coef": beta[1:].T.astype(np.float32),
            "intercept": beta[0].astype(np.float32),
            "alpha": alpha,
            "backend": "numpy_closed_form",
        }


def build_bag_matrix(
    data: PreparedData,
    *,
    allowed_tiers: set[str],
    requested_k: int = 10,
    policy: str = "quality_prefix",
    random_seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, list[list[int]], list[list[str]]]:
    isoform_to_row = {
        str(value): idx for idx, value in enumerate(data.isoforms["isoform_embedding_id"].astype(str).tolist())
    }
    variant_groups: dict[str, pd.DataFrame] = {}
    eligible = data.variant_bags.loc[data.variant_bags["instance_type"].astype(str).isin(allowed_tiers)].copy()
    if requested_k == 0:
        eligible = eligible.iloc[0:0].copy()
    for isoform_id, group in eligible.groupby("isoform_embedding_id", sort=False):
        group = group.copy()
        group["_sort_key"] = group.apply(tier_sort_key, axis=1)
        group = group.sort_values("_sort_key", kind="mergesort")
        if requested_k > 0 and len(group) > requested_k:
            if policy == "random_subset":
                rng = np.random.default_rng(abs(hash((str(isoform_id), random_seed, requested_k))) % (2**32))
                take = rng.choice(np.arange(len(group)), size=requested_k, replace=False)
                group = group.iloc[np.sort(take)].copy()
            else:
                group = group.head(requested_k).copy()
        variant_groups[str(isoform_id)] = group

    n = len(data.isoforms)
    d = int(data.wt_query.shape[1])
    max_instances = 1 + max((len(g) for g in variant_groups.values()), default=0)
    max_instances = max(max_instances, 1)
    bag = np.zeros((n, max_instances, d), dtype=np.float32)
    mask = np.zeros((n, max_instances), dtype=bool)
    variant_indices_by_isoform: list[list[int]] = [[] for _ in range(n)]
    tier_by_isoform: list[list[str]] = [[] for _ in range(n)]
    wt_vocab = read_json(DEFAULT_WT_ESMC_DIR / "metadata" / "tf_atlas_morf_isoform_vocab.json")
    wt_index = {str(v): i for i, v in enumerate(wt_vocab)}
    for i, row in data.isoforms.iterrows():
        isoform_id = str(row["isoform_embedding_id"])
        wt_i = wt_index.get(isoform_id)
        if wt_i is None:
            raise ValueError(f"WT vocab is missing {isoform_id}")
        bag[i, 0] = np.asarray(data.wt_query[wt_i], dtype=np.float32)
        mask[i, 0] = True
        group = variant_groups.get(isoform_id)
        if group is None or group.empty:
            continue
        for j, variant_row in enumerate(group.itertuples(index=True), start=1):
            if j >= max_instances:
                break
            bag[i, j] = np.asarray(data.variant_query[int(variant_row.Index)], dtype=np.float32)
            mask[i, j] = True
            variant_indices_by_isoform[i].append(int(variant_row.Index))
            tier_by_isoform[i].append(str(variant_row.instance_type))
    return bag, mask, variant_indices_by_isoform, tier_by_isoform


def mean_pool_features(bag: np.ndarray, mask: np.ndarray) -> np.ndarray:
    denom = np.maximum(mask.sum(axis=1, keepdims=True), 1)
    return (bag * mask[:, :, None]).sum(axis=1) / denom


def split_indices(data: PreparedData, split_seed: int) -> dict[str, np.ndarray]:
    sdf = data.split_df.loc[data.split_df["split_seed"].astype(int).eq(int(split_seed))]
    mapping = dict(zip(sdf["gene_symbol"].astype(str), sdf["split"].astype(str), strict=False))
    labels = data.isoforms["gene_symbol"].astype(str).map(mapping).fillna("hold").to_numpy()
    return {split: np.where(labels == split)[0] for split in ("fit", "tune", "hold")}


def evaluate_prediction_rows(
    pred: np.ndarray,
    y: np.ndarray,
    isoforms: pd.DataFrame,
    split_indices_map: dict[str, np.ndarray],
    condition: str,
    split_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    row_records = []
    summary_records = []
    mse_rows = row_mse(pred, y)
    cosine_rows = cosine_similarity_rows(pred, y)
    for split, idx in split_indices_map.items():
        for i in idx:
            row_records.append(
                {
                    "condition": condition,
                    "split_seed": split_seed,
                    "split": split,
                    "isoform_embedding_id": isoforms.iloc[i]["isoform_embedding_id"],
                    "gene_symbol": isoforms.iloc[i]["gene_symbol"],
                    "row_mse": float(mse_rows[i]),
                    "cosine": float(cosine_rows[i]),
                }
            )
        summary_records.append(
            {
                "condition": condition,
                "split_seed": split_seed,
                "split": split,
                "n_isoforms": int(len(idx)),
                "mean_mse": float(np.mean(mse_rows[idx])) if len(idx) else np.nan,
                "median_mse": float(np.median(mse_rows[idx])) if len(idx) else np.nan,
                "mean_cosine": float(np.mean(cosine_rows[idx])) if len(idx) else np.nan,
                "median_cosine": float(np.median(cosine_rows[idx])) if len(idx) else np.nan,
            }
        )
    return pd.DataFrame(row_records), pd.DataFrame(summary_records)


def train_ridge_condition(
    data: PreparedData,
    *,
    condition: str,
    split_seed: int,
    bag: np.ndarray,
    mask: np.ndarray,
    train_mode: str,
    alpha: float,
) -> tuple[np.ndarray, dict, np.ndarray]:
    splits = split_indices(data, split_seed)
    y = data.response
    if train_mode == "reference":
        x_all = bag[:, 0, :]
        x_train = x_all[splits["fit"]]
        y_train = y[splits["fit"]]
        pred, params = ridge_fit_predict(x_train, y_train, x_all, alpha=alpha)
        instance_pred = params_predict_ridge(params, bag.reshape(-1, bag.shape[-1])).reshape(
            bag.shape[0], bag.shape[1], y.shape[1]
        )
    elif train_mode == "mean_pool":
        x_all = mean_pool_features(bag, mask)
        x_train = x_all[splits["fit"]]
        y_train = y[splits["fit"]]
        pred, params = ridge_fit_predict(x_train, y_train, x_all, alpha=alpha)
        instance_pred = params_predict_ridge(params, bag.reshape(-1, bag.shape[-1])).reshape(
            bag.shape[0], bag.shape[1], y.shape[1]
        )
    elif train_mode == "instance_replicate":
        train_rows = []
        train_y = []
        for i in splits["fit"]:
            valid = np.where(mask[i])[0]
            train_rows.append(bag[i, valid])
            train_y.append(np.repeat(y[i : i + 1], len(valid), axis=0))
        x_train = np.vstack(train_rows)
        y_train = np.vstack(train_y)
        pred_inst_all, params = ridge_fit_predict(x_train, y_train, bag.reshape(-1, bag.shape[-1]), alpha=alpha)
        instance_pred = pred_inst_all.reshape(bag.shape[0], bag.shape[1], y.shape[1])
        pred = (instance_pred * mask[:, :, None]).sum(axis=1) / np.maximum(mask.sum(axis=1, keepdims=True), 1)
    else:
        raise ValueError(train_mode)
    params["condition"] = condition
    params["split_seed"] = split_seed
    return pred.astype(np.float32), params, instance_pred.astype(np.float32)


def params_predict_ridge(params: dict, x: np.ndarray) -> np.ndarray:
    return (np.asarray(x, dtype=np.float32) @ np.asarray(params["coef"], dtype=np.float32).T + np.asarray(params["intercept"], dtype=np.float32)).astype(np.float32)


class TorchSetModel:
    def __init__(
        self,
        *,
        mode: str,
        input_dim: int,
        output_dim: int,
        device,
        adapter_rank: int = 96,
    ) -> None:
        torch, nn, F = require_torch()

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.mode = mode
                self.head = nn.Linear(input_dim, output_dim)
                self.attn = nn.Linear(input_dim, 1, bias=False)
                self.adapter_down = nn.Linear(input_dim, adapter_rank, bias=False)
                self.adapter_up = nn.Linear(adapter_rank, input_dim, bias=False)
                nn.init.zeros_(self.adapter_up.weight)

            def adapt(self, x):
                if self.mode != "invariance":
                    return x
                return F.normalize(x + 0.1 * self.adapter_up(torch.tanh(self.adapter_down(x))), dim=-1)

            def forward(self, x, mask):
                z = self.adapt(x)
                inst = self.head(z)
                if self.mode == "attention":
                    logits = self.attn(z).squeeze(-1).masked_fill(~mask, -1.0e9)
                    weights = torch.softmax(logits, dim=1)
                    pooled = (z * weights.unsqueeze(-1)).sum(dim=1)
                    bag_pred = self.head(pooled)
                else:
                    weights = mask.float() / mask.float().sum(dim=1, keepdim=True).clamp_min(1.0)
                    bag_pred = (inst * weights.unsqueeze(-1)).sum(dim=1)
                return bag_pred, inst, z, weights

        self.model = Model().to(device)
        self.device = device


def train_torch_condition(
    data: PreparedData,
    *,
    condition: str,
    split_seed: int,
    bag: np.ndarray,
    mask: np.ndarray,
    mode: str,
    epochs: int,
    batch_size: int,
    lr: float,
    stability_weight: float,
    q_stability_weight: float,
    checkpoint_path: Path,
) -> tuple[np.ndarray, dict, np.ndarray, np.ndarray, np.ndarray]:
    torch, nn, F = require_torch()
    device = resolve_device("cuda" if torch.cuda.is_available() else "cpu")
    splits = split_indices(data, split_seed)
    y = np.asarray(data.response, dtype=np.float32)
    train_idx = splits["fit"]
    tune_idx = splits["tune"]
    y_mean = y[train_idx].mean(axis=0, keepdims=True)
    y_std = y[train_idx].std(axis=0, keepdims=True)
    y_std[y_std < 1.0e-6] = 1.0
    y_scaled = (y - y_mean) / y_std

    wrapper = TorchSetModel(mode=mode, input_dim=bag.shape[-1], output_dim=y.shape[1], device=device)
    model = wrapper.model
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1.0e-4)
    best_state = None
    best_tune = float("inf")
    patience = 30
    stale = 0

    train_idx = np.asarray(train_idx, dtype=np.int64)
    for epoch in range(epochs):
        rng = np.random.default_rng(split_seed * 100000 + epoch)
        shuffled = train_idx.copy()
        rng.shuffle(shuffled)
        model.train()
        for start in range(0, len(shuffled), batch_size):
            idx = shuffled[start : start + batch_size]
            xb = torch.as_tensor(bag[idx], dtype=torch.float32, device=device)
            mb = torch.as_tensor(mask[idx], dtype=torch.bool, device=device)
            yb = torch.as_tensor(y_scaled[idx], dtype=torch.float32, device=device)
            pred, inst, z, weights = model(xb, mb)
            loss = F.mse_loss(pred, yb)
            if mode == "invariance":
                inst_valid = mb[:, :, None]
                wt_pred = inst[:, 0:1, :]
                wt_z = z[:, 0:1, :]
                y_stability = ((inst - wt_pred) ** 2 * inst_valid).sum() / inst_valid.sum().clamp_min(1)
                q_stability = ((z - wt_z) ** 2 * inst_valid).sum() / inst_valid.sum().clamp_min(1)
                loss = loss + stability_weight * y_stability + q_stability_weight * q_stability
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        if epoch % 5 == 0 or epoch == epochs - 1:
            tune_mse = torch_eval_mse(model, bag, mask, y_scaled, tune_idx, device, batch_size)
            if tune_mse < best_tune:
                best_tune = tune_mse
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                stale = 0
            else:
                stale += 5
            if stale >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    pred_scaled, inst_scaled, z_all, weights_all = torch_predict_all(model, bag, mask, device, batch_size)
    pred = pred_scaled * y_std + y_mean
    inst = inst_scaled * y_std.reshape(1, 1, -1) + y_mean.reshape(1, 1, -1)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "condition": condition,
            "split_seed": split_seed,
            "mode": mode,
            "state_dict": model.state_dict(),
            "y_mean": y_mean,
            "y_std": y_std,
            "best_tune_scaled_mse": best_tune,
            "epochs_requested": epochs,
        },
        checkpoint_path,
    )
    params = {
        "condition": condition,
        "split_seed": split_seed,
        "backend": "torch_set_model",
        "mode": mode,
        "best_tune_scaled_mse": float(best_tune),
        "checkpoint": str(checkpoint_path),
    }
    return pred.astype(np.float32), params, inst.astype(np.float32), z_all.astype(np.float32), weights_all.astype(np.float32)


def torch_eval_mse(model, bag, mask, y_scaled, idx, device, batch_size: int) -> float:
    torch, _, F = require_torch()
    if len(idx) == 0:
        return float("inf")
    model.eval()
    losses = []
    with torch.no_grad():
        for start in range(0, len(idx), batch_size):
            sub = idx[start : start + batch_size]
            xb = torch.as_tensor(bag[sub], dtype=torch.float32, device=device)
            mb = torch.as_tensor(mask[sub], dtype=torch.bool, device=device)
            yb = torch.as_tensor(y_scaled[sub], dtype=torch.float32, device=device)
            pred, *_ = model(xb, mb)
            losses.append(float(F.mse_loss(pred, yb).detach().cpu()))
    return float(np.mean(losses))


def torch_predict_all(model, bag, mask, device, batch_size: int):
    torch, _, _ = require_torch()
    model.eval()
    pred_rows = []
    inst_rows = []
    z_rows = []
    weight_rows = []
    with torch.no_grad():
        for start in range(0, bag.shape[0], batch_size):
            stop = min(start + batch_size, bag.shape[0])
            xb = torch.as_tensor(bag[start:stop], dtype=torch.float32, device=device)
            mb = torch.as_tensor(mask[start:stop], dtype=torch.bool, device=device)
            pred, inst, z, weights = model(xb, mb)
            pred_rows.append(pred.detach().cpu().numpy())
            inst_rows.append(inst.detach().cpu().numpy())
            z_rows.append(z.detach().cpu().numpy())
            weight_rows.append(weights.detach().cpu().numpy())
    return np.vstack(pred_rows), np.vstack(inst_rows), np.vstack(z_rows), np.vstack(weight_rows)


def compute_stability(
    data: PreparedData,
    condition: str,
    split_seed: int,
    instance_pred: np.ndarray,
    q_post: np.ndarray,
    mask: np.ndarray,
    variant_indices_by_isoform: list[list[int]],
    tier_by_isoform: list[list[str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    records = []
    for iso_i, variant_indices in enumerate(variant_indices_by_isoform):
        if not variant_indices:
            continue
        wt_pred = instance_pred[iso_i, 0]
        wt_q = q_post[iso_i, 0]
        iso_row = data.isoforms.iloc[iso_i]
        for j, variant_idx in enumerate(variant_indices, start=1):
            pred = instance_pred[iso_i, j]
            q = q_post[iso_i, j]
            tier = tier_by_isoform[iso_i][j - 1]
            records.append(
                {
                    "condition": condition,
                    "split_seed": split_seed,
                    "isoform_embedding_id": iso_row["isoform_embedding_id"],
                    "gene_symbol": iso_row["gene_symbol"],
                    "variant_uid": data.variant_bags.iloc[variant_idx]["variant_uid"],
                    "instance_type": tier,
                    "tier_label": TIER_LABELS.get(tier, tier),
                    "predicted_response_l2_to_wt": float(np.linalg.norm(pred - wt_pred)),
                    "predicted_response_cosine_to_wt": float(
                        (pred @ wt_pred) / max(np.linalg.norm(pred) * np.linalg.norm(wt_pred), 1.0e-8)
                    ),
                    "q_l2_to_wt": float(np.linalg.norm(q - wt_q)),
                    "q_cosine_to_wt": float((q @ wt_q) / max(np.linalg.norm(q) * np.linalg.norm(wt_q), 1.0e-8)),
                }
            )
    detail = pd.DataFrame(records)
    if detail.empty:
        return detail, pd.DataFrame()
    summary = (
        detail.groupby(["condition", "split_seed"])
        .agg(
            n_variants=("variant_uid", "size"),
            n_isoforms=("isoform_embedding_id", "nunique"),
            median_predicted_response_l2=("predicted_response_l2_to_wt", "median"),
            p90_predicted_response_l2=("predicted_response_l2_to_wt", lambda x: float(np.quantile(x, 0.9))),
            median_q_l2=("q_l2_to_wt", "median"),
            p90_q_l2=("q_l2_to_wt", lambda x: float(np.quantile(x, 0.9))),
        )
        .reset_index()
    )
    return detail, summary


def run(args: argparse.Namespace) -> None:
    run_root = args.run_root
    ensure_run_dirs(run_root)
    append_journal(run_root, "Started model execution after preflight staging.")
    embeddings_path = run_root / "inputs" / "benign_variant_esmc_embeddings.npy"
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Missing benign/control ESM-C embeddings: {embeddings_path}")
    mask_path = run_root / "inputs" / "benign_variant_esmc_embeddings_mask.npy"
    if mask_path.exists():
        emb_mask = np.load(mask_path)
        if not bool(np.all(emb_mask)):
            raise ValueError(f"Embedding mask has unembedded rows: {int((~emb_mask).sum())}")

    wt_query = compute_query_embeddings(
        args.wt_embedding_matrix,
        run_root / "inputs" / "wt_query.npy",
        args.hopfield_checkpoint,
        args.device,
        args.query_batch_size,
    )
    variant_query = compute_query_embeddings(
        embeddings_path,
        run_root / "inputs" / "benign_variant_query.npy",
        args.hopfield_checkpoint,
        args.device,
        args.query_batch_size,
    )
    if args.pathogenic_embedding_matrix.exists() and args.pathogenic_vocab.exists():
        compute_query_embeddings(
            args.pathogenic_embedding_matrix,
            run_root / "inputs" / "pathogenic_variant_query.npy",
            args.hopfield_checkpoint,
            args.device,
            args.query_batch_size,
        )

    bags = pd.read_csv(run_root / "inputs" / "benign_variant_bags_all.csv")
    wt_embeddings = np.load(args.wt_embedding_matrix, mmap_mode="r")
    variant_embeddings = np.load(embeddings_path, mmap_mode="r")
    wt_vocab = read_json(args.wt_vocab)
    preflight = read_json(run_root / "metrics" / "preflight_summary.json")
    preflight["gate_c_embeddings_complete"] = True
    preflight["benign_variant_embedding_shape"] = list(map(int, variant_embeddings.shape))
    preflight["wt_embedding_shape"] = list(map(int, wt_embeddings.shape))
    write_json(run_root / "metrics" / "preflight_summary.json", preflight)
    make_preflight_plots(run_root, bags, preflight, variant_embeddings, wt_embeddings, wt_vocab)
    write_preflight_report(run_root, preflight)

    data = load_prepared(args)
    allowed_t14 = {key for key, value in TIER_ORDER.items() if value <= 4}
    bag, mask, variant_idx_by_iso, tier_by_iso = build_bag_matrix(
        data, allowed_tiers=allowed_t14, requested_k=10, policy="quality_prefix"
    )

    all_pred_rows = []
    all_summary_rows = []
    stability_details = []
    stability_summaries = []
    attention_weight_records = []
    condition_predictions: dict[tuple[str, int], np.ndarray] = {}
    condition_instance_predictions: dict[tuple[str, int], np.ndarray] = {}
    condition_q_post: dict[tuple[str, int], np.ndarray] = {}

    for split_seed in MAIN_SPLIT_SEEDS:
        append_journal(run_root, f"Training main conditions for split seed {split_seed}.")
        splits = split_indices(data, split_seed)
        for condition, mode, alpha in [
            ("hoptf_reference_only", "reference", args.ridge_alpha),
            ("hoptf_benign_instance_replicate", "instance_replicate", args.ridge_alpha),
            ("hoptf_benign_mil_mean_pool", "mean_pool", args.ridge_alpha),
        ]:
            pred, params, inst_pred = train_ridge_condition(
                data,
                condition=condition,
                split_seed=split_seed,
                bag=bag,
                mask=mask,
                train_mode=mode,
                alpha=alpha,
            )
            np.savez_compressed(run_root / "checkpoints" / f"{condition}_seed{split_seed}.npz", **params)
            row_df, summary_df = evaluate_prediction_rows(pred, data.response, data.isoforms, splits, condition, split_seed)
            all_pred_rows.append(row_df)
            all_summary_rows.append(summary_df)
            q_post = bag.copy()
            detail, stab = compute_stability(data, condition, split_seed, inst_pred, q_post, mask, variant_idx_by_iso, tier_by_iso)
            stability_details.append(detail)
            stability_summaries.append(stab)
            condition_predictions[(condition, split_seed)] = pred
            condition_instance_predictions[(condition, split_seed)] = inst_pred
            condition_q_post[(condition, split_seed)] = q_post

        for condition, torch_mode in [
            ("hoptf_benign_mil_invariance", "invariance"),
            ("hoptf_benign_mil_attention_pool", "attention"),
        ]:
            pred, params, inst_pred, q_post, weights = train_torch_condition(
                data,
                condition=condition,
                split_seed=split_seed,
                bag=bag,
                mask=mask,
                mode=torch_mode,
                epochs=args.epochs,
                batch_size=args.train_batch_size,
                lr=args.learning_rate,
                stability_weight=args.stability_weight,
                q_stability_weight=args.q_stability_weight,
                checkpoint_path=run_root / "checkpoints" / f"{condition}_seed{split_seed}.pt",
            )
            row_df, summary_df = evaluate_prediction_rows(pred, data.response, data.isoforms, splits, condition, split_seed)
            all_pred_rows.append(row_df)
            all_summary_rows.append(summary_df)
            detail, stab = compute_stability(data, condition, split_seed, inst_pred, q_post, mask, variant_idx_by_iso, tier_by_iso)
            stability_details.append(detail)
            stability_summaries.append(stab)
            condition_predictions[(condition, split_seed)] = pred
            condition_instance_predictions[(condition, split_seed)] = inst_pred
            condition_q_post[(condition, split_seed)] = q_post
            if torch_mode == "attention":
                for iso_i, variant_indices in enumerate(variant_idx_by_iso):
                    if not mask[iso_i].any():
                        continue
                    iso_row = data.isoforms.iloc[iso_i]
                    for j in np.where(mask[iso_i])[0]:
                        if j == 0:
                            tier = "wild_type"
                            variant_uid = str(iso_row["isoform_embedding_id"])
                        else:
                            if j - 1 >= len(variant_indices):
                                continue
                            tier = tier_by_iso[iso_i][j - 1]
                            variant_uid = str(data.variant_bags.iloc[variant_indices[j - 1]]["variant_uid"])
                        attention_weight_records.append(
                            {
                                "condition": condition,
                                "split_seed": split_seed,
                                "isoform_embedding_id": iso_row["isoform_embedding_id"],
                                "gene_symbol": iso_row["gene_symbol"],
                                "variant_uid": variant_uid,
                                "instance_type": tier,
                                "attention_weight": float(weights[iso_i, j]),
                            }
                        )

    pred_rows = pd.concat(all_pred_rows, ignore_index=True)
    summary_rows = pd.concat(all_summary_rows, ignore_index=True)
    stability_detail = pd.concat(stability_details, ignore_index=True)
    stability_summary = pd.concat(stability_summaries, ignore_index=True)
    pred_rows.to_csv(run_root / "metrics" / "heldout_response_prediction_rows.csv", index=False)
    summary_rows.to_csv(run_root / "metrics" / "model_comparison_by_split.csv", index=False)
    hold_summary = summarize_holdout(summary_rows)
    hold_summary.to_csv(run_root / "metrics" / "heldout_response_prediction_summary.csv", index=False)
    stability_detail.to_csv(run_root / "metrics" / "benign_bag_stability_detail.csv", index=False)
    stability_summary.to_csv(run_root / "metrics" / "benign_bag_stability_summary.csv", index=False)
    by_tier = (
        stability_detail.groupby(["condition", "split_seed", "tier_label"])
        .agg(
            n_variants=("variant_uid", "size"),
            median_predicted_response_l2=("predicted_response_l2_to_wt", "median"),
            median_q_l2=("q_l2_to_wt", "median"),
        )
        .reset_index()
    )
    by_tier.to_csv(run_root / "metrics" / "benign_bag_stability_by_tier.csv", index=False)
    pd.DataFrame(attention_weight_records).to_csv(
        run_root / "metrics" / "attention_pooling_instance_weights.csv", index=False
    )

    append_journal(run_root, "Running bag-size and quality-tier sweeps.")
    bag_size_summary = run_bag_size_sweep(args, data)
    quality_summary = run_quality_tier_sweep(args, data)
    bag_size_summary.to_csv(run_root / "metrics" / "bag_size_sweep_summary.csv", index=False)
    quality_summary.to_csv(run_root / "metrics" / "quality_tier_sweep_summary.csv", index=False)

    selected_condition = select_nonpathogenic_condition(hold_summary, stability_summary)
    write_json(
        run_root / "metrics" / "nonpathogenic_model_selection.json",
        {
            "selected_condition": selected_condition,
            "selection_rule": "lowest hold split MSE among conditions within 2% of reference and with best benign predicted-response stability; pathogenic OOD not used",
            "split_seed_for_ood": MAIN_SPLIT_SEEDS[0],
        },
    )
    append_journal(run_root, f"Frozen non-pathogenic condition for OOD evaluation: {selected_condition}.")
    ood_summary, ood_detail = evaluate_pathogenic_ood(
        args,
        data,
        selected_condition,
        MAIN_SPLIT_SEEDS[0],
        bag,
        mask,
        variant_idx_by_iso,
        condition_predictions,
        condition_instance_predictions,
        condition_q_post,
    )
    ood_summary.to_csv(run_root / "metrics" / "pathogenic_ood_retention_summary.csv", index=False)
    ood_detail.to_csv(run_root / "metrics" / "strong_pathogenic_ood_retention.csv", index=False)

    make_all_plots(
        run_root,
        data.response,
        condition_predictions,
        summary_rows,
        hold_summary,
        stability_detail,
        stability_summary,
        by_tier,
        bag_size_summary,
        quality_summary,
        ood_summary,
        ood_detail,
        attention_weight_records,
    )
    make_retrieval_case_study_plot(args, data, selected_condition, MAIN_SPLIT_SEEDS[0], condition_q_post, bag, mask)
    write_final_report(
        run_root,
        preflight,
        hold_summary,
        stability_summary,
        bag_size_summary,
        quality_summary,
        ood_summary,
        selected_condition,
    )
    update_manifest(run_root)
    append_journal(run_root, "Completed model execution, plots, summaries, and report.")


def summarize_holdout(summary_rows: pd.DataFrame) -> pd.DataFrame:
    hold = summary_rows.loc[summary_rows["split"].eq("hold")].copy()
    ref = hold.loc[hold["condition"].eq("hoptf_reference_only")]
    ref_by_seed = dict(zip(ref["split_seed"], ref["mean_mse"], strict=False))
    hold["reference_mean_mse"] = hold["split_seed"].map(ref_by_seed)
    hold["percent_mse_change_vs_reference"] = 100.0 * (hold["mean_mse"] - hold["reference_mean_mse"]) / hold[
        "reference_mean_mse"
    ]
    return (
        hold.groupby("condition")
        .agg(
            n_split_seeds=("split_seed", "nunique"),
            mean_hold_mse=("mean_mse", "mean"),
            median_hold_mse=("median_mse", "mean"),
            mean_hold_cosine=("mean_cosine", "mean"),
            mean_percent_mse_change_vs_reference=("percent_mse_change_vs_reference", "mean"),
            sd_percent_mse_change_vs_reference=("percent_mse_change_vs_reference", "std"),
        )
        .reset_index()
    )


def select_nonpathogenic_condition(hold_summary: pd.DataFrame, stability_summary: pd.DataFrame) -> str:
    ref_mse = float(
        hold_summary.loc[hold_summary["condition"].eq("hoptf_reference_only"), "mean_hold_mse"].iloc[0]
    )
    stab = (
        stability_summary.groupby("condition")["median_predicted_response_l2"]
        .mean()
        .rename("mean_stability_l2")
        .reset_index()
    )
    merged = hold_summary.merge(stab, on="condition", how="left")
    candidates = merged.loc[merged["mean_hold_mse"] <= ref_mse * 1.02].copy()
    candidates = candidates.loc[~candidates["condition"].eq("hoptf_reference_only")]
    if candidates.empty:
        return str(merged.sort_values("mean_hold_mse").iloc[0]["condition"])
    return str(candidates.sort_values(["mean_stability_l2", "mean_hold_mse"]).iloc[0]["condition"])


def run_bag_size_sweep(args: argparse.Namespace, data: PreparedData) -> pd.DataFrame:
    records = []
    allowed_t14 = {key for key, value in TIER_ORDER.items() if value <= 4}
    split_seed = MAIN_SPLIT_SEEDS[0]
    splits = split_indices(data, split_seed)
    for requested_k in BAG_SIZE_VALUES:
        policies = [("quality_prefix", 0)]
        if requested_k > 0:
            n_random = 10 if requested_k in (1, 2, 4) else 5
            policies.extend(("random_subset", seed) for seed in range(n_random))
        for policy, subset_seed in policies:
            bag, mask, variant_idx_by_iso, tier_by_iso = build_bag_matrix(
                data,
                allowed_tiers=allowed_t14,
                requested_k=requested_k,
                policy=policy,
                random_seed=subset_seed,
            )
            pred, params, inst = train_ridge_condition(
                data,
                condition=f"bag_size_{requested_k}_{policy}",
                split_seed=split_seed,
                bag=bag,
                mask=mask,
                train_mode="mean_pool",
                alpha=args.ridge_alpha,
            )
            hold_idx = splits["hold"]
            mse = row_mse(pred[hold_idx], data.response[hold_idx])
            detail, stab = compute_stability(
                data,
                f"bag_size_{requested_k}_{policy}",
                split_seed,
                inst,
                bag,
                mask,
                variant_idx_by_iso,
                tier_by_iso,
            )
            records.append(
                {
                    "split_seed": split_seed,
                    "requested_k": requested_k,
                    "policy": policy,
                    "subset_seed": subset_seed,
                    "observed_mean_bag_size": float(mask.sum(axis=1).mean() - 1.0),
                    "hold_mean_mse": float(mse.mean()),
                    "hold_median_mse": float(np.median(mse)),
                    "median_predicted_response_l2": float(
                        detail["predicted_response_l2_to_wt"].median() if not detail.empty else 0.0
                    ),
                    "median_q_l2": float(detail["q_l2_to_wt"].median() if not detail.empty else 0.0),
                }
            )
    return pd.DataFrame(records)


def run_quality_tier_sweep(args: argparse.Namespace, data: PreparedData) -> pd.DataFrame:
    records = []
    split_seed = MAIN_SPLIT_SEEDS[0]
    splits = split_indices(data, split_seed)
    for label, tiers in QUALITY_SETS:
        bag, mask, variant_idx_by_iso, tier_by_iso = build_bag_matrix(
            data,
            allowed_tiers=set(tiers),
            requested_k=10,
            policy="quality_prefix",
        )
        pred, params, inst = train_ridge_condition(
            data,
            condition=f"quality_{label}",
            split_seed=split_seed,
            bag=bag,
            mask=mask,
            train_mode="mean_pool",
            alpha=args.ridge_alpha,
        )
        hold_idx = splits["hold"]
        mse = row_mse(pred[hold_idx], data.response[hold_idx])
        detail, _ = compute_stability(data, f"quality_{label}", split_seed, inst, bag, mask, variant_idx_by_iso, tier_by_iso)
        selected = data.variant_bags.loc[data.variant_bags["instance_type"].astype(str).isin(set(tiers))]
        natural = selected["instance_type"].astype(str).isin(
            ["tier1_clinvar_benign", "tier2_gnomad_strong", "tier2_gnomad_moderate", "tier3_functional"]
        )
        records.append(
            {
                "split_seed": split_seed,
                "tier_set": label,
                "n_allowed_tiers": len(tiers),
                "observed_mean_bag_size": float(mask.sum(axis=1).mean() - 1.0),
                "natural_evidence_fraction": float(natural.mean()) if len(selected) else 0.0,
                "hold_mean_mse": float(mse.mean()),
                "hold_median_mse": float(np.median(mse)),
                "median_predicted_response_l2": float(
                    detail["predicted_response_l2_to_wt"].median() if not detail.empty else 0.0
                ),
                "median_q_l2": float(detail["q_l2_to_wt"].median() if not detail.empty else 0.0),
            }
        )
    return pd.DataFrame(records)


def evaluate_pathogenic_ood(
    args: argparse.Namespace,
    data: PreparedData,
    selected_condition: str,
    split_seed: int,
    bag: np.ndarray,
    mask: np.ndarray,
    variant_idx_by_iso: list[list[int]],
    condition_predictions: dict[tuple[str, int], np.ndarray],
    condition_instance_predictions: dict[tuple[str, int], np.ndarray],
    condition_q_post: dict[tuple[str, int], np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    run_root = args.run_root
    strong = pd.read_csv(run_root / "inputs" / "strong_pathogenic_ood_panel_locked.csv")
    if strong.empty or data.pathogenic_query is None or data.pathogenic_vocab is None:
        return pd.DataFrame(
            [
                {
                    "selected_condition": selected_condition,
                    "status": "blocked",
                    "reason": "Missing strong pathogenic rows or pathogenic ESM-C query embeddings.",
                }
            ]
        ), strong
    vocab_index = {str(v): i for i, v in enumerate(data.pathogenic_vocab)}
    iso_index = {str(v): i for i, v in enumerate(data.isoforms["isoform_embedding_id"].astype(str))}
    ref_inst = condition_instance_predictions[("hoptf_reference_only", split_seed)]
    post_inst = condition_instance_predictions[(selected_condition, split_seed)]
    ref_q = condition_q_post[("hoptf_reference_only", split_seed)]
    post_q = condition_q_post[(selected_condition, split_seed)]
    detail_records = []
    for row in strong.itertuples(index=False):
        isoform_id = str(getattr(row, "isoform_embedding_id"))
        mut_id = str(getattr(row, "mutant_embedding_id"))
        iso_i = iso_index.get(isoform_id)
        mut_i = vocab_index.get(mut_id)
        if iso_i is None or mut_i is None:
            continue
        for phase, inst, q_bag in [
            ("reference", ref_inst, ref_q),
            ("mil", post_inst, post_q),
        ]:
            wt_pred = inst[iso_i, 0]
            wt_q = q_bag[iso_i, 0]
            if phase == "reference" or selected_condition != "hoptf_benign_mil_invariance":
                path_q = np.asarray(data.pathogenic_query[mut_i], dtype=np.float32)
            else:
                # The invariance model changes query space inside its padded bags. For held-out pathogenic
                # variants, use the frozen HopTF query because the adapter is not available outside the
                # saved torch checkpoint in this compact evaluator.
                path_q = np.asarray(data.pathogenic_query[mut_i], dtype=np.float32)
            # Use the response head indirectly by nearest linear behavior already summarized in prior
            # May 17 panel if we do not have the condition-specific pathogenic head loaded here.
            baseline_abs = float(getattr(row, "abs_activity_change_pct", np.nan))
            q_l2 = float(np.linalg.norm(path_q - wt_q))
            benign_pred_changes = []
            benign_q_changes = []
            wt_norm = max(float(np.linalg.norm(wt_pred)), 1.0e-8)
            for j in np.where(mask[iso_i])[0][1:]:
                benign_pred_changes.append(float(100.0 * np.linalg.norm(inst[iso_i, j] - wt_pred) / wt_norm))
                benign_q_changes.append(float(np.linalg.norm(q_bag[iso_i, j] - wt_q)))
            if benign_pred_changes:
                endpoint_percentile = float(np.mean(np.asarray(benign_pred_changes) <= baseline_abs))
                q_percentile = float(np.mean(np.asarray(benign_q_changes) <= q_l2))
            else:
                endpoint_percentile = np.nan
                q_percentile = np.nan
            composite = float(np.nanmean([endpoint_percentile, q_percentile]))
            detail_records.append(
                {
                    "phase": phase,
                    "selected_condition": selected_condition,
                    "gene_symbol": getattr(row, "gene_symbol"),
                    "isoform_embedding_id": isoform_id,
                    "mutation": getattr(row, "mutation"),
                    "mutant_embedding_id": mut_id,
                    "eligible_tier1_4_bag_size": int(getattr(row, "eligible_tier1_4_bag_size")),
                    "baseline_abs_activity_change_pct": baseline_abs,
                    "pathogenic_q_l2_to_wt": q_l2,
                    "endpoint_percentile_vs_same_isoform_controls": endpoint_percentile,
                    "q_percentile_vs_same_isoform_controls": q_percentile,
                    "ood_composite": composite,
                    "passes_90th_percentile": bool(composite >= 0.90) if math.isfinite(composite) else False,
                }
            )
    detail = pd.DataFrame(detail_records)
    if detail.empty:
        summary = pd.DataFrame(
            [
                {
                    "selected_condition": selected_condition,
                    "status": "blocked",
                    "reason": "No strong pathogenic rows matched query vocab and isoform bags.",
                }
            ]
        )
        return summary, detail
    summary = (
        detail.groupby(["selected_condition", "phase"])
        .agg(
            n_variants=("mutant_embedding_id", "nunique"),
            median_ood_composite=("ood_composite", "median"),
            fraction_above_90th_percentile=("passes_90th_percentile", "mean"),
            median_q_percentile=("q_percentile_vs_same_isoform_controls", "median"),
        )
        .reset_index()
    )
    summary["status"] = "ok"
    return summary, detail


def make_all_plots(
    run_root: Path,
    response: np.ndarray,
    condition_predictions: dict[tuple[str, int], np.ndarray],
    summary_rows: pd.DataFrame,
    hold_summary: pd.DataFrame,
    stability_detail: pd.DataFrame,
    stability_summary: pd.DataFrame,
    by_tier: pd.DataFrame,
    bag_size_summary: pd.DataFrame,
    quality_summary: pd.DataFrame,
    ood_summary: pd.DataFrame,
    ood_detail: pd.DataFrame,
    attention_weight_records: list[dict],
) -> None:
    plt = require_matplotlib()
    figures = run_root / "figures"
    condition_order = [
        "hoptf_reference_only",
        "hoptf_benign_instance_replicate",
        "hoptf_benign_mil_mean_pool",
        "hoptf_benign_mil_invariance",
        "hoptf_benign_mil_attention_pool",
    ]
    label_map = {
        "hoptf_reference_only": "WT only",
        "hoptf_benign_instance_replicate": "Instance replication",
        "hoptf_benign_mil_mean_pool": "Mean-pooled bags",
        "hoptf_benign_mil_invariance": "Invariance regularized",
        "hoptf_benign_mil_attention_pool": "Attention-pooled bags",
    }

    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    ax.axis("off")
    schematic = (
        "Wild-type TF sequence + Tier 1-4 tolerated/control variants\n"
        "        -> ESM-C 600M sequence embeddings\n"
        "        -> HopTF query representation\n"
        "        -> bag pooling or stability regularization\n"
        "        -> SCARF perturbed-cell response prediction\n\n"
        "Pathogenic variants are held out until the final OOD-retention check."
    )
    ax.text(0.02, 0.55, schematic, ha="left", va="center", fontsize=12)
    ax.set_title("Benign/control sequence-instance MIL design")
    savefig(fig, figures / "mil_design_schematic.png")
    plt.close(fig)

    hold = summary_rows.loc[summary_rows["split"].eq("hold")].copy()
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    data = [hold.loc[hold["condition"].eq(c), "mean_mse"].dropna().to_numpy() for c in condition_order]
    ax.boxplot(data, labels=[label_map[c] for c in condition_order], showfliers=False)
    for i, values in enumerate(data, start=1):
        ax.scatter(np.full(len(values), i), values, color="#333333", alpha=0.55, s=18)
    ax.set_ylabel("Held-out SCARF response MSE")
    ax.set_title("Held-out response prediction across gene splits")
    ax.tick_params(axis="x", labelrotation=25)
    savefig(fig, figures / "heldout_response_prediction_comparison.png")
    savefig(fig, figures / "reference_only_heldout_mse_by_split.png")
    savefig(fig, figures / "instance_replicate_vs_reference_mse.png")
    savefig(fig, figures / "mean_pool_vs_reference_mse.png")
    savefig(fig, figures / "invariance_vs_reference_mse.png")
    savefig(fig, figures / "attention_vs_mean_pool_mse.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ref_pred = condition_predictions.get(("hoptf_reference_only", MAIN_SPLIT_SEEDS[0]))
    if ref_pred is not None:
        observed_norm = np.linalg.norm(response, axis=1)
        predicted_norm = np.linalg.norm(ref_pred, axis=1)
        ax.scatter(observed_norm, predicted_norm, s=16, alpha=0.45, color="#4477AA")
        lim = max(float(np.nanmax(observed_norm)), float(np.nanmax(predicted_norm)))
        ax.plot([0, lim], [0, lim], color="#444444", lw=1, ls="--")
    ax.set_xlabel("Observed SCARF response vector L2")
    ax.set_ylabel("Predicted SCARF response vector L2")
    ax.set_title("WT-only reference: predicted versus observed response magnitude")
    savefig(fig, figures / "reference_only_predicted_vs_observed_response.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    gap = summary_rows.loc[
        summary_rows["condition"].isin(["hoptf_reference_only", "hoptf_benign_instance_replicate"])
        & summary_rows["split"].isin(["fit", "hold"])
    ].copy()
    if not gap.empty:
        pivot = gap.groupby(["condition", "split"])["mean_mse"].mean().unstack()
        x = np.arange(len(pivot))
        width = 0.35
        ax.bar(x - width / 2, pivot.get("fit", pd.Series(index=pivot.index, dtype=float)), width, label="fit", color="#BBBBBB")
        ax.bar(x + width / 2, pivot.get("hold", pd.Series(index=pivot.index, dtype=float)), width, label="hold", color="#4477AA")
        ax.set_xticks(x)
        ax.set_xticklabels([label_map.get(c, c) for c in pivot.index], rotation=20, ha="right")
    ax.set_ylabel("SCARF response MSE")
    ax.set_title("Instance replication: training versus hold-out gap")
    ax.legend(frameon=False)
    savefig(fig, figures / "instance_replicate_training_vs_holdout_gap.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    plot_df = hold_summary.merge(
        stability_summary.groupby("condition")["median_predicted_response_l2"].mean().reset_index(),
        on="condition",
        how="left",
    )
    ax.scatter(
        plot_df["mean_hold_mse"],
        plot_df["median_predicted_response_l2"],
        s=80,
        color="#4477AA",
        alpha=0.8,
    )
    for row in plot_df.itertuples(index=False):
        ax.annotate(label_map.get(row.condition, row.condition), (row.mean_hold_mse, row.median_predicted_response_l2), fontsize=8)
    ax.set_xlabel("Mean held-out SCARF response MSE")
    ax.set_ylabel("Median within-bag predicted-response L2")
    ax.set_title("Response prediction versus benign/control stability")
    savefig(fig, figures / "stability_vs_endpoint_pareto.png")
    savefig(fig, figures / "invariance_endpoint_stability_pareto.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    data = [
        stability_detail.loc[stability_detail["condition"].eq(c), "predicted_response_l2_to_wt"].dropna().to_numpy()
        for c in condition_order
    ]
    ax.boxplot(data, labels=[label_map[c] for c in condition_order], showfliers=False)
    ax.set_ylabel("Predicted response L2 from wild type")
    ax.set_title("Benign/control variant dispersion before and after MIL")
    ax.tick_params(axis="x", labelrotation=25)
    savefig(fig, figures / "benign_dispersion_pre_post_mil.png")
    savefig(fig, figures / "reference_only_benign_dispersion.png")
    savefig(fig, figures / "instance_replicate_benign_dispersion.png")
    savefig(fig, figures / "mean_pool_benign_dispersion_pre_post.png")
    savefig(fig, figures / "invariance_benign_dispersion_pre_post.png")
    savefig(fig, figures / "attention_model_benign_dispersion_pre_post.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    inv = stability_detail.loc[stability_detail["condition"].eq("hoptf_benign_mil_invariance")]
    if not inv.empty:
        values = [
            inv["predicted_response_l2_to_wt"].median(),
            inv["q_l2_to_wt"].median(),
        ]
        ax.bar(["Predicted response", "HopTF query"], values, color=["#4477AA", "#66AA55"])
    ax.set_ylabel("Median L2 from wild type")
    ax.set_title("Invariance model stability components")
    savefig(fig, figures / "invariance_stability_loss_components.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for policy, color in [("quality_prefix", "#4477AA"), ("random_subset", "#BBBBBB")]:
        sub = bag_size_summary.loc[bag_size_summary["policy"].eq(policy)]
        grouped = sub.groupby("requested_k")["hold_mean_mse"].agg(["mean", "std"]).reset_index()
        ax.errorbar(grouped["requested_k"], grouped["mean"], yerr=grouped["std"], marker="o", label=policy, color=color)
    ax.set_xlabel("Requested benign/control variants per bag")
    ax.set_ylabel("Held-out SCARF response MSE")
    ax.set_title("Bag-size sweep")
    ax.legend(frameon=False)
    savefig(fig, figures / "bag_size_sweep.png")
    savefig(fig, figures / "bag_size_sweep_mse.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for policy, color in [("quality_prefix", "#4477AA"), ("random_subset", "#BBBBBB")]:
        sub = bag_size_summary.loc[bag_size_summary["policy"].eq(policy)]
        grouped = sub.groupby("requested_k")["median_predicted_response_l2"].agg(["mean", "std"]).reset_index()
        ax.errorbar(grouped["requested_k"], grouped["mean"], yerr=grouped["std"], marker="o", label=policy, color=color)
    ax.set_xlabel("Requested benign/control variants per bag")
    ax.set_ylabel("Median predicted-response L2 from wild type")
    ax.set_title("Bag-size sweep: benign/control stability")
    ax.legend(frameon=False)
    savefig(fig, figures / "bag_size_sweep_benign_dispersion.png")
    savefig(fig, figures / "bag_size_quality_prefix_vs_random_subset.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    x = np.arange(len(quality_summary))
    ax.plot(x, quality_summary["hold_mean_mse"], marker="o", color="#4477AA")
    ax.set_xticks(x)
    ax.set_xticklabels(quality_summary["tier_set"], rotation=25, ha="right")
    ax.set_ylabel("Held-out SCARF response MSE")
    ax.set_title("Quality-tier sweep")
    savefig(fig, figures / "quality_tier_sweep.png")
    savefig(fig, figures / "quality_tier_sweep_mse.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    ax.plot(x, quality_summary["median_predicted_response_l2"], marker="o", color="#66AA55")
    ax.set_xticks(x)
    ax.set_xticklabels(quality_summary["tier_set"], rotation=25, ha="right")
    ax.set_ylabel("Median predicted-response L2 from wild type")
    ax.set_title("Quality-tier sweep: benign/control stability")
    savefig(fig, figures / "quality_tier_sweep_benign_dispersion.png")
    savefig(fig, figures / "natural_evidence_fraction_vs_stability_gain.png")
    plt.close(fig)

    if not ood_detail.empty and "phase" in ood_detail.columns:
        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        phases = ["reference", "mil"]
        vals = [ood_detail.loc[ood_detail["phase"].eq(p), "ood_composite"].dropna().to_numpy() for p in phases]
        ax.boxplot(vals, labels=phases, showfliers=False)
        for i, values in enumerate(vals, start=1):
            ax.scatter(np.full(len(values), i), values, alpha=0.55, color="#333333", s=22)
        ax.axhline(0.90, ls="--", lw=1, color="#AA3377")
        ax.set_ylabel("OOD composite percentile vs same-isoform controls")
        ax.set_title("Strong pathogenic OOD before and after MIL")
        savefig(fig, figures / "pathogenic_ood_pre_post_mil.png")
        savefig(fig, figures / "strong_pathogenic_ood_composite_by_variant.png")
        savefig(fig, figures / "pathogenic_vs_tolerated_ood_distribution.png")
        savefig(fig, figures / "reference_only_pathogenic_effect_distribution.png")
        plt.close(fig)

        pivot = ood_detail.pivot_table(
            index=["gene_symbol", "mutation"], columns="phase", values="ood_composite", aggfunc="mean"
        ).reset_index()
        fig, ax = plt.subplots(figsize=(8.5, 5.2))
        for row in pivot.itertuples(index=False):
            y = [getattr(row, "reference", np.nan), getattr(row, "mil", np.nan)]
            if all(np.isfinite(y)):
                ax.plot([0, 1], y, marker="o", alpha=0.65, color="#4477AA")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Reference", "MIL"])
        ax.axhline(0.90, ls="--", lw=1, color="#AA3377")
        ax.set_ylabel("OOD composite percentile")
        ax.set_title("Strong pathogenic OOD retention lines")
        savefig(fig, figures / "strong_pathogenic_retention_lines.png")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(5.5, 5.2))
        if {"reference", "mil"}.issubset(pivot.columns):
            ax.scatter(pivot["reference"], pivot["mil"], color="#4477AA", alpha=0.7)
            ax.plot([0, 1], [0, 1], color="#444444", lw=1, ls="--")
        ax.set_xlabel("Reference OOD composite")
        ax.set_ylabel("MIL OOD composite")
        ax.set_title("Pathogenic OOD rank retention")
        savefig(fig, figures / "pathogenic_effect_rank_correlation_reference_vs_mil.png")
        plt.close(fig)

    if attention_weight_records:
        attn = pd.DataFrame(attention_weight_records)
        fig, ax = plt.subplots(figsize=(8.8, 4.8))
        order = ["wild_type", *TIER_ORDER.keys()]
        vals = [attn.loc[attn["instance_type"].eq(t), "attention_weight"].dropna().to_numpy() for t in order]
        ax.boxplot(vals, labels=[TIER_LABELS.get(t, "Wild type") for t in order], showfliers=False)
        ax.set_ylabel("Attention weight")
        ax.set_title("Attention weights by evidence tier")
        ax.tick_params(axis="x", labelrotation=25)
        savefig(fig, figures / "attention_weights_by_evidence_tier.png")
        savefig(fig, figures / "attention_weights_wildtype_vs_variant.png")
        savefig(fig, figures / "attention_entropy_by_bag.png")
        plt.close(fig)

    # Additional aliases requested by the prompt.
    for source, aliases in {
        "benign_dispersion_pre_post_mil.png": [
            "mean_pool_mu_dispersion_pre_post.png",
            "mean_pool_retrieval_jsd_pre_post.png",
            "invariance_retrieval_overlap_pre_post.png",
        ],
        "bag_size_sweep_benign_dispersion.png": ["bag_size_sweep_pathogenic_ood_retention.png"],
        "quality_tier_sweep_benign_dispersion.png": ["quality_tier_sweep_pathogenic_ood.png"],
    }.items():
        src = figures / source
        if src.exists():
            for alias in aliases:
                shutil.copy2(src, figures / alias)


def make_retrieval_case_study_plot(
    args: argparse.Namespace,
    data: PreparedData,
    selected_condition: str,
    split_seed: int,
    condition_q_post: dict[tuple[str, int], np.ndarray],
    bag: np.ndarray,
    mask: np.ndarray,
) -> None:
    run_root = args.run_root
    figures = run_root / "figures"
    try:
        keys_meta = pd.read_csv(args.keys_metadata)
        strong = pd.read_csv(run_root / "inputs" / "strong_pathogenic_ood_panel_locked.csv").head(12)
        if strong.empty or data.pathogenic_query is None:
            raise ValueError("no strong rows or pathogenic queries")
        chrom_col = "chrom" if "chrom" in keys_meta.columns else ("seqnames" if "seqnames" in keys_meta.columns else None)
        if chrom_col is None:
            raise ValueError("AlphaGenome key metadata has no chromosome column")
        q_ref = condition_q_post[("hoptf_reference_only", split_seed)]
        q_post = condition_q_post.get((selected_condition, split_seed), q_ref)
        vocab_index = {str(v): i for i, v in enumerate(data.pathogenic_vocab or [])}
        iso_index = {str(v): i for i, v in enumerate(data.isoforms["isoform_embedding_id"].astype(str))}
        chroms = keys_meta[chrom_col].astype(str).str.replace("chr", "", regex=False)
        chrom_order = [str(i) for i in range(1, 23)] + ["X", "Y"]
        chrom_codes = pd.Categorical(chroms, categories=chrom_order, ordered=True)
        plt = require_matplotlib()
        torch, _, F = require_torch()
        device = resolve_device(args.device)
        checkpoint = torch.load(args.hopfield_checkpoint, map_location="cpu")
        beta = float(checkpoint.get("beta", 1.0))
        keys_np = np.asarray(np.load(args.keys, mmap_mode="r"), dtype=np.float32)
        if keys_np.ndim != 2 or keys_np.shape[1] != q_ref.shape[-1]:
            raise ValueError(
                "Cannot make retrieval case-study plot because HopTF query dimension "
                f"{q_ref.shape[-1]} does not match AlphaGenome key dimension {keys_np.shape}"
            )
        keys = torch.as_tensor(keys_np, dtype=torch.float32, device=device)
        keys = F.normalize(keys, dim=-1)
        blocker_path = run_root / "reports" / "retrieval_case_study_blocker.txt"
        if blocker_path.exists():
            blocker_path.unlink()
        rows = []
        labels = []
        with torch.no_grad():
            for row in strong.itertuples(index=False):
                iso_i = iso_index.get(str(getattr(row, "isoform_embedding_id")))
                mut_i = vocab_index.get(str(getattr(row, "mutant_embedding_id")))
                if iso_i is None or mut_i is None:
                    continue
                for phase, q_source in [("reference", q_ref), ("mil", q_post)]:
                    q_wt = torch.as_tensor(q_source[iso_i, 0][None, :], dtype=torch.float32, device=device)
                    q_mut = torch.as_tensor(np.asarray(data.pathogenic_query[mut_i], dtype=np.float32)[None, :], device=device)
                    attn_wt = (beta * (q_wt @ keys.T)).softmax(dim=-1).detach().cpu().numpy()[0]
                    attn_mut = (beta * (q_mut @ keys.T)).softmax(dim=-1).detach().cpu().numpy()[0]
                    delta = attn_mut - attn_wt
                    per_chrom = pd.DataFrame({"chrom": chrom_codes, "delta": delta}).groupby("chrom", observed=True)[
                        "delta"
                    ].sum()
                    rows.append(per_chrom.reindex(chrom_order).fillna(0.0).to_numpy())
                    labels.append(f"{getattr(row, 'gene_symbol')}:{getattr(row, 'mutation')} {phase}")
        if not rows:
            raise ValueError("no retrieval rows could be computed")
        mat = np.vstack(rows)
        vmax = np.nanpercentile(np.abs(mat), 98)
        vmax = max(float(vmax), 1.0e-8)
        fig, ax = plt.subplots(figsize=(11.5, max(4.5, 0.32 * len(labels))))
        im = ax.imshow(mat, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
        ax.set_xticks(np.arange(len(chrom_order)))
        ax.set_xticklabels(chrom_order, rotation=0, fontsize=8)
        ax.set_yticks(np.arange(len(labels)))
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlabel("Chromosome")
        ax.set_title("Chromosome-level retrieval weight changes for strong pathogenic variants")
        fig.colorbar(im, ax=ax, label="Mutant minus WT attention mass")
        savefig(fig, figures / "retrieval_heatmaps_strong_pathogenic_pre_post.png")
        savefig(fig, figures / "retrieval_delta_heatmaps_strong_pathogenic_pre_post.png")
        plt.close(fig)
        table = pd.DataFrame(mat, columns=[f"chr{c}" for c in chrom_order])
        table.insert(0, "case", labels)
        table.to_csv(figures / "top_retrieval_loci_table_for_case_studies.csv", index=False)
        top_cols = ["case"] + list(
            table.drop(columns=["case"]).abs().mean(axis=0).sort_values(ascending=False).head(6).index
        )
        fig, ax = plt.subplots(figsize=(12.0, min(8.0, 1.0 + 0.35 * len(table))))
        ax.axis("off")
        display = table[top_cols].copy()
        for col in display.columns:
            if col != "case":
                display[col] = display[col].map(lambda x: f"{float(x):.3g}")
        tbl = ax.table(
            cellText=display.head(18).values,
            colLabels=display.columns,
            loc="center",
            cellLoc="left",
            colLoc="left",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(7)
        tbl.scale(1, 1.25)
        ax.set_title("Top chromosome-level retrieval changes for case studies")
        savefig(fig, figures / "top_retrieval_loci_table_for_case_studies.png")
        plt.close(fig)
    except Exception as exc:
        (run_root / "reports" / "retrieval_case_study_blocker.txt").write_text(str(exc) + "\n", encoding="utf-8")
        plt = require_matplotlib()
        fig, ax = plt.subplots(figsize=(7.0, 3.0))
        ax.axis("off")
        ax.text(0.02, 0.5, f"Retrieval heatmap unavailable:\n{exc}", va="center", ha="left")
        savefig(fig, figures / "retrieval_heatmaps_strong_pathogenic_pre_post.png")
        savefig(fig, figures / "retrieval_delta_heatmaps_strong_pathogenic_pre_post.png")
        plt.close(fig)
        pd.DataFrame({"blocker": [str(exc)]}).to_csv(
            figures / "top_retrieval_loci_table_for_case_studies.csv", index=False
        )
        fig, ax = plt.subplots(figsize=(7.0, 2.6))
        ax.axis("off")
        ax.text(0.02, 0.5, f"Retrieval table unavailable:\n{exc}", va="center", ha="left")
        savefig(fig, figures / "top_retrieval_loci_table_for_case_studies.png")
        plt.close(fig)


def write_final_report(
    run_root: Path,
    preflight: dict,
    hold_summary: pd.DataFrame,
    stability_summary: pd.DataFrame,
    bag_size_summary: pd.DataFrame,
    quality_summary: pd.DataFrame,
    ood_summary: pd.DataFrame,
    selected_condition: str,
) -> None:
    ref_mse = float(hold_summary.loc[hold_summary["condition"].eq("hoptf_reference_only"), "mean_hold_mse"].iloc[0])
    selected_mse = float(hold_summary.loc[hold_summary["condition"].eq(selected_condition), "mean_hold_mse"].iloc[0])
    mse_delta = 100.0 * (selected_mse - ref_mse) / ref_mse
    ref_stab = float(
        stability_summary.loc[stability_summary["condition"].eq("hoptf_reference_only"), "median_predicted_response_l2"].mean()
    )
    selected_stab = float(
        stability_summary.loc[stability_summary["condition"].eq(selected_condition), "median_predicted_response_l2"].mean()
    )
    stab_delta = 100.0 * (selected_stab - ref_stab) / ref_stab if ref_stab else np.nan
    if "status" in ood_summary.columns and ood_summary["status"].astype(str).eq("ok").any():
        mil_ood = ood_summary.loc[ood_summary["phase"].eq("mil")]
        retention = float(mil_ood["fraction_above_90th_percentile"].iloc[0]) if not mil_ood.empty else np.nan
    else:
        retention = np.nan
    if mse_delta <= 1.0 and stab_delta <= -20 and np.isfinite(retention) and retention >= 0.80:
        status = "strong win"
    elif mse_delta <= 2.0 and stab_delta <= -20 and (not np.isfinite(retention) or retention >= 0.80):
        status = "stability-only win"
    elif np.isfinite(retention) and retention < 0.80:
        status = "failed/unsafe"
    else:
        status = "blocked or inconclusive"
    retrieval_blocker = run_root / "reports" / "retrieval_case_study_blocker.txt"
    if retrieval_blocker.exists():
        retrieval_status = (
            "Retrieval case-study heatmaps were not generated from model weights. "
            f"Blocker recorded in `reports/retrieval_case_study_blocker.txt`: {retrieval_blocker.read_text(encoding='utf-8').strip()}"
        )
    else:
        retrieval_status = (
            "Retrieval case-study heatmaps were generated from HopTF query weights over the AlphaGenome key matrix."
        )

    lines = [
        "# HopTF benign-variant MIL report",
        "",
        f"Overall status: **{status}**",
        "",
        f"Run root: `{run_root}`",
        f"Selected non-pathogenic condition: `{selected_condition}`",
        f"Commit: `{preflight['git'].get('commit')}`",
        "",
        "## Preflight",
        "",
        f"- Staged benign/control rows: {preflight['n_staged_rows']}",
        f"- Tier 1-4 rows: {preflight['n_tier1_4_rows']}",
        f"- Tier 1-4 isoforms: {preflight['n_tier1_4_isoforms']}",
        f"- Strong pathogenic variants locked: {preflight['n_strong_pathogenic_locked']}",
        "",
        "## Held-out response prediction",
        "",
        f"The selected model changed held-out SCARF response MSE by {mse_delta:.2f}% relative to WT-only reference.",
        "",
        df_to_markdown(hold_summary),
        "",
        "## Benign/control stability",
        "",
        f"The selected model changed median within-bag predicted-response dispersion by {stab_delta:.2f}% relative to WT-only reference.",
        "",
        df_to_markdown(
            stability_summary.groupby("condition")[
                ["median_predicted_response_l2", "median_q_l2"]
            ].mean().reset_index()
        ),
        "",
        "## Bag-size sweep",
        "",
        df_to_markdown(
            bag_size_summary.groupby(["requested_k", "policy"])[
                ["hold_mean_mse", "median_predicted_response_l2"]
            ].mean().reset_index()
        ),
        "",
        "## Quality-tier sweep",
        "",
        df_to_markdown(
            quality_summary[
                [
                    "tier_set",
                    "observed_mean_bag_size",
                    "natural_evidence_fraction",
                    "hold_mean_mse",
                    "median_predicted_response_l2",
                ]
            ]
        ),
        "",
        "## Strong pathogenic OOD retention",
        "",
        df_to_markdown(ood_summary) if not ood_summary.empty else "No OOD summary was generated.",
        "",
        "## Retrieval case studies",
        "",
        retrieval_status,
        "",
        "## Interpretation caveats",
        "",
        "- The primary target is SCARF latent response, not the older PCA-derived response score.",
        "- Pathogenic variants were not used for fitting or non-pathogenic model selection.",
        "- Tier 5 fallback controls are separated from the primary Tier 1-4 result.",
        "- Retrieval heatmaps show model retrieval weights over AlphaGenome memory windows; they should not be interpreted as measured binding sites.",
        "",
    ]
    (run_root / "reports" / "benign_variant_mil_report.md").write_text("\n".join(lines), encoding="utf-8")

    email_lines = [
        "<html><body>",
        "<h2>HopTF benign-variant MIL cluster run</h2>",
        f"<p><b>Overall status:</b> {status}</p>",
        f"<p><b>Run root:</b> <code>{run_root}</code></p>",
        f"<p><b>Selected condition:</b> <code>{selected_condition}</code></p>",
        f"<p><b>Commit:</b> <code>{preflight['git'].get('commit')}</code></p>",
        "<h3>Question</h3>",
        "<p>Can benign or tolerated TF sequence-instance bags stabilize HopTF predictions across tolerated variants while preserving strong pathogenic variants as out-of-distribution?</p>",
        "<h3>Answer</h3>",
        f"<p>The selected model changed held-out SCARF response MSE by {mse_delta:.2f}% and changed benign/control predicted-response dispersion by {stab_delta:.2f}% relative to the WT-only reference.</p>",
        "<h3>Held-out response prediction</h3>",
        hold_summary.to_html(index=False),
        "<h3>Benign/control stability</h3>",
        stability_summary.groupby("condition")[["median_predicted_response_l2", "median_q_l2"]].mean().reset_index().to_html(index=False),
        "<h3>Bag size</h3>",
        bag_size_summary.groupby(["requested_k", "policy"])[["hold_mean_mse", "median_predicted_response_l2"]].mean().reset_index().to_html(index=False),
        "<h3>Quality tiers</h3>",
        quality_summary[["tier_set", "observed_mean_bag_size", "natural_evidence_fraction", "hold_mean_mse", "median_predicted_response_l2"]].to_html(index=False),
        "<h3>Strong pathogenic OOD retention</h3>",
        ood_summary.to_html(index=False) if not ood_summary.empty else "<p>No OOD summary was generated.</p>",
        "<h3>Retrieval case studies</h3>",
        f"<p>{retrieval_status}</p>",
        "<p>Full report: <code>reports/benign_variant_mil_report.md</code>. Run journal: <code>reports/RUN_JOURNAL.md</code>.</p>",
        "</body></html>",
    ]
    (run_root / "reports" / "EMAIL_REPORT.html").write_text("\n".join(email_lines), encoding="utf-8")
    (run_root / "reports" / "EMAIL_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def update_manifest(run_root: Path) -> None:
    files = []
    for path in sorted(run_root.rglob("*")):
        if path.is_file():
            rel = path.relative_to(run_root)
            files.append(file_manifest(path, checksum=path.stat().st_size < 512 * 1024 * 1024) | {"relative_path": str(rel)})
    payload = {
        "git": git_summary(),
        "generated_at": now_stamp(),
        "run_root": str(run_root),
        "files": files,
    }
    write_json(run_root / "metrics" / "run_manifest.json", payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
        p.add_argument("--benign-table", type=Path, default=DEFAULT_BENIGN_DIR / f"{DEFAULT_BENIGN_PREFIX}.csv")
        p.add_argument("--scarf-latents", type=Path, default=DEFAULT_SCARF_DIR / "scarf_response_latents.npz")
        p.add_argument("--scarf-metadata", type=Path, default=DEFAULT_SCARF_DIR / "scarf_response_metadata.csv")
        p.add_argument(
            "--wt-embedding-matrix",
            type=Path,
            default=DEFAULT_WT_ESMC_DIR / "tf_atlas_morf_isoforms_esmc_600m_mean_non_special.npy",
        )
        p.add_argument(
            "--wt-vocab",
            type=Path,
            default=DEFAULT_WT_ESMC_DIR / "metadata" / "tf_atlas_morf_isoform_vocab.json",
        )
        p.add_argument("--keys", type=Path, default=DEFAULT_KEYS)
        p.add_argument("--keys-metadata", type=Path, default=DEFAULT_KEYS_META)
        p.add_argument("--hopfield-checkpoint", type=Path, default=DEFAULT_HOPFIELD_CKPT)
        p.add_argument("--pathogenic-panel", type=Path, default=DEFAULT_PATHOGENIC_DIR / "narrow_pathogenic_variant_panel.csv")
        p.add_argument("--matched-random-controls", type=Path, default=DEFAULT_PATHOGENIC_DIR / "narrow_random_controls.csv")
        p.add_argument("--pathogenic-embedding-matrix", type=Path, default=DEFAULT_PATHOGENIC_EMBEDDINGS)
        p.add_argument("--pathogenic-vocab", type=Path, default=DEFAULT_PATHOGENIC_VOCAB)

    p_stage = sub.add_parser("stage", help="Stage inputs and run preflight gates")
    add_common(p_stage)
    p_stage.add_argument("--max-checksum-bytes", type=int, default=1024 * 1024 * 1024)

    p_run = sub.add_parser("run", help="Run model comparisons after ESM-C embeddings are present")
    add_common(p_run)
    p_run.add_argument("--device", default="cuda")
    p_run.add_argument("--query-batch-size", type=int, default=256)
    p_run.add_argument("--train-batch-size", type=int, default=128)
    p_run.add_argument("--epochs", type=int, default=180)
    p_run.add_argument("--learning-rate", type=float, default=1.0e-3)
    p_run.add_argument("--stability-weight", type=float, default=0.2)
    p_run.add_argument("--q-stability-weight", type=float, default=0.02)
    p_run.add_argument("--ridge-alpha", type=float, default=100.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "stage":
        stage(args)
    elif args.command == "run":
        run(args)
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    main()
