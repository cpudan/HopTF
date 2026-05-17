#!/usr/bin/env python3
"""Measure Hopfield retrieval concentration and stability across beta values."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from hopfield_fitting_common import DEFAULT_ESM, DEFAULT_METADATA, DEFAULT_VOCAB, ensure_dir, load_metadata, load_vocab, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--embedding-matrix", type=Path, default=DEFAULT_ESM)
    parser.add_argument("--vocab", type=Path, default=DEFAULT_VOCAB)
    parser.add_argument("--keys", type=Path, default=None)
    parser.add_argument("--key-metadata", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--beta-grid", default="0.1,0.3,1,3,10,30")
    parser.add_argument("--top-k", default="10,100,1000")
    parser.add_argument("--reference-beta", type=float, default=1.0)
    parser.add_argument("--min-cells", type=int, default=5)
    parser.add_argument("--max-tfs", type=int, default=None)
    parser.add_argument("--max-keys", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--plot-existing", action="store_true", help="Render plots from existing CSV outputs in --outdir.")
    return parser.parse_args()


def parse_number_list(value: str, *, cast=float) -> list:
    out = [cast(item.strip()) for item in value.split(",") if item.strip()]
    if not out:
        raise ValueError(f"empty comma-separated list: {value!r}")
    return out


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
    if args.max_tfs is not None:
        rows = rows.head(int(args.max_tfs)).copy()
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
    keys = np.asarray(np.load(path, mmap_mode="r"), dtype=np.float32)
    if keys.ndim != 2:
        raise ValueError(f"key matrix must be 2D, got {keys.shape}")
    if max_keys is not None:
        keys = keys[: int(max_keys)]
    norms = np.linalg.norm(keys, axis=1, keepdims=True)
    return keys / np.maximum(norms, 1.0e-8)


def summarize_beta(
    *,
    rows: pd.DataFrame,
    queries: np.ndarray,
    keys: np.ndarray,
    beta_grid: list[float],
    top_ks: list[int],
    batch_size: int,
    device,
) -> tuple[pd.DataFrame, dict[float, np.ndarray]]:
    import torch
    import torch.nn.functional as F

    q_t = torch.as_tensor(queries, dtype=torch.float32, device=device)
    k_t = torch.as_tensor(keys, dtype=torch.float32, device=device)
    n_keys = int(keys.shape[0])
    log_n_keys = math.log(float(n_keys))
    max_top_k = min(max(top_ks), n_keys)
    metric_rows: list[dict[str, object]] = []
    top_indices: dict[float, list[np.ndarray]] = {float(beta): [] for beta in beta_grid}

    base_cols = [
        "perturbation_id",
        "isoform_embedding_id",
        "gene_symbol",
        "isoform_id",
        "label_status",
        "n_cells",
        "response_score",
    ]
    available_base_cols = [col for col in base_cols if col in rows.columns]

    with torch.no_grad():
        for beta in beta_grid:
            beta = float(beta)
            for start in range(0, q_t.shape[0], int(batch_size)):
                end = min(start + int(batch_size), q_t.shape[0])
                logits = beta * (q_t[start:end] @ k_t.T)
                attn = F.softmax(logits, dim=-1)
                entropy = -(attn * attn.clamp_min(1.0e-12).log()).sum(dim=1)
                effective_loci = torch.exp(entropy)
                inverse_simpson = 1.0 / torch.square(attn).sum(dim=1).clamp_min(1.0e-12)
                top_prob, top_idx = torch.topk(attn, k=max_top_k, dim=1)
                top_indices[beta].append(top_idx.detach().cpu().numpy().astype(np.int32))
                max_attention = top_prob[:, 0]
                batch_rows = rows.iloc[start:end].reset_index(drop=True)
                for local_idx, row in batch_rows.iterrows():
                    record = {col: row[col] for col in available_base_cols}
                    record.update(
                        {
                            "row_index": int(start + local_idx),
                            "beta": beta,
                            "n_keys": n_keys,
                            "attention_entropy": float(entropy[local_idx].detach().cpu()),
                            "normalized_entropy": float(entropy[local_idx].detach().cpu()) / log_n_keys,
                            "effective_loci_exp_entropy": float(effective_loci[local_idx].detach().cpu()),
                            "inverse_simpson_effective_loci": float(inverse_simpson[local_idx].detach().cpu()),
                            "max_attention": float(max_attention[local_idx].detach().cpu()),
                        }
                    )
                    for top_k in top_ks:
                        k = min(int(top_k), max_top_k)
                        record[f"top_{top_k}_mass"] = float(top_prob[local_idx, :k].sum().detach().cpu())
                    metric_rows.append(record)
    joined_top_indices = {beta: np.concatenate(chunks, axis=0) for beta, chunks in top_indices.items()}
    return pd.DataFrame(metric_rows), joined_top_indices


def summarize_stability(
    *,
    rows: pd.DataFrame,
    top_indices: dict[float, np.ndarray],
    beta_grid: list[float],
    top_ks: list[int],
    reference_beta: float,
) -> pd.DataFrame:
    pairs: list[tuple[float, float, str]] = []
    beta_grid = [float(value) for value in beta_grid]
    for left, right in zip(beta_grid[:-1], beta_grid[1:]):
        pairs.append((left, right, "adjacent"))
    if float(reference_beta) in top_indices:
        for beta in beta_grid:
            if beta != float(reference_beta):
                pairs.append((float(reference_beta), beta, "reference"))
    base_cols = ["perturbation_id", "isoform_embedding_id", "gene_symbol", "isoform_id", "label_status"]
    available_base_cols = [col for col in base_cols if col in rows.columns]
    out = []
    for beta_a, beta_b, pair_type in pairs:
        idx_a = top_indices[beta_a]
        idx_b = top_indices[beta_b]
        for row_idx, row in rows.iterrows():
            record = {col: row[col] for col in available_base_cols}
            record.update({"row_index": int(row_idx), "beta_a": beta_a, "beta_b": beta_b, "pair_type": pair_type})
            for top_k in top_ks:
                k = min(int(top_k), idx_a.shape[1], idx_b.shape[1])
                set_a = set(int(value) for value in idx_a[row_idx, :k])
                set_b = set(int(value) for value in idx_b[row_idx, :k])
                record[f"top_{top_k}_overlap_fraction"] = len(set_a & set_b) / float(k)
            out.append(record)
    return pd.DataFrame(out)


def median_iqr(values: pd.Series) -> tuple[float, float, float]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return float("nan"), float("nan"), float("nan")
    return float(clean.median()), float(clean.quantile(0.25)), float(clean.quantile(0.75))


def save_metric_plot(metrics: pd.DataFrame, y: str, ylabel: str, out: Path, *, dpi: int) -> None:
    import matplotlib.pyplot as plt

    summary = []
    for beta, group in metrics.groupby("beta"):
        med, q1, q3 = median_iqr(group[y])
        summary.append({"beta": float(beta), "median": med, "q1": q1, "q3": q3})
    plot_df = pd.DataFrame(summary).sort_values("beta")
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    x = plot_df["beta"].to_numpy(dtype=float)
    y_med = plot_df["median"].to_numpy(dtype=float)
    y_q1 = plot_df["q1"].to_numpy(dtype=float)
    y_q3 = plot_df["q3"].to_numpy(dtype=float)
    ax.plot(x, y_med, marker="o", color="#2563EB")
    ax.fill_between(x, y_q1, y_q3, color="#93C5FD", alpha=0.45, linewidth=0)
    ax.set_xscale("log")
    ax.set_xlabel("Beta")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=dpi, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def save_stability_plot(stability: pd.DataFrame, out: Path, *, top_k: int, dpi: int) -> None:
    import matplotlib.pyplot as plt

    column = f"top_{top_k}_overlap_fraction"
    plot_df = stability.loc[stability["pair_type"].eq("adjacent")].copy()
    summary = []
    for (beta_a, beta_b), group in plot_df.groupby(["beta_a", "beta_b"]):
        med, q1, q3 = median_iqr(group[column])
        summary.append({"pair": f"{beta_a:g}->{beta_b:g}", "median": med, "q1": q1, "q3": q3})
    summary_df = pd.DataFrame(summary)
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    x = np.arange(summary_df.shape[0])
    y = summary_df["median"].to_numpy(dtype=float)
    yerr = np.vstack(
        [
            y - summary_df["q1"].to_numpy(dtype=float),
            summary_df["q3"].to_numpy(dtype=float) - y,
        ]
    )
    ax.errorbar(x, y, yerr=yerr, marker="o", color="#2563EB", capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(summary_df["pair"], rotation=30, ha="right")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Adjacent beta pair")
    ax.set_ylabel(f"Top-{top_k} overlap fraction")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=dpi, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def write_report(out: Path, summary: dict[str, object], metrics: pd.DataFrame, stability: pd.DataFrame) -> None:
    eff_summary = metrics.groupby("beta")["effective_loci_exp_entropy"].median().sort_index()
    top_summary = metrics.groupby("beta")["top_100_mass"].median().sort_index() if "top_100_mass" in metrics else None
    stability_summary = (
        stability.loc[stability["pair_type"].eq("adjacent")]
        .groupby(["beta_a", "beta_b"])["top_100_overlap_fraction"]
        .median()
        .reset_index()
        if "top_100_overlap_fraction" in stability
        else pd.DataFrame()
    )
    lines = [
        "# Beta Sensitivity Analysis",
        "",
        "Question: does beta control how diffuse or concentrated the Hopfield retrieval distribution is?",
        "",
        "## Inputs",
        "",
        f"- TF rows: `{summary['n_tf_rows']}`",
        f"- AlphaGenome key rows: `{summary['n_keys']}`",
        f"- Query dimension: `{summary['query_dim']}`",
        f"- Beta grid: `{summary['beta_grid']}`",
        f"- Checkpoint: `{summary['checkpoint']}`",
        "",
        "## Main Result",
        "",
        "Median effective loci, defined as `exp(attention entropy)`:",
        "",
        "| Beta | Median effective loci |",
        "|---:|---:|",
    ]
    for beta, value in eff_summary.items():
        lines.append(f"| {float(beta):g} | {float(value):.2f} |")
    if top_summary is not None:
        lines.extend(["", "Median top-100 attention mass:", "", "| Beta | Median top-100 mass |", "|---:|---:|"])
        for beta, value in top_summary.items():
            lines.append(f"| {float(beta):g} | {float(value):.4f} |")
    if not stability_summary.empty:
        lines.extend(["", "Median top-100 overlap between adjacent beta values:", "", "| Beta pair | Median overlap |", "|---|---:|"])
        for row in stability_summary.itertuples(index=False):
            lines.append(f"| {float(row.beta_a):g} -> {float(row.beta_b):g} | {float(row.top_100_overlap_fraction):.3f} |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            str(summary["headline_interpretation"]),
            "",
            "Safe wording: beta is an inverse-temperature parameter controlling retrieval sharpness. These plots do not show TF concentration or binding affinity.",
        ]
    )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    outdir = ensure_dir(args.outdir)
    if args.plot_existing:
        metrics_path = outdir / "beta_sensitivity_metrics.csv"
        stability_path = outdir / "beta_sensitivity_stability.csv"
        if not metrics_path.exists():
            raise FileNotFoundError(metrics_path)
        if not stability_path.exists():
            raise FileNotFoundError(stability_path)
        metrics = pd.read_csv(metrics_path)
        stability = pd.read_csv(stability_path)
        save_metric_plot(
            metrics,
            "effective_loci_exp_entropy",
            "Effective loci, exp(entropy)",
            outdir / "beta_vs_effective_loci.png",
            dpi=args.dpi,
        )
        save_metric_plot(
            metrics,
            "top_100_mass",
            "Top-100 attention mass",
            outdir / "beta_vs_top100_mass.png",
            dpi=args.dpi,
        )
        save_metric_plot(
            metrics,
            "max_attention",
            "Maximum single-locus attention",
            outdir / "beta_vs_max_attention.png",
            dpi=args.dpi,
        )
        save_stability_plot(stability, outdir / "beta_top100_stability.png", top_k=100, dpi=args.dpi)
        print(
            json.dumps(
                {
                    "outdir": str(outdir),
                    "plots": sorted(path.name for path in outdir.glob("*.png")),
                    "source": "existing beta_sensitivity CSV outputs",
                },
                indent=2,
            )
        )
        return

    if args.keys is None:
        raise ValueError("--keys is required unless --plot-existing is set")
    if args.checkpoint is None:
        raise ValueError("--checkpoint is required unless --plot-existing is set")

    beta_grid = parse_number_list(args.beta_grid, cast=float)
    top_ks = sorted(set(parse_number_list(args.top_k, cast=int)))
    device = choose_device(args.device)
    rows, embeddings = load_rows_and_embeddings(args)
    queries, checkpoint_info = load_query_projection(args.checkpoint, embeddings, device)
    keys = load_keys(args.keys, max_keys=args.max_keys)
    if queries.shape[1] != keys.shape[1]:
        raise ValueError(f"query dim {queries.shape[1]} does not match key dim {keys.shape[1]}")
    metrics, top_indices = summarize_beta(
        rows=rows,
        queries=queries,
        keys=keys,
        beta_grid=beta_grid,
        top_ks=top_ks,
        batch_size=args.batch_size,
        device=device,
    )
    stability = summarize_stability(
        rows=rows,
        top_indices=top_indices,
        beta_grid=beta_grid,
        top_ks=top_ks,
        reference_beta=args.reference_beta,
    )
    metrics_path = outdir / "beta_sensitivity_metrics.csv"
    stability_path = outdir / "beta_sensitivity_stability.csv"
    metrics.to_csv(metrics_path, index=False)
    stability.to_csv(stability_path, index=False)

    if not args.skip_plots:
        save_metric_plot(
            metrics,
            "effective_loci_exp_entropy",
            "Effective loci, exp(entropy)",
            outdir / "beta_vs_effective_loci.png",
            dpi=args.dpi,
        )
        save_metric_plot(
            metrics,
            "top_100_mass",
            "Top-100 attention mass",
            outdir / "beta_vs_top100_mass.png",
            dpi=args.dpi,
        )
        save_metric_plot(
            metrics,
            "max_attention",
            "Maximum single-locus attention",
            outdir / "beta_vs_max_attention.png",
            dpi=args.dpi,
        )
        save_stability_plot(stability, outdir / "beta_top100_stability.png", top_k=100, dpi=args.dpi)

    eff_median = metrics.groupby("beta")["effective_loci_exp_entropy"].median().sort_index()
    top100_median = metrics.groupby("beta")["top_100_mass"].median().sort_index()
    if eff_median.iloc[0] > eff_median.iloc[-1] and top100_median.iloc[0] < top100_median.iloc[-1]:
        headline = (
            "Beta produces the expected diffuse-to-concentrated retrieval pattern: low beta spreads weight "
            "across many loci, while high beta concentrates weight into a smaller set of loci."
        )
    else:
        headline = (
            "Beta changes the retrieval distribution, but the diffuse-to-concentrated pattern is not clean. "
            "This should be treated as a diagnostic result rather than strong memory-regime evidence."
        )
    summary = {
        **checkpoint_info,
        "metadata": str(args.metadata),
        "embedding_matrix": str(args.embedding_matrix),
        "vocab": str(args.vocab),
        "keys": str(args.keys),
        "key_metadata": str(args.key_metadata) if args.key_metadata else None,
        "outdir": str(outdir),
        "device": str(device),
        "n_tf_rows": int(rows.shape[0]),
        "n_keys": int(keys.shape[0]),
        "query_dim": int(queries.shape[1]),
        "beta_grid": beta_grid,
        "top_k": top_ks,
        "reference_beta": float(args.reference_beta),
        "metrics_csv": str(metrics_path),
        "stability_csv": str(stability_path),
        "headline_interpretation": headline,
        "median_effective_loci_by_beta": {str(k): float(v) for k, v in eff_median.items()},
        "median_top100_mass_by_beta": {str(k): float(v) for k, v in top100_median.items()},
    }
    write_json(outdir / "beta_sensitivity_summary.json", summary)
    write_report(outdir / "beta_sensitivity_report.md", summary, metrics, stability)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
