#!/usr/bin/env python3
"""Train small sequence-conditioned OT-CFM smoke models."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from hopfield_fitting_common import DEFAULT_ESM, DEFAULT_METADATA, DEFAULT_OUTDIR, DEFAULT_VOCAB, ensure_dir, load_metadata, load_vocab, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("synthetic", "real-overfit", "leave-one-isoform"), default="synthetic")
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--embedding-matrix", type=Path, default=DEFAULT_ESM)
    parser.add_argument("--vocab", type=Path, default=DEFAULT_VOCAB)
    parser.add_argument("--latents", type=Path, default=DEFAULT_OUTDIR / "perturbation_latents.npz")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--holdout-gene", default="HNF4A")
    parser.add_argument("--max-holdouts", type=int, default=2)
    parser.add_argument("--max-train-rows", type=int, default=512)
    parser.add_argument("--min-cells", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=700)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--cond-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--integration-steps", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--disable-torchcfm", action="store_true")
    return parser.parse_args()


def require_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except Exception as exc:
        raise RuntimeError("PyTorch is required for train_otcfm_sequence_conditioned.py") from exc
    return torch, nn, F


def choose_device(value: str):
    torch, _, _ = require_torch()
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def make_model(x_dim: int, esm_dim: int, cond_dim: int, hidden_dim: int):
    torch, nn, _ = require_torch()

    class ConditionalVectorField(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.cond = nn.Sequential(
                nn.Linear(esm_dim, cond_dim),
                nn.LayerNorm(cond_dim),
                nn.GELU(),
                nn.Linear(cond_dim, cond_dim),
                nn.GELU(),
            )
            self.net = nn.Sequential(
                nn.Linear(x_dim + cond_dim + 4, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, x_dim),
            )

        def time_features(self, t):
            return torch.cat([t, t * t, torch.sin(np.pi * t), torch.cos(np.pi * t)], dim=-1)

        def forward(self, t, x, esm):
            c = self.cond(esm)
            return self.net(torch.cat([x, c, self.time_features(t)], dim=-1))

    return ConditionalVectorField()


def standardize(train: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = train.mean(axis=0, keepdims=True).astype(np.float32)
    sd = train.std(axis=0, keepdims=True).astype(np.float32)
    sd = np.maximum(sd, 1.0e-4)
    return ((values - mean) / sd).astype(np.float32), mean, sd


def load_real_dataset(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, np.ndarray, np.ndarray]:
    metadata = load_metadata(
        args.metadata,
        require_columns=["perturbation_id", "isoform_embedding_id", "gene_symbol", "isoform_id", "n_cells", "response_score"],
    )
    metadata = metadata.loc[pd.to_numeric(metadata["n_cells"], errors="coerce").fillna(0) >= args.min_cells].copy()
    vocab = load_vocab(args.vocab)
    vocab_index = {value: index for index, value in enumerate(vocab)}
    esm_matrix = np.load(args.embedding_matrix, mmap_mode="r")
    latent = np.load(args.latents, allow_pickle=False)
    latent_ids = {str(value): index for index, value in enumerate(latent["perturbation_id"])}

    rows = []
    esm_indices = []
    latent_indices = []
    for _, row in metadata.iterrows():
        iso_id = str(row["isoform_embedding_id"])
        pert_id = str(row["perturbation_id"])
        if iso_id in vocab_index and pert_id in latent_ids:
            rows.append(row)
            esm_indices.append(vocab_index[iso_id])
            latent_indices.append(latent_ids[pert_id])
    if not rows:
        raise ValueError("no rows matched metadata, ESM-C vocab, and latent endpoints")
    row_df = pd.DataFrame(rows).reset_index(drop=True)
    esm = np.asarray(esm_matrix[np.asarray(esm_indices, dtype=np.int64)], dtype=np.float32)
    x1 = np.asarray(latent["latent_pca"][np.asarray(latent_indices, dtype=np.int64)], dtype=np.float32)
    control_mean = np.asarray(latent["control_mean"], dtype=np.float32)
    control_sd = np.asarray(latent["control_sd"], dtype=np.float32)
    control_sd = np.maximum(control_sd, 1.0e-3)
    return esm, x1, row_df, control_mean, control_sd


def select_training_indices(
    rows: pd.DataFrame,
    *,
    exclude_index: int | None,
    holdout_gene: str | None,
    max_train_rows: int,
    seed: int,
) -> np.ndarray:
    all_indices = np.arange(rows.shape[0], dtype=np.int64)
    mask = np.ones(rows.shape[0], dtype=bool)
    if exclude_index is not None:
        mask[int(exclude_index)] = False
    available = all_indices[mask]
    if available.shape[0] <= max_train_rows:
        return available

    rng = np.random.default_rng(seed)
    must_keep = np.array([], dtype=np.int64)
    if holdout_gene:
        sibling_mask = rows["gene_symbol"].astype(str).to_numpy() == str(holdout_gene)
        if exclude_index is not None:
            sibling_mask[int(exclude_index)] = False
        must_keep = all_indices[mask & sibling_mask]
    remaining = np.setdiff1d(available, must_keep, assume_unique=False)
    n_remaining = max(0, int(max_train_rows) - must_keep.shape[0])
    sampled = rng.choice(remaining, size=min(n_remaining, remaining.shape[0]), replace=False)
    return np.sort(np.concatenate([must_keep, sampled]).astype(np.int64))


def train_one(
    *,
    esm: np.ndarray,
    x1: np.ndarray,
    control_mean: np.ndarray,
    control_sd: np.ndarray,
    train_idx: np.ndarray,
    eval_idx: np.ndarray,
    args: argparse.Namespace,
    seed: int,
    name: str,
) -> tuple[dict[str, object], np.ndarray]:
    torch, _, F = require_torch()
    device = choose_device(args.device)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    esm_train_std, esm_mean, esm_sd = standardize(esm[train_idx], esm)
    x_train_std, x_mean, x_sd = standardize(x1[train_idx], x1)
    control_mean_std = ((control_mean.reshape(1, -1) - x_mean) / x_sd).astype(np.float32).reshape(-1)
    control_sd_std = (control_sd.reshape(1, -1) / x_sd).astype(np.float32).reshape(-1)

    model = make_model(x1.shape[1], esm.shape[1], args.cond_dim, args.hidden_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1.0e-4)
    torchcfm_available = bool(importlib.util.find_spec("torchcfm"))
    torchcfm_used = False
    cfm = None
    ot_sampler = None
    if torchcfm_available and not args.disable_torchcfm:
        from torchcfm import ConditionalFlowMatcher, OTPlanSampler

        cfm = ConditionalFlowMatcher(sigma=0.0)
        ot_sampler = OTPlanSampler(method="exact", warn=False)
        torchcfm_used = True
    esm_t = torch.as_tensor(esm_train_std, dtype=torch.float32, device=device)
    x1_t = torch.as_tensor(x_train_std, dtype=torch.float32, device=device)
    train_t = torch.as_tensor(train_idx, dtype=torch.long, device=device)
    control_mean_t = torch.as_tensor(control_mean_std, dtype=torch.float32, device=device)
    control_sd_t = torch.as_tensor(control_sd_std, dtype=torch.float32, device=device)
    baseline = float(np.mean((x_train_std[train_idx] - control_mean_std.reshape(1, -1)) ** 2))
    history = []

    for step in range(int(args.steps) + 1):
        batch_size = min(int(args.batch_size), int(train_idx.shape[0]))
        batch_np = rng.choice(train_idx, size=batch_size, replace=train_idx.shape[0] < batch_size)
        batch = torch.as_tensor(batch_np, dtype=torch.long, device=device)
        x1_b = x1_t[batch]
        esm_b = esm_t[batch]
        x0_b = control_mean_t.reshape(1, -1) + torch.randn_like(x1_b) * control_sd_t.reshape(1, -1)
        if cfm is not None and ot_sampler is not None:
            x0_b, x1_b, _, esm_b = ot_sampler.sample_plan_with_labels(x0_b, x1_b, y1=esm_b)
            t_raw, xt, target_v = cfm.sample_location_and_conditional_flow(x0_b, x1_b)
            t = t_raw.reshape(-1, 1)
        else:
            t = torch.rand((batch_size, 1), device=device)
            xt = (1.0 - t) * x0_b + t * x1_b
            target_v = x1_b - x0_b
        pred_v = model(t, xt, esm_b)
        loss = F.mse_loss(pred_v, target_v)
        if step == 0 or step == int(args.steps):
            history.append({"step": int(step), "velocity_mse": float(loss.detach().cpu())})
        if step == int(args.steps):
            break
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    eval_esm = torch.as_tensor(esm_train_std[eval_idx], dtype=torch.float32, device=device)
    x = torch.as_tensor(
        np.repeat(control_mean_std.reshape(1, -1), repeats=int(eval_idx.shape[0]), axis=0),
        dtype=torch.float32,
        device=device,
    )
    with torch.no_grad():
        for step in range(int(args.integration_steps)):
            t_value = (step + 0.5) / float(args.integration_steps)
            t = torch.full((x.shape[0], 1), t_value, dtype=torch.float32, device=device)
            x = x + model(t, x, eval_esm) / float(args.integration_steps)
    pred_std = x.detach().cpu().numpy()
    pred = pred_std * x_sd + x_mean
    endpoint_mse = float(np.mean((pred - x1[eval_idx]) ** 2))
    baseline_endpoint_mse = float(np.mean((control_mean.reshape(1, -1) - x1[eval_idx]) ** 2))
    metrics = {
        "name": name,
        "device": str(device),
        "torchcfm_available": torchcfm_available,
        "torchcfm_used": torchcfm_used,
        "n_train": int(train_idx.shape[0]),
        "n_eval": int(eval_idx.shape[0]),
        "x_dim": int(x1.shape[1]),
        "esm_dim": int(esm.shape[1]),
        "steps": int(args.steps),
        "history": history,
        "train_baseline_standardized_mse": baseline,
        "endpoint_mse": endpoint_mse,
        "baseline_endpoint_mse": baseline_endpoint_mse,
        "endpoint_mse_fraction_of_baseline": endpoint_mse / baseline_endpoint_mse if baseline_endpoint_mse > 0 else None,
    }
    ckpt_path = ensure_dir(args.outdir) / f"otcfm_{name}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "x_dim": int(x1.shape[1]),
            "esm_dim": int(esm.shape[1]),
            "cond_dim": int(args.cond_dim),
            "hidden_dim": int(args.hidden_dim),
            "esm_mean": esm_mean,
            "esm_sd": esm_sd,
            "x_mean": x_mean,
            "x_sd": x_sd,
        },
        ckpt_path,
    )
    metrics["checkpoint"] = str(ckpt_path)
    return metrics, pred


def synthetic_dataset(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(args.seed)
    n = 256
    esm_dim = 64
    x_dim = 16
    esm = rng.normal(size=(n, esm_dim)).astype(np.float32)
    w = rng.normal(scale=0.15, size=(esm_dim, x_dim)).astype(np.float32)
    control_mean = np.zeros(x_dim, dtype=np.float32)
    control_sd = np.ones(x_dim, dtype=np.float32) * 0.1
    x1 = esm @ w + 0.25 * np.tanh(esm[:, :x_dim]) + rng.normal(scale=0.02, size=(n, x_dim)).astype(np.float32)
    rows = pd.DataFrame({"perturbation_id": [f"synthetic_{idx:04d}" for idx in range(n)], "gene_symbol": "synthetic"})
    return esm, x1.astype(np.float32), rows, control_mean, control_sd


def main() -> None:
    args = parse_args()
    outdir = ensure_dir(args.outdir)
    if args.mode == "synthetic":
        esm, x1, rows, control_mean, control_sd = synthetic_dataset(args)
        train_idx = np.arange(200, dtype=np.int64)
        eval_idx = np.arange(200, 256, dtype=np.int64)
        metrics, pred = train_one(
            esm=esm,
            x1=x1,
            control_mean=control_mean,
            control_sd=control_sd,
            train_idx=train_idx,
            eval_idx=eval_idx,
            args=args,
            seed=args.seed,
            name="synthetic",
        )
        metrics["passed"] = bool(metrics["endpoint_mse_fraction_of_baseline"] is not None and metrics["endpoint_mse_fraction_of_baseline"] < 0.35)
        np.savez_compressed(outdir / "otcfm_synthetic_predictions.npz", pred=pred, target=x1[eval_idx], eval_idx=eval_idx)
        write_json(outdir / "otcfm_synthetic_metrics.json", metrics)
        print(f"wrote {outdir / 'otcfm_synthetic_metrics.json'}; passed={metrics['passed']}")
        if not metrics["passed"]:
            raise SystemExit(2)
        return

    esm, x1, rows, control_mean, control_sd = load_real_dataset(args)
    if args.mode == "real-overfit":
        train_idx = select_training_indices(
            rows,
            exclude_index=None,
            holdout_gene=None,
            max_train_rows=int(args.max_train_rows),
            seed=args.seed,
        )
        metrics, pred = train_one(
            esm=esm,
            x1=x1,
            control_mean=control_mean,
            control_sd=control_sd,
            train_idx=train_idx,
            eval_idx=train_idx,
            args=args,
            seed=args.seed,
            name="real_overfit",
        )
        metrics["passed"] = bool(metrics["endpoint_mse_fraction_of_baseline"] is not None and metrics["endpoint_mse_fraction_of_baseline"] < 0.75)
        np.savez_compressed(outdir / "otcfm_real_overfit_predictions.npz", pred=pred, target=x1[train_idx], eval_idx=train_idx)
        write_json(outdir / "otcfm_real_overfit_metrics.json", metrics)
        print(f"wrote {outdir / 'otcfm_real_overfit_metrics.json'}; passed={metrics['passed']}")
        if not metrics["passed"]:
            raise SystemExit(2)
        return

    gene_mask = rows["gene_symbol"].astype(str).to_numpy() == str(args.holdout_gene)
    holdout_candidates = np.flatnonzero(gene_mask)
    if holdout_candidates.size == 0:
        raise ValueError(f"no rows found for holdout gene {args.holdout_gene}")
    scores = pd.to_numeric(rows.iloc[holdout_candidates]["response_score"], errors="coerce").fillna(-np.inf).to_numpy()
    order = np.argsort(-scores)
    holdouts = holdout_candidates[order[: int(args.max_holdouts)]]
    records = []
    all_predictions = []
    for run_idx, holdout in enumerate(holdouts):
        train_idx = select_training_indices(
            rows,
            exclude_index=int(holdout),
            holdout_gene=str(args.holdout_gene),
            max_train_rows=int(args.max_train_rows),
            seed=args.seed + run_idx,
        )
        metrics, pred = train_one(
            esm=esm,
            x1=x1,
            control_mean=control_mean,
            control_sd=control_sd,
            train_idx=train_idx,
            eval_idx=np.asarray([holdout], dtype=np.int64),
            args=args,
            seed=args.seed + run_idx,
            name=f"leave_one_{args.holdout_gene}_{run_idx}",
        )
        row = rows.iloc[int(holdout)].to_dict()
        row.update(metrics)
        row["holdout_index"] = int(holdout)
        row["passed"] = bool(metrics["endpoint_mse"] < metrics["baseline_endpoint_mse"])
        records.append(row)
        all_predictions.append(pred[0])

    results = pd.DataFrame(records)
    results_path = outdir / f"otcfm_leave_one_{args.holdout_gene}_metrics.csv"
    json_path = outdir / f"otcfm_leave_one_{args.holdout_gene}_summary.json"
    results.to_csv(results_path, index=False)
    np.savez_compressed(
        outdir / f"otcfm_leave_one_{args.holdout_gene}_predictions.npz",
        pred=np.asarray(all_predictions, dtype=np.float32),
        holdout_idx=holdouts,
        target=x1[holdouts],
    )
    summary = {
        "mode": args.mode,
        "holdout_gene": args.holdout_gene,
        "n_holdouts": int(len(records)),
        "n_passed": int(results["passed"].sum()),
        "mean_endpoint_mse_fraction_of_baseline": float(results["endpoint_mse_fraction_of_baseline"].mean()),
        "metrics_csv": results_path,
        "passed": bool(results["passed"].all()),
    }
    write_json(json_path, summary)
    print(f"wrote {results_path} {json_path}; passed={summary['passed']}")
    if not summary["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
