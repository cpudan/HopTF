#!/usr/bin/env python3
"""Train Hopfield projection smoke tests for ESM-C to AlphaGenome-style keys."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from hopfield_fitting_common import (
    DEFAULT_ESM,
    DEFAULT_METADATA,
    DEFAULT_OUTDIR,
    DEFAULT_VOCAB,
    ensure_dir,
    load_metadata,
    load_vocab,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("synthetic", "real-overfit"), default="synthetic")
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--embedding-matrix", type=Path, default=DEFAULT_ESM)
    parser.add_argument("--vocab", type=Path, default=DEFAULT_VOCAB)
    parser.add_argument("--keys", type=Path, default=DEFAULT_OUTDIR / "alphagenome_keys.npy")
    parser.add_argument("--latents", type=Path, default=DEFAULT_OUTDIR / "perturbation_latents.npz")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--esm-dim", type=int, default=1152)
    parser.add_argument("--synthetic-key-dim", type=int, default=64)
    parser.add_argument("--synthetic-key-count", type=int, default=128)
    parser.add_argument("--synthetic-pairs", type=int, default=64)
    parser.add_argument("--max-real-rows", type=int, default=64)
    parser.add_argument("--min-cells", type=int, default=5)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=3.0e-3)
    parser.add_argument("--beta", type=float, default=15.0)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def require_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except Exception as exc:
        raise RuntimeError("PyTorch is required for train_hopfield_projection.py") from exc
    return torch, nn, F


class SequenceToHopfieldBase:
    pass


def make_model(esm_dim: int, key_dim: int, beta: float):
    torch, nn, F = require_torch()

    class SequenceToHopfield(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.query = nn.Sequential(nn.Linear(esm_dim, key_dim), nn.LayerNorm(key_dim))
            self.beta = float(beta)

        def forward(self, esm, keys, values):
            q = F.normalize(self.query(esm), dim=-1)
            k = F.normalize(keys, dim=-1)
            logits = self.beta * (q @ k.T)
            attn = logits.softmax(dim=-1)
            retrieved = attn @ values
            return retrieved, attn, q, logits

    return SequenceToHopfield()


def choose_device(value: str):
    torch, _, _ = require_torch()
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def synthetic_arrays(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    rng = np.random.default_rng(args.seed)
    key_count = int(args.synthetic_key_count)
    key_dim = int(args.synthetic_key_dim)
    n_pairs = min(int(args.synthetic_pairs), key_count)
    keys = rng.normal(size=(key_count, key_dim)).astype(np.float32)
    true_w = rng.normal(scale=0.2, size=(key_dim, int(args.esm_dim))).astype(np.float32)
    target_indices = rng.choice(key_count, size=n_pairs, replace=False).astype(np.int64)
    target_keys = keys[target_indices]
    esm = target_keys @ true_w + rng.normal(scale=0.05, size=(n_pairs, int(args.esm_dim))).astype(np.float32)
    values = keys.copy()
    rows = pd.DataFrame(
        {
            "perturbation_id": [f"synthetic_pair_{idx:05d}" for idx in range(n_pairs)],
            "target_index": target_indices,
            "gene_symbol": "synthetic",
            "label_status": "synthetic",
        }
    )
    return esm.astype(np.float32), keys.astype(np.float32), values.astype(np.float32), target_indices, rows


def real_arrays(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    metadata = load_metadata(
        args.metadata,
        require_columns=["perturbation_id", "isoform_embedding_id", "gene_symbol", "n_cells", "response_score"],
    )
    metadata = metadata.loc[pd.to_numeric(metadata["n_cells"], errors="coerce").fillna(0) >= args.min_cells].copy()
    metadata = metadata.sort_values(["label_status", "n_cells", "gene_symbol"], ascending=[False, False, True])

    vocab = load_vocab(args.vocab)
    vocab_index = {value: index for index, value in enumerate(vocab)}
    esm_matrix = np.load(args.embedding_matrix, mmap_mode="r")
    latent = np.load(args.latents, allow_pickle=False)
    latent_ids = {str(value): index for index, value in enumerate(latent["perturbation_id"])}
    key_matrix = np.asarray(np.load(args.keys, mmap_mode="r"), dtype=np.float32)
    keep = []
    for _, row in metadata.iterrows():
        iso_id = str(row["isoform_embedding_id"])
        pert_id = str(row["perturbation_id"])
        if iso_id in vocab_index and pert_id in latent_ids:
            keep.append(row)
        if len(keep) >= args.max_real_rows:
            break
    if not keep:
        raise ValueError("no real rows matched metadata, ESM-C vocab, and latent endpoints")
    rows = pd.DataFrame(keep).reset_index(drop=True)
    if key_matrix.shape[0] < rows.shape[0]:
        raise ValueError(f"need at least {rows.shape[0]} keys for real overfit, got {key_matrix.shape[0]}")
    esm_indices = np.asarray([vocab_index[str(value)] for value in rows["isoform_embedding_id"]], dtype=np.int64)
    latent_indices = np.asarray([latent_ids[str(value)] for value in rows["perturbation_id"]], dtype=np.int64)
    esm = np.asarray(esm_matrix[esm_indices], dtype=np.float32)
    values = np.asarray(latent["latent_pca"][latent_indices], dtype=np.float32)
    keys = np.asarray(key_matrix[: rows.shape[0]], dtype=np.float32)
    target_indices = np.arange(rows.shape[0], dtype=np.int64)
    return esm, keys, values, target_indices, rows


def train(args: argparse.Namespace) -> dict[str, object]:
    torch, _, F = require_torch()
    device = choose_device(args.device)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    if args.mode == "synthetic":
        esm, keys, values, target_indices, rows = synthetic_arrays(args)
    else:
        esm, keys, values, target_indices, rows = real_arrays(args)

    model = make_model(esm.shape[1], keys.shape[1], args.beta).to(device)
    esm_t = torch.as_tensor(np.array(esm, dtype=np.float32, copy=True), dtype=torch.float32, device=device)
    keys_t = torch.as_tensor(np.array(keys, dtype=np.float32, copy=True), dtype=torch.float32, device=device)
    values_t = torch.as_tensor(np.array(values, dtype=np.float32, copy=True), dtype=torch.float32, device=device)
    target_t = torch.as_tensor(target_indices, dtype=torch.long, device=device)
    target_values_t = values_t[target_t]
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1.0e-4)

    history = []
    for step in range(int(args.steps) + 1):
        retrieved, attn, _, logits = model(esm_t, keys_t, values_t)
        ce = F.cross_entropy(logits, target_t)
        mse = F.mse_loss(retrieved, target_values_t)
        loss = ce + mse
        if step == 0 or step == int(args.steps):
            pred = logits.argmax(dim=-1)
            entropy = -(attn * (attn.clamp_min(1.0e-8).log())).sum(dim=-1).mean()
            history.append(
                {
                    "step": int(step),
                    "loss": float(loss.detach().cpu()),
                    "cross_entropy": float(ce.detach().cpu()),
                    "mse": float(mse.detach().cpu()),
                    "top1_accuracy": float((pred == target_t).float().mean().detach().cpu()),
                    "mean_attention_entropy": float(entropy.detach().cpu()),
                }
            )
        if step == int(args.steps):
            break
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    with torch.no_grad():
        retrieved, attn, q, logits = model(esm_t, keys_t, values_t)
        pred = logits.argmax(dim=-1).detach().cpu().numpy()
        top_prob = attn.max(dim=-1).values.detach().cpu().numpy()

    outdir = ensure_dir(args.outdir)
    ckpt_path = outdir / f"hopfield_{args.mode.replace('-', '_')}.pt"
    metrics_path = outdir / f"hopfield_{args.mode.replace('-', '_')}_metrics.json"
    predictions_path = outdir / f"hopfield_{args.mode.replace('-', '_')}_predictions.csv"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "esm_dim": int(esm.shape[1]),
            "key_dim": int(keys.shape[1]),
            "value_dim": int(values.shape[1]),
            "beta": float(args.beta),
            "mode": args.mode,
        },
        ckpt_path,
    )
    pred_df = rows.copy()
    pred_df["target_index"] = target_indices
    pred_df["predicted_index"] = pred
    pred_df["top_attention_probability"] = top_prob
    pred_df["correct_top1"] = pred_df["target_index"] == pred_df["predicted_index"]
    pred_df.to_csv(predictions_path, index=False)
    metrics = {
        "mode": args.mode,
        "device": str(device),
        "n_examples": int(esm.shape[0]),
        "esm_dim": int(esm.shape[1]),
        "key_shape": list(keys.shape),
        "value_shape": list(values.shape),
        "steps": int(args.steps),
        "history": history,
        "checkpoint": ckpt_path,
        "predictions": predictions_path,
        "passed": bool(history[-1]["top1_accuracy"] >= 0.95 and history[-1]["loss"] < history[0]["loss"]),
    }
    write_json(metrics_path, metrics)
    print(f"wrote {metrics_path}; passed={metrics['passed']}")
    if not metrics["passed"]:
        raise SystemExit(2)
    return metrics


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
