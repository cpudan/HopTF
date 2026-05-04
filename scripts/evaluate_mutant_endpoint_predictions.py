#!/usr/bin/env python3
"""Predict frozen OT-CFM endpoints for verified mutant ESM-C embeddings when available."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from hopfield_fitting_common import DEFAULT_ESM, DEFAULT_OUTDIR, DEFAULT_VOCAB, ensure_dir, load_vocab, write_json
from train_otcfm_sequence_conditioned import choose_device, endpoint_metrics, integrate_from_control, make_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mutant-metadata", type=Path, default=DEFAULT_OUTDIR / "mutant_sequences_metadata.csv")
    parser.add_argument("--mutant-embedding-matrix", type=Path, default=DEFAULT_OUTDIR / "mutant_esmc_embeddings.npy")
    parser.add_argument("--mutant-vocab", type=Path, default=DEFAULT_OUTDIR / "mutant_esmc_vocab.json")
    parser.add_argument("--wt-embedding-matrix", type=Path, default=DEFAULT_ESM)
    parser.add_argument("--wt-vocab", type=Path, default=DEFAULT_VOCAB)
    parser.add_argument("--latents", type=Path, default=DEFAULT_OUTDIR / "perturbation_latents.npz")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_OUTDIR / "otcfm_real_overfit.pt")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--integration-steps", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--strict", action="store_true", help="Exit nonzero if mutant embeddings are unavailable.")
    return parser.parse_args()


def missing_inputs(args: argparse.Namespace) -> list[str]:
    required = [
        args.mutant_metadata,
        args.mutant_embedding_matrix,
        args.mutant_vocab,
        args.wt_embedding_matrix,
        args.wt_vocab,
        args.latents,
        args.checkpoint,
    ]
    return [str(path) for path in required if not path.exists()]


def predict_endpoints(
    *,
    checkpoint: Path,
    latents: Path,
    embeddings: np.ndarray,
    integration_steps: int,
    device_value: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import torch

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = make_model(
        int(payload["x_dim"]),
        int(payload["esm_dim"]),
        int(payload["cond_dim"]),
        int(payload["hidden_dim"]),
    )
    model.load_state_dict(payload["model_state_dict"])
    device = choose_device(device_value)
    model.to(device)
    model.eval()

    latent = np.load(latents, allow_pickle=False)
    control_mean = np.asarray(latent["control_mean"], dtype=np.float32)
    control_sd = np.maximum(np.asarray(latent["control_sd"], dtype=np.float32), 1.0e-3)
    esm_mean = np.asarray(payload["esm_mean"], dtype=np.float32)
    esm_sd = np.asarray(payload["esm_sd"], dtype=np.float32)
    x_mean = np.asarray(payload["x_mean"], dtype=np.float32)
    x_sd = np.asarray(payload["x_sd"], dtype=np.float32)
    esm_std = ((np.asarray(embeddings, dtype=np.float32) - esm_mean) / esm_sd).astype(np.float32)
    control_mean_std = ((control_mean.reshape(1, -1) - x_mean) / x_sd).astype(np.float32).reshape(-1)

    with torch.no_grad():
        endpoint_std = integrate_from_control(
            model,
            torch.as_tensor(esm_std, dtype=torch.float32, device=device),
            torch.as_tensor(control_mean_std, dtype=torch.float32, device=device),
            n_steps=int(integration_steps),
        )
    endpoint = endpoint_std.detach().cpu().numpy() * x_sd + x_mean
    return endpoint.astype(np.float32), control_mean, control_sd


def main() -> None:
    args = parse_args()
    outdir = ensure_dir(args.outdir)
    predictions_path = outdir / "mutant_endpoint_predictions.csv"
    report_path = outdir / "mutant_endpoint_predictions_report.json"
    missing = missing_inputs(args)
    if missing:
        report = {
            "status": "blocked",
            "reason": "required mutant endpoint prediction inputs are missing",
            "missing": missing,
            "expected_mutant_embedding_inputs": {
                "mutant_embedding_matrix": str(args.mutant_embedding_matrix),
                "mutant_vocab": str(args.mutant_vocab),
            },
            "predictions_csv": str(predictions_path),
        }
        pd.DataFrame().to_csv(predictions_path, index=False)
        write_json(report_path, report)
        print(f"wrote {predictions_path} {report_path}; status=blocked")
        if args.strict:
            raise SystemExit(2)
        return

    metadata = pd.read_csv(args.mutant_metadata)
    verified = metadata.loc[metadata["status"].astype(str) == "ok"].copy()
    if verified.empty:
        report = {
            "status": "blocked",
            "reason": "no verified mutant rows were present",
            "mutant_metadata": str(args.mutant_metadata),
            "predictions_csv": str(predictions_path),
        }
        pd.DataFrame().to_csv(predictions_path, index=False)
        write_json(report_path, report)
        print(f"wrote {predictions_path} {report_path}; status=blocked")
        if args.strict:
            raise SystemExit(2)
        return

    mutant_vocab = load_vocab(args.mutant_vocab)
    wt_vocab = load_vocab(args.wt_vocab)
    mutant_index = {value: index for index, value in enumerate(mutant_vocab)}
    wt_index = {value: index for index, value in enumerate(wt_vocab)}
    mutant_matrix = np.load(args.mutant_embedding_matrix, mmap_mode="r")
    wt_matrix = np.load(args.wt_embedding_matrix, mmap_mode="r")

    records = []
    mutant_vectors = []
    wt_vectors = []
    used_rows = []
    for _, row in verified.iterrows():
        mutant_id = str(row["mutant_embedding_id"])
        wt_id = str(row["isoform_embedding_id"])
        if mutant_id not in mutant_index or wt_id not in wt_index:
            records.append({**row.to_dict(), "prediction_status": "skipped", "prediction_reason": "embedding id missing from vocab"})
            continue
        mutant_vectors.append(np.asarray(mutant_matrix[mutant_index[mutant_id]], dtype=np.float32))
        wt_vectors.append(np.asarray(wt_matrix[wt_index[wt_id]], dtype=np.float32))
        used_rows.append(row)
    if used_rows:
        mutant_pred, control_mean, control_sd = predict_endpoints(
            checkpoint=args.checkpoint,
            latents=args.latents,
            embeddings=np.vstack(mutant_vectors),
            integration_steps=int(args.integration_steps),
            device_value=args.device,
        )
        wt_pred, _, _ = predict_endpoints(
            checkpoint=args.checkpoint,
            latents=args.latents,
            embeddings=np.vstack(wt_vectors),
            integration_steps=int(args.integration_steps),
            device_value=args.device,
        )
        for row, mutant_endpoint, wt_endpoint in zip(used_rows, mutant_pred, wt_pred, strict=True):
            mutant_metrics = endpoint_metrics(
                pred=mutant_endpoint.reshape(1, -1),
                target=wt_endpoint.reshape(1, -1),
                control_mean=control_mean,
                control_sd=control_sd,
            )
            wt_response_l2 = float(
                np.linalg.norm((wt_endpoint.reshape(1, -1) - control_mean.reshape(1, -1)) / control_sd.reshape(1, -1), axis=1)[0]
            )
            mutant_response_l2 = float(
                np.linalg.norm(
                    (mutant_endpoint.reshape(1, -1) - control_mean.reshape(1, -1)) / control_sd.reshape(1, -1),
                    axis=1,
                )[0]
            )
            records.append(
                {
                    **row.to_dict(),
                    "prediction_status": "ok",
                    "prediction_reason": "",
                    "wt_predicted_response_l2": wt_response_l2,
                    "mutant_predicted_response_l2": mutant_response_l2,
                    "mutant_minus_wt_predicted_response_l2": mutant_response_l2 - wt_response_l2,
                    "mutant_vs_wt_endpoint_mse": mutant_metrics["endpoint_mse"],
                    "mutant_vs_wt_control_standardized_endpoint_mse": mutant_metrics[
                        "control_standardized_endpoint_mse"
                    ],
                }
            )

    result = pd.DataFrame(records)
    result.to_csv(predictions_path, index=False)
    report = {
        "status": "ok" if not result.empty and (result["prediction_status"] == "ok").any() else "blocked",
        "n_verified_mutants": int(verified.shape[0]),
        "n_predicted": int((result["prediction_status"] == "ok").sum()) if not result.empty else 0,
        "predictions_csv": str(predictions_path),
    }
    write_json(report_path, report)
    print(f"wrote {predictions_path} {report_path}; status={report['status']}")
    if args.strict and report["status"] != "ok":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
