#!/usr/bin/env python3
# Example:
# python add_response_score.py \
#   --h5ad /path/to/input_with_pca.h5ad

from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import h5py
import numpy as np


DEFAULT_CONTROL_IDS = ("TFORF3549-GFP", "TFORF3550-mCherry")
PCA_GROUP = "uns/pca"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add control-standardized response scores to an H5AD that already has uns/pca."
    )
    parser.add_argument("--h5ad", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--control-id", action="append", dest="control_ids", default=None)
    return parser.parse_args()


def decode_text(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            value.decode("utf-8") if isinstance(value, (bytes, np.bytes_)) else str(value)
            for value in values
        ],
        dtype=object,
    )


def compute_response_score(
    h5ad_path: Path,
    *,
    control_ids: tuple[str, ...],
) -> dict[str, np.ndarray]:
    """
    Compute responder scores from the perturbation PCA already stored in `uns/pca`.

    The score is the L2 distance from the control centroid after standardizing
    each PCA coordinate by the control standard deviation.
    """
    with h5py.File(h5ad_path, "r") as ad_file:
        if PCA_GROUP not in ad_file:
            raise KeyError(f"missing {PCA_GROUP}")
        pca_group = ad_file[PCA_GROUP]
        if "perturbation_id" not in pca_group or "pca" not in pca_group:
            raise KeyError(f"{PCA_GROUP} must contain perturbation_id and pca")
        perturbation_ids = decode_text(pca_group["perturbation_id"][:])
        states = np.asarray(pca_group["pca"][:], dtype=np.float32)

    control_mask = np.isin(perturbation_ids, list(control_ids))
    if int(control_mask.sum()) < 2:
        raise ValueError("at least two control perturbations are required")

    control_mean = states[control_mask].mean(axis=0).astype(np.float32)
    control_sd = np.maximum(
        states[control_mask].std(axis=0).astype(np.float32),
        np.float32(1.0e-6),
    )
    response_score = np.linalg.norm(
        (states - control_mean[None, :]) / control_sd[None, :],
        axis=1,
    ).astype(np.float32)
    return {
        "control_mean": control_mean,
        "control_sd": control_sd,
        "response_score": response_score,
    }


def main() -> None:
    args = parse_args()
    target_path = args.h5ad if args.out is None else args.out
    if args.out is not None:
        if args.out.exists():
            raise FileExistsError(args.out)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.h5ad, args.out)

    payload = compute_response_score(
        args.h5ad,
        control_ids=tuple(args.control_ids or DEFAULT_CONTROL_IDS),
    )

    with h5py.File(target_path, "r+") as ad_file:
        if PCA_GROUP not in ad_file:
            raise KeyError(f"missing {PCA_GROUP}")
        pca_group = ad_file[PCA_GROUP]
        for key, value in payload.items():
            if key in pca_group:
                raise ValueError(f"{PCA_GROUP}/{key} already exists")
            pca_group.create_dataset(key, data=np.asarray(value))
    print(target_path)


if __name__ == "__main__":
    main()
