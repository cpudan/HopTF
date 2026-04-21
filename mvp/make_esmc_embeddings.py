#!/usr/bin/env python3
# Example:
# python make_esmc_embeddings.py \
#   --vocab /path/to/tf_atlas_morf_isoform_vocab.json \
#   --sequences /path/to/tf_atlas_morf_isoform_protein_sequences.json \
#   --snapshot-dir /path/to/esmc-600m-2024-12 \
#   --out /path/to/esmc_embeddings.npy

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ESMC_MODELS = {
    "300m": {
        "snapshot_name": "esmc-300m-2024-12",
        "builder": "ESMC_300M_202412",
        "weight_path": "data/weights/esmc_300m_2024_12_v0.pth",
    },
    "600m": {
        "snapshot_name": "esmc-600m-2024-12",
        "builder": "ESMC_600M_202412",
        "weight_path": "data/weights/esmc_600m_2024_12_v0.pth",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate ESM-C mean-pooled protein embeddings from a vocab and sequence map."
    )
    parser.add_argument("--vocab", type=Path, required=True)
    parser.add_argument("--sequences", type=Path, required=True)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mask-out", type=Path, default=None)
    parser.add_argument("--meta-out", type=Path, default=None)
    parser.add_argument("--model-size", choices=tuple(ESMC_MODELS), default="600m")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--pool", choices=("cls", "mean_non_special"), default="mean_non_special")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-flash-attn", action="store_true")
    return parser.parse_args()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_vocab(path: Path) -> list[str]:
    payload = load_json(path)
    if not isinstance(payload, list) or not payload:
        raise ValueError("vocab must be a non-empty JSON list")
    return [str(value) for value in payload]


def load_sequence_map(path: Path) -> dict[str, str]:
    """Load the protein sequence lookup keyed by row ID."""
    payload = load_json(path)
    if not isinstance(payload, dict):
        raise ValueError("sequences must be a JSON object")
    return {str(key): str(value) for key, value in payload.items()}


def sanitize_sequence(sequence: str, max_length: int) -> str:
    return "".join(str(sequence).split()).upper()[:max_length]


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def pool_token_embeddings_batch(
    embeddings: np.ndarray,
    lengths: list[int],
    pool: str,
) -> np.ndarray:
    """
    Reduce token-level ESM-C embeddings to one vector per sequence.

    `lengths` are the sanitized amino-acid lengths before special tokens are
    added. The pooled output keeps the input row order.
    """
    arr = np.asarray(embeddings, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"expected batched token embeddings with shape [B, L, D], got {arr.shape}")
    if arr.shape[0] != len(lengths):
        raise ValueError(f"batch/length mismatch: {arr.shape[0]} vs {len(lengths)}")

    rows: list[np.ndarray] = []
    for batch_index, seq_len in enumerate(lengths):
        row = arr[batch_index]
        if pool == "cls":
            rows.append(np.asarray(row[0], dtype=np.float32))
            continue
        token_arr = row[1 : seq_len + 1]
        rows.append(np.asarray(token_arr.mean(axis=0), dtype=np.float32))
    return np.vstack(rows).astype(np.float32, copy=False)


def assert_snapshot_complete(snapshot_dir: Path, model_size: str) -> None:
    """Fail early if the local ESM-C snapshot is missing required files."""
    info = ESMC_MODELS[model_size]
    required = ("README.md", "config.json", info["weight_path"])
    missing = [str(snapshot_dir / rel_path) for rel_path in required if not (snapshot_dir / rel_path).exists()]
    if missing:
        raise FileNotFoundError("ESM-C snapshot is incomplete:\n" + "\n".join(missing))


def load_model(snapshot_dir: Path, model_size: str, device: str, use_flash_attn: bool):
    """Load a local ESM-C snapshot and return the model plus helper classes."""
    import esm.pretrained as pretrained
    import esm.models.esmc as esmc_module
    import esm.utils.constants.esm3 as constants
    from esm.sdk.api import LogitsConfig

    assert_snapshot_complete(snapshot_dir, model_size)

    def local_data_root(_: str) -> Path:
        return snapshot_dir

    constants.data_root = local_data_root
    pretrained.data_root = local_data_root
    builder = getattr(pretrained, ESMC_MODELS[model_size]["builder"])
    model = builder(device, use_flash_attn=use_flash_attn)
    model.eval()
    return model, esmc_module, LogitsConfig


def embed_batch(
    model,
    esmc_module,
    logits_config_class,
    sequences: list[str],
    *,
    pool: str,
    lengths: list[int],
) -> np.ndarray:
    """Run ESM-C on a batch of sequences and return pooled embedding vectors."""
    if not sequences:
        raise ValueError("cannot embed an empty batch")
    tokens = model._tokenize(sequences)
    batch = esmc_module._BatchedESMProteinTensor(sequence=tokens)
    logits_config = logits_config_class(return_embeddings=True)
    import torch

    with torch.inference_mode():
        output = model.logits(batch, logits_config)
    return pool_token_embeddings_batch(
        output.embeddings.detach().float().cpu().numpy(),
        lengths=lengths,
        pool=pool,
    )


def main() -> None:
    args = parse_args()
    genes = load_vocab(args.vocab)
    if args.limit is not None:
        genes = genes[: int(args.limit)]
    sequence_map = load_sequence_map(args.sequences)
    resolved_device = resolve_device(args.device)
    model, esmc_module, logits_config_class = load_model(
        args.snapshot_dir,
        args.model_size,
        resolved_device,
        use_flash_attn=not args.no_flash_attn,
    )

    clean_sequences = [sanitize_sequence(sequence_map.get(gene, ""), args.max_length) for gene in genes]
    valid_indices = [index for index, sequence in enumerate(clean_sequences) if sequence]
    mask = np.zeros(len(genes), dtype=bool)
    truncated_count = sum(1 for gene in genes if len("".join(str(sequence_map.get(gene, "")).split()).upper()) > int(args.max_length))
    matrix: np.ndarray | None = None

    if not valid_indices:
        raise ValueError("no sequences were embedded")

    for start in range(0, len(valid_indices), int(args.batch_size)):
        batch_indices = valid_indices[start : start + int(args.batch_size)]
        batch_sequences = [clean_sequences[index] for index in batch_indices]
        batch_lengths = [len(sequence) for sequence in batch_sequences]
        batch_vectors = embed_batch(
            model,
            esmc_module,
            logits_config_class,
            batch_sequences,
            pool=args.pool,
            lengths=batch_lengths,
        )
        if batch_vectors.ndim != 2:
            raise ValueError(f"expected a 2D embedding batch, got {batch_vectors.shape}")
        if matrix is None:
            matrix = np.zeros((len(genes), int(batch_vectors.shape[1])), dtype=np.float32)
        elif batch_vectors.shape[1] != matrix.shape[1]:
            raise ValueError(f"embedding dimension changed from {matrix.shape[1]} to {batch_vectors.shape[1]}")
        matrix[np.asarray(batch_indices, dtype=np.int64)] = np.asarray(batch_vectors, dtype=np.float32)
        mask[np.asarray(batch_indices, dtype=np.int64)] = True

    assert matrix is not None
    mask_out = args.mask_out or args.out.with_name(args.out.stem + "_mask.npy")
    meta_out = args.meta_out or args.out.with_suffix(".meta.json")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if mask_out.parent != args.out.parent:
        mask_out.parent.mkdir(parents=True, exist_ok=True)
    if meta_out.parent != args.out.parent:
        meta_out.parent.mkdir(parents=True, exist_ok=True)

    np.save(args.out, matrix)
    np.save(mask_out, mask)
    meta_out.write_text(
        json.dumps(
            {
                "model": f"esmc-{args.model_size}",
                "snapshot_dir": str(args.snapshot_dir),
                "vocab": str(args.vocab),
                "sequences": str(args.sequences),
                "out": str(args.out),
                "mask_out": str(mask_out),
                "pool": args.pool,
                "batch_size": int(args.batch_size),
                "max_length": int(args.max_length),
                "truncates_over_length": True,
                "truncated_count": int(truncated_count),
                "limit": None if args.limit is None else int(args.limit),
                "device": resolved_device,
                "n_rows": int(matrix.shape[0]),
                "dimension": int(matrix.shape[1]),
                "n_embedded": int(mask.sum()),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(args.out)


if __name__ == "__main__":
    main()
