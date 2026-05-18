#!/usr/bin/env python3
"""Generate aligned ESM-DBP protein embeddings for HopTF isoforms."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_METADATA = Path("data/processed/linear_probe/tfatlas_subsample/PERTURBATION_METADATA_hard_local_subsample.csv")
DEFAULT_VOCAB = Path(
    "data/hf_min/processed/protein_embeddings/"
    "tf_atlas_morf_isoforms_esmc_600m/metadata/tf_atlas_morf_isoform_vocab.json"
)
DEFAULT_MODEL = Path("/gpfs/commons/groups/knowles_lab/data/ESM/ESM-DBP/ESM-DBP.model")
DEFAULT_OUT = Path(
    "data/hf_min/processed/protein_embeddings/"
    "tf_atlas_morf_isoforms_esm_dbp/tf_atlas_morf_isoforms_esm_dbp_mean_non_special.npy"
)
DEFAULT_VOCAB_OUT = Path(
    "data/hf_min/processed/protein_embeddings/"
    "tf_atlas_morf_isoforms_esm_dbp/metadata/tf_atlas_morf_isoform_vocab.json"
)
POOL_MODES = ("cls", "mean_non_special")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--vocab", type=Path, default=DEFAULT_VOCAB)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--vocab-out", type=Path, default=DEFAULT_VOCAB_OUT)
    parser.add_argument("--mask-out", type=Path, default=None)
    parser.add_argument("--meta-out", type=Path, default=None)
    parser.add_argument("--sequence-out", type=Path, default=None)
    parser.add_argument("--pool", choices=POOL_MODES, default="mean_non_special")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=1022)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024 * 8), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sanitize_sequence(sequence: str, max_length: int) -> str:
    cleaned = "".join(str(sequence).split()).upper()
    return cleaned[:max_length]


def resolve_device(value: str):
    import torch

    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def load_fair_esm2_model(model_path: Path, device):
    try:
        from esm.data import Alphabet
        from esm.model.esm2 import ESM2
    except Exception as exc:
        raise RuntimeError(
            "ESM-DBP generation requires the legacy fair-esm package. "
            "Install it in the ESM-DBP runtime with: python -m pip install --user fair-esm"
        ) from exc

    import torch

    alphabet = Alphabet.from_architecture("ESM-1b")
    model = ESM2(num_layers=33, embed_dim=1280, attention_heads=20, alphabet=alphabet, token_dropout=True)
    state = torch.load(model_path, map_location="cpu")
    if not isinstance(state, dict):
        raise TypeError(f"expected a state dict in {model_path}, got {type(state)!r}")
    state = {key.removeprefix("module."): value for key, value in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    allowed_missing: set[str] = set()
    allowed_unexpected: set[str] = set()
    missing_bad = sorted(set(missing) - allowed_missing)
    unexpected_bad = sorted(set(unexpected) - allowed_unexpected)
    if missing_bad or unexpected_bad:
        raise RuntimeError(
            "ESM-DBP checkpoint did not match fair-esm ESM2 architecture: "
            f"missing={missing_bad[:20]} unexpected={unexpected_bad[:20]}"
        )
    model.eval().to(device)
    return model, alphabet


def sequence_map_from_metadata(metadata: Path) -> dict[str, str]:
    df = pd.read_csv(metadata)
    required = {"isoform_embedding_id", "sequence"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"metadata is missing required columns: {sorted(missing)}")
    out: dict[str, str] = {}
    for _, row in df.iterrows():
        isoform_id = str(row["isoform_embedding_id"])
        sequence = "" if pd.isna(row["sequence"]) else str(row["sequence"])
        if isoform_id not in out or (not out[isoform_id] and sequence):
            out[isoform_id] = sequence
    return out


def pool_representations(representations, lengths: list[int], *, pool: str) -> np.ndarray:
    rows = []
    for index, seq_len in enumerate(lengths):
        if pool == "cls":
            rows.append(representations[index, 0].detach().float().cpu().numpy())
        else:
            token_block = representations[index, 1 : seq_len + 1]
            rows.append(token_block.mean(dim=0).detach().float().cpu().numpy())
    return np.vstack(rows).astype(np.float32, copy=False)


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.max_length <= 0:
        raise ValueError("--max-length must be positive")
    if not args.model.exists():
        raise FileNotFoundError(args.model)

    import torch

    vocab = [str(value) for value in load_json(args.vocab)]
    if args.limit is not None:
        vocab = vocab[: int(args.limit)]
    sequence_map = sequence_map_from_metadata(args.metadata)
    device = resolve_device(args.device)
    model, alphabet = load_fair_esm2_model(args.model, device)
    batch_converter = alphabet.get_batch_converter(truncation_seq_length=int(args.max_length))

    clean_sequences = [sanitize_sequence(sequence_map.get(isoform_id, ""), int(args.max_length)) for isoform_id in vocab]
    valid_indices = [index for index, sequence in enumerate(clean_sequences) if sequence]
    mask = np.zeros(len(vocab), dtype=bool)
    truncated_count = sum(
        1
        for isoform_id in vocab
        if len("".join(str(sequence_map.get(isoform_id, "")).split()).upper()) > int(args.max_length)
    )
    matrix = np.zeros((len(vocab), 1280), dtype=np.float32)

    if not valid_indices:
        raise ValueError("no valid sequences found for ESM-DBP embedding generation")

    for start in range(0, len(valid_indices), int(args.batch_size)):
        batch_indices = valid_indices[start : start + int(args.batch_size)]
        batch_records = [(vocab[index], clean_sequences[index]) for index in batch_indices]
        _, _, tokens = batch_converter(batch_records)
        lengths = [len(sequence) for _, sequence in batch_records]
        tokens = tokens.to(device)
        with torch.inference_mode():
            output = model(tokens, repr_layers=[33], return_contacts=False)
        vectors = pool_representations(output["representations"][33], lengths, pool=args.pool)
        matrix[np.asarray(batch_indices, dtype=np.int64)] = vectors
        mask[np.asarray(batch_indices, dtype=np.int64)] = True
        print(
            f"[{datetime.now().isoformat(timespec='seconds')}] "
            f"embedded {min(start + len(batch_indices), len(valid_indices))}/{len(valid_indices)}",
            flush=True,
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.vocab_out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, matrix)
    write_json(args.vocab_out, vocab)

    mask_out = args.mask_out or args.out.with_name(args.out.stem + "_mask.npy")
    meta_out = args.meta_out or args.out.with_suffix(".meta.json")
    sequence_out = args.sequence_out or args.vocab_out.with_name("tf_atlas_morf_isoform_sequences.json")
    mask_out.parent.mkdir(parents=True, exist_ok=True)
    meta_out.parent.mkdir(parents=True, exist_ok=True)
    np.save(mask_out, mask)
    write_json(sequence_out, {isoform_id: sequence_map.get(isoform_id, "") for isoform_id in vocab})

    metadata = {
        "encoder": "ESM-DBP",
        "source_model": "zengwenwu/ESM-DBP",
        "model_path": str(args.model),
        "model_sha256": sha256_file(args.model),
        "architecture": "fair-esm ESM2 33-layer 650M, ESM-1b alphabet",
        "embedding_dim": 1280,
        "pool": args.pool,
        "max_length": int(args.max_length),
        "device": str(device),
        "n_vocab": len(vocab),
        "n_embedded": int(mask.sum()),
        "n_missing_sequence": int((~mask).sum()),
        "n_truncated": int(truncated_count),
        "embedding_path": str(args.out),
        "vocab_path": str(args.vocab_out),
        "mask_path": str(mask_out),
        "sequence_path": str(sequence_out),
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
    }
    write_json(meta_out, metadata)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
