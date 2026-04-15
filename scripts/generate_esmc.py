from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from mantra.embeddings.generate_esm3 import POOL_MODES, pool_token_embeddings, sanitize_sequence


DEFAULT_META_DIR = Path("data/processed/metadata")
DEFAULT_SNAPSHOT_ROOT = Path("data/raw/protein/esmc")
DEFAULT_OUT_ROOT = Path("data/processed/embeddings/registry/protein_sequence/source_gene_artifacts")
DEFAULT_MODEL_SIZE = "300m"
DEFAULT_POOL = "mean_non_special"

ESMC_MODELS: dict[str, dict[str, Any]] = {
    "300m": {
        "repo_id": "EvolutionaryScale/esmc-300m-2024-12",
        "revision": "a19d363f07313a10a64d08a2d6b41376a73df5c8",
        "snapshot_name": "esmc-300m-2024-12",
        "weight_path": "data/weights/esmc_300m_2024_12_v0.pth",
        "builder": "ESMC_300M_202412",
        "dimension": 960,
    },
    "600m": {
        "repo_id": "EvolutionaryScale/esmc-600m-2024-12",
        "revision": "d11cc14d44078eaecbc6a843d5eb20f4eecc1e7e",
        "snapshot_name": "esmc-600m-2024-12",
        "weight_path": "data/weights/esmc_600m_2024_12_v0.pth",
        "builder": "ESMC_600M_202412",
        "dimension": 1152,
    },
}

# Example invocation:
# PYTHONPATH=src:. python -m mantra.embeddings.generate_esmc \
#   --device cuda \
#   --vocab-path /tmp/mantra_one_tf_vocab.json \
#   --sequence-path data/processed/metadata/regulator_sequences.json \
#   --out-dir /tmp/mantra_esm3_gata1 \
#   --out-prefix GATA1_esm3


class SequenceExtractor(Protocol):
    def embed(self, sequence: str) -> np.ndarray:
        raise NotImplementedError


@dataclass(frozen=True)
class EsmcGenerationConfig:
    model_size: str = DEFAULT_MODEL_SIZE
    meta_dir: Path = DEFAULT_META_DIR
    snapshot_root: Path = DEFAULT_SNAPSHOT_ROOT
    snapshot_dir: Path | None = None
    out_root: Path = DEFAULT_OUT_ROOT
    out_dir: Path | None = None
    out_prefix: str | None = None
    vocab_path: Path | None = None
    sequence_path: Path | None = None
    device: str = "auto"
    pool: str = DEFAULT_POOL
    max_length: int = 2048
    limit: int | None = None
    use_flash_attn: bool = True

    @property
    def model_info(self) -> dict[str, Any]:
        if self.model_size not in ESMC_MODELS:
            raise ValueError(f"model_size must be one of {tuple(ESMC_MODELS)}; got {self.model_size!r}")
        return ESMC_MODELS[self.model_size]

    @property
    def resolved_snapshot_dir(self) -> Path:
        return self.snapshot_dir if self.snapshot_dir is not None else self.snapshot_root / self.model_info["snapshot_name"]

    @property
    def resolved_out_dir(self) -> Path:
        return self.out_dir if self.out_dir is not None else self.out_root / self.model_info["snapshot_name"].replace("-", "_")

    @property
    def resolved_out_prefix(self) -> str:
        return self.out_prefix if self.out_prefix is not None else f"{self.model_info['snapshot_name'].replace('-', '_')}_{self.pool}"

    @property
    def resolved_vocab_path(self) -> Path:
        return self.vocab_path if self.vocab_path is not None else self.meta_dir / "vocab_k562_gwps_full.json"

    @property
    def resolved_sequence_path(self) -> Path:
        return self.sequence_path if self.sequence_path is not None else self.meta_dir / "regulator_sequences.json"

    @property
    def embedding_path(self) -> Path:
        return self.resolved_out_dir / f"{self.resolved_out_prefix}.npy"

    @property
    def mask_path(self) -> Path:
        return self.resolved_out_dir / f"{self.resolved_out_prefix}_mask.npy"

    @property
    def metadata_path(self) -> Path:
        return self.resolved_out_dir / f"{self.resolved_out_prefix}.meta.json"


@dataclass
class EsmcSequenceExtractor:
    model_size: str = DEFAULT_MODEL_SIZE
    snapshot_root: Path = DEFAULT_SNAPSHOT_ROOT
    snapshot_dir: Path | None = None
    device: str = "auto"
    pool: str = DEFAULT_POOL
    max_length: int = 2048
    use_flash_attn: bool = True
    _model: Any = field(default=None, init=False, repr=False)
    _esm_protein: Any = field(default=None, init=False, repr=False)
    _logits_config: Any = field(default=None, init=False, repr=False)
    _resolved_device: str | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.model_size not in ESMC_MODELS:
            raise ValueError(f"model_size must be one of {tuple(ESMC_MODELS)}; got {self.model_size!r}")
        if self.pool not in POOL_MODES:
            raise ValueError(f"pool must be one of {POOL_MODES}; got {self.pool!r}")
        if self.max_length <= 0:
            raise ValueError("max_length must be positive")

    @property
    def model_info(self) -> dict[str, Any]:
        return ESMC_MODELS[self.model_size]

    @property
    def resolved_snapshot_dir(self) -> Path:
        return self.snapshot_dir if self.snapshot_dir is not None else self.snapshot_root / self.model_info["snapshot_name"]

    @property
    def model(self) -> Any:
        if self._model is None:
            self._load_model()
        return self._model

    @property
    def resolved_device(self) -> str:
        if self._resolved_device is None:
            self._resolved_device = resolve_device(self.device)
        return self._resolved_device

    def _load_model(self) -> None:
        import esm.pretrained as pretrained
        import esm.utils.constants.esm3 as constants
        from esm.sdk.api import ESMProtein, LogitsConfig

        snapshot_dir = self.resolved_snapshot_dir
        assert_snapshot_complete(snapshot_dir, model_size=self.model_size)

        def local_data_root(model: str) -> Path:
            if str(model).startswith("esmc-300"):
                return snapshot_dir if self.model_size == "300m" else self.snapshot_root / ESMC_MODELS["300m"]["snapshot_name"]
            if str(model).startswith("esmc-600"):
                return snapshot_dir if self.model_size == "600m" else self.snapshot_root / ESMC_MODELS["600m"]["snapshot_name"]
            return snapshot_dir

        constants.data_root = local_data_root
        pretrained.data_root = local_data_root
        builder = getattr(pretrained, self.model_info["builder"])
        self._esm_protein = ESMProtein
        self._logits_config = LogitsConfig
        self._model = builder(self.resolved_device, use_flash_attn=self.use_flash_attn)
        self._model.eval()

    def embed(self, sequence: str) -> np.ndarray:
        sequence_clean = sanitize_sequence(sequence, self.max_length)
        if not sequence_clean:
            raise ValueError("cannot embed an empty sequence")
        protein = self._esm_protein(sequence=sequence_clean) if self._esm_protein is not None else None
        if protein is None:
            self._load_model()
            protein = self._esm_protein(sequence=sequence_clean)
        tokens = self.model.encode(protein)
        logits_config = self._logits_config(return_embeddings=True)
        import torch

        with torch.inference_mode():
            output = self.model.logits(tokens, logits_config)
        embedding = output.embeddings.detach().float().cpu().numpy()
        return pool_token_embeddings(embedding, pool=self.pool)


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_vocab(path: Path) -> list[str]:
    payload = load_json(path)
    if not isinstance(payload, list) or not all(isinstance(item, str) for item in payload):
        raise ValueError(f"vocab must be a JSON string list: {path}")
    return payload


def load_sequence_map(path: Path) -> dict[str, str]:
    payload = load_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"sequence map must be a JSON object: {path}")
    return {str(key): str(value) for key, value in payload.items() if str(value)}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_snapshot_complete(snapshot_dir: Path, *, model_size: str) -> None:
    if model_size not in ESMC_MODELS:
        raise ValueError(f"model_size must be one of {tuple(ESMC_MODELS)}; got {model_size!r}")
    required = (
        ".gitattributes",
        "README.md",
        "config.json",
        ESMC_MODELS[model_size]["weight_path"],
    )
    missing = [str(snapshot_dir / rel_path) for rel_path in required if not (snapshot_dir / rel_path).exists()]
    if missing:
        raise FileNotFoundError("ESM-C snapshot is incomplete:\n" + "\n".join(missing))


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in ("esm", "torch", "numpy"):
        try:
            module = __import__(package)
            versions[package] = getattr(module, "__version__", None)
        except Exception:
            versions[package] = None
    return versions


def build_metadata(
    *,
    config: EsmcGenerationConfig,
    genes: list[str],
    mask: np.ndarray,
    dim: int,
    embedding_path: Path,
) -> dict[str, Any]:
    info = config.model_info
    return {
        "schema_version": "1.0",
        "generated_utc": utc_now_iso(),
        "model": info["snapshot_name"],
        "model_size": config.model_size,
        "hf_repo_id": info["repo_id"],
        "snapshot_dir": str(config.resolved_snapshot_dir),
        "snapshot_commit": info["revision"],
        "pool": config.pool,
        "max_length": config.max_length,
        "device": config.device,
        "limit": config.limit,
        "use_flash_attn": config.use_flash_attn,
        "vocab_path": str(config.resolved_vocab_path),
        "sequence_path": str(config.resolved_sequence_path),
        "vocab_sha256": sha256_file(config.resolved_vocab_path),
        "sequence_sha256": sha256_file(config.resolved_sequence_path),
        "embedding_path": str(embedding_path),
        "mask_path": str(config.mask_path),
        "n_vocab": len(genes),
        "n_embedded": int(mask.sum()),
        "dimension": int(dim),
        "expected_dimension": int(info["dimension"]),
        "dtype": "float32",
        "packages": package_versions(),
    }


def generate_esmc_embeddings(
    config: EsmcGenerationConfig,
    *,
    extractor: SequenceExtractor | None = None,
) -> dict[str, Any]:
    if config.pool not in POOL_MODES:
        raise ValueError(f"pool must be one of {POOL_MODES}; got {config.pool!r}")
    genes_all = load_vocab(config.resolved_vocab_path)
    genes = genes_all[: config.limit] if config.limit is not None else genes_all
    seq_map = load_sequence_map(config.resolved_sequence_path)
    sequence_extractor = extractor or EsmcSequenceExtractor(
        model_size=config.model_size,
        snapshot_root=config.snapshot_root,
        snapshot_dir=config.snapshot_dir,
        device=config.device,
        pool=config.pool,
        max_length=config.max_length,
        use_flash_attn=config.use_flash_attn,
    )
    rows: list[np.ndarray] = []
    mask = np.zeros(len(genes), dtype=bool)
    dim: int | None = None
    for idx, gene in enumerate(genes):
        sequence = sanitize_sequence(seq_map.get(gene, ""), config.max_length)
        if not sequence:
            if dim is None:
                rows.append(np.zeros((0,), dtype=np.float32))
            else:
                rows.append(np.zeros((dim,), dtype=np.float32))
            continue
        vector = np.asarray(sequence_extractor.embed(sequence), dtype=np.float32)
        if vector.ndim != 1:
            raise ValueError(f"extractor returned non-vector embedding for {gene}: {vector.shape}")
        if dim is None:
            dim = int(vector.shape[0])
            rows = [np.zeros((dim,), dtype=np.float32) if row.size == 0 else row for row in rows]
        elif vector.shape[0] != dim:
            raise ValueError(f"extractor dimension changed from {dim} to {vector.shape[0]} for {gene}")
        rows.append(vector)
        mask[idx] = True
    if dim is None:
        raise ValueError("no sequences were embedded; check the vocabulary and sequence map")
    matrix = np.vstack(rows).astype(np.float32, copy=False)
    config.resolved_out_dir.mkdir(parents=True, exist_ok=True)
    np.save(config.embedding_path, matrix)
    np.save(config.mask_path, mask)
    metadata = build_metadata(
        config=config,
        genes=genes,
        mask=mask,
        dim=dim,
        embedding_path=config.embedding_path,
    )
    config.metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def parse_args(argv: list[str] | None = None) -> EsmcGenerationConfig:
    parser = argparse.ArgumentParser(description="Generate local ESM-C protein embeddings.")
    parser.add_argument("--model-size", choices=tuple(ESMC_MODELS), default=DEFAULT_MODEL_SIZE)
    parser.add_argument("--meta-dir", type=Path, default=DEFAULT_META_DIR)
    parser.add_argument("--snapshot-root", type=Path, default=DEFAULT_SNAPSHOT_ROOT)
    parser.add_argument("--snapshot-dir", type=Path)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--out-prefix", type=str)
    parser.add_argument("--vocab-path", type=Path)
    parser.add_argument("--sequence-path", type=Path)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--pool", choices=POOL_MODES, default=DEFAULT_POOL)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--no-flash-attn", action="store_true")
    args = parser.parse_args(argv)
    return EsmcGenerationConfig(
        model_size=args.model_size,
        meta_dir=args.meta_dir,
        snapshot_root=args.snapshot_root,
        snapshot_dir=args.snapshot_dir,
        out_root=args.out_root,
        out_dir=args.out_dir,
        out_prefix=args.out_prefix,
        vocab_path=args.vocab_path,
        sequence_path=args.sequence_path,
        device=args.device,
        pool=args.pool,
        max_length=args.max_length,
        limit=args.limit,
        use_flash_attn=not args.no_flash_attn,
    )


def main(argv: list[str] | None = None) -> int:
    metadata = generate_esmc_embeddings(parse_args(argv))
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
