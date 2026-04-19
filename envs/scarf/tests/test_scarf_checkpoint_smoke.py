from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
import torch
from transformers import BertConfig

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scarf import TotalModel_downstream


DEFAULT_CHECKPOINT_DIR = REPO_ROOT / "data" / "reference" / "scarf" / "weights"
REQUIRED_WEIGHT_FILES = (
    "config.json",
    "pytorch_model.bin.index.json",
    "pytorch_model-00001-of-00002.bin",
    "pytorch_model-00002-of-00002.bin",
)
ALLOWED_UNEXPECTED_KEY_PREFIXES = (
    "loss_R2R.",
    "encoder_rna.cls4value_rec.",
    "encoder_rna.cls4clsGeneAlign.",
    "encoder_rna.kl_enc.",
    "encoder_atac.",
)


def _load_state_dict_from_index(checkpoint_dir: Path) -> dict[str, torch.Tensor]:
    index_file = checkpoint_dir / "pytorch_model.bin.index.json"
    with index_file.open() as handle:
        index_data = json.load(handle)

    state_dict: dict[str, torch.Tensor] = {}
    for shard_name in sorted(set(index_data["weight_map"].values())):
        shard_path = checkpoint_dir / shard_name
        checkpoint = torch.load(shard_path, map_location="cpu")
        state_dict.update(checkpoint)
    return state_dict


def test_local_scarf_module_is_available() -> None:
    import scarf

    assert hasattr(scarf, "TotalModel_downstream")


def test_scarf_rna_checkpoint_smoke() -> None:
    checkpoint_dir = Path(os.environ.get("SCARF_CHECKPOINT_DIR", DEFAULT_CHECKPOINT_DIR))
    missing_files = [name for name in REQUIRED_WEIGHT_FILES if not (checkpoint_dir / name).exists()]
    if missing_files:
        pytest.skip(
            f"SCARF checkpoint not available at {checkpoint_dir}; missing {', '.join(missing_files)}"
        )

    with (checkpoint_dir / "config.json").open() as handle:
        config = BertConfig(**json.load(handle))

    state_dict = _load_state_dict_from_index(checkpoint_dir)
    model = TotalModel_downstream(config, priors={})
    incompat = model.load_state_dict(state_dict, strict=False)
    assert not incompat.missing_keys
    unexpected_keys = [
        key
        for key in incompat.unexpected_keys
        if not key.startswith(ALLOWED_UNEXPECTED_KEY_PREFIXES)
    ]
    assert not unexpected_keys

    if not torch.cuda.is_available():
        pytest.skip(
            "Checkpoint weights load on CPU, but the current upstream SCARF/Mamba2 "
            "forward path still requires CUDA for end-to-end embedding inference."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    batch_size = 2
    seq_len = min(32, config.rna_model_cfg["rna_max_input_size"])

    inputs = {
        "rna_gene_ids": torch.randint(
            low=1,
            high=config.rna_model_cfg["vocab_size"],
            size=(batch_size, seq_len),
            device=device,
        ),
        "rna_gene_values": torch.rand(batch_size, seq_len, device=device),
        "rna_attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long, device=device),
        "species": torch.zeros(batch_size, dtype=torch.long, device=device),
        "modality": torch.zeros(batch_size, dtype=torch.long, device=device),
        "cell_name": [f"cell_{idx}" for idx in range(batch_size)],
    }

    with torch.no_grad():
        result = model.get_rna_embeddings(**inputs)

    embeddings = torch.stack([result[f"cell_{idx}"] for idx in range(batch_size)])
    assert embeddings.shape == (batch_size, config.mlp_out_size)
    assert torch.isfinite(embeddings).all()
    assert torch.allclose(
        embeddings.norm(dim=1),
        torch.ones(batch_size),
        atol=1e-4,
        rtol=1e-4,
    )
