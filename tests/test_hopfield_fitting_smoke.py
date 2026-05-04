from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.hopfield_fitting_common import candidate_isoform_groups, load_vocab, normalize_rows, spearman
from scripts.make_mutant_esmc_embeddings import parse_mutation


def test_load_vocab_list_and_dict(tmp_path: Path) -> None:
    list_path = tmp_path / "vocab_list.json"
    list_path.write_text(json.dumps(["b", "a"]), encoding="utf-8")
    assert load_vocab(list_path) == ["b", "a"]

    dict_path = tmp_path / "vocab_dict.json"
    dict_path.write_text(json.dumps({"b": 1, "a": 0}), encoding="utf-8")
    assert load_vocab(dict_path) == ["a", "b"]


def test_normalize_rows_handles_zero_rows() -> None:
    matrix = np.asarray([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32)
    out = normalize_rows(matrix)
    assert np.allclose(out[0], [0.6, 0.8])
    assert np.allclose(out[1], [0.0, 0.0])


def test_spearman_reports_expected_sign() -> None:
    result = spearman(np.asarray([1, 2, 3, 4]), np.asarray([4, 3, 2, 1]))
    assert result["rho"] == pytest.approx(-1.0)
    assert result["n"] == 4


def test_candidate_isoform_groups_mixed_labels() -> None:
    df = pd.DataFrame(
        {
            "gene_symbol": ["A", "A", "B"],
            "isoform_id": ["A-1", "A-2", "B-1"],
            "n_cells": [10, 20, 1],
            "label_status": ["responder", "nonresponder", "responder"],
            "response_score": [10.0, 1.0, 2.0],
        }
    )
    groups = candidate_isoform_groups(df, min_cells=5)
    assert groups.iloc[0]["gene_symbol"] == "A"
    assert groups.iloc[0]["n_responders"] == 1
    assert groups.iloc[0]["n_nonresponders"] == 1


def test_parse_mutation() -> None:
    assert parse_mutation("R248Q") == ("R", 248, "Q")
    with pytest.raises(ValueError):
        parse_mutation("bad")
