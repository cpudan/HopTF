from __future__ import annotations

import pandas as pd

from scripts.build_scp3357_h5ad import (
    derive_perturbation_class,
    derive_perturbation_gene,
    table_to_uns_dict,
    translate_orf_to_protein,
)


def test_derive_perturbation_gene_and_class() -> None:
    assert derive_perturbation_gene("HLF.1") == "HLF"
    assert derive_perturbation_gene("BFP.Control") == "BFP"
    assert derive_perturbation_gene("None.Lipid") == "None"

    assert derive_perturbation_class("HLF.1") == "tf"
    assert derive_perturbation_class("BFP.Control") == "control_bfp"
    assert derive_perturbation_class("None.Lipid") == "control_none"


def test_translate_orf_to_protein_stops_at_stop_codon() -> None:
    assert translate_orf_to_protein("ATGGCCGACTAA") == "MAD"


def test_table_to_uns_dict_preserves_scalar_types() -> None:
    table = pd.DataFrame(
        {
            "name": ["TFORF0001", "TFORF0002"],
            "length": [123, 456],
            "is_exact": [True, False],
        }
    )

    uns = table_to_uns_dict(table)

    assert uns["name"] == ["TFORF0001", "TFORF0002"]
    assert uns["length"] == [123, 456]
    assert uns["is_exact"] == [True, False]
