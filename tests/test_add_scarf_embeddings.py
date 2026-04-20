from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

from scripts.add_scarf_embeddings import derive_nonzero_gene_medians


def test_derive_nonzero_gene_medians_dense() -> None:
    counts = np.array(
        [
            [0, 1, 0],
            [2, 0, 0],
            [4, 3, 5],
        ],
        dtype=np.float32,
    )
    gene_ids = pd.Series(["ENSG1", "ENSG2", "ENSG3"])
    token_dictionary = {"ENSG1": 0, "ENSG2": 1}

    medians = derive_nonzero_gene_medians(counts, gene_ids, token_dictionary)

    assert medians == {"ENSG1": 3.0, "ENSG2": 2.0}


def test_derive_nonzero_gene_medians_sparse() -> None:
    counts = sparse.csr_matrix(
        np.array(
            [
                [0, 1, 0],
                [2, 0, 0],
                [4, 3, 5],
            ],
            dtype=np.float32,
        )
    )
    gene_ids = pd.Series(["ENSG1", "ENSG2", "ENSG3"])
    token_dictionary = {"ENSG1": 0, "ENSG2": 1}

    medians = derive_nonzero_gene_medians(counts, gene_ids, token_dictionary)

    assert medians == {"ENSG1": 3.0, "ENSG2": 2.0}
