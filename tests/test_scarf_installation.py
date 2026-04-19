from __future__ import annotations

import tempfile
from pathlib import Path

import h5py
import numpy as np


def test_scarf_imports_compiled_dependencies() -> None:
    import hnswlib  # noqa: F401
    import leidenalg
    import numba
    import scarf

    assert scarf.__version__
    assert leidenalg.__version__
    assert numba.__version__
    assert hasattr(scarf, "DataStore")
    assert hasattr(scarf, "H5adReader")


def test_scarf_h5ad_reader_reads_minimal_dataset() -> None:
    import scarf

    with tempfile.TemporaryDirectory() as tmpdir:
        h5ad_path = Path(tmpdir) / "mini.h5ad"
        with h5py.File(h5ad_path, "w") as handle:
            handle.create_dataset("X", data=np.array([[1, 2], [3, 4]], dtype=np.int32))

            obs = handle.create_group("obs")
            obs.create_dataset("_index", data=np.array([b"cell1", b"cell2"]))

            var = handle.create_group("var")
            var.create_dataset("_index", data=np.array([b"gene1", b"gene2"]))
            var.create_dataset("gene_short_name", data=np.array([b"G1", b"G2"]))

            # The reader expects the group to exist even if no embeddings are present.
            handle.create_group("obsm")

        reader = scarf.H5adReader(str(h5ad_path), feature_name_key="gene_short_name")

        assert reader.nCells == 2
        assert reader.nFeatures == 2
        assert reader.cell_ids().tolist() == [b"cell1", b"cell2"]
        assert reader.feat_ids().tolist() == [b"gene1", b"gene2"]
        assert reader.feat_names().tolist() == [b"G1", b"G2"]

        matrix = next(reader.consume(batch_size=10)).toarray()
        np.testing.assert_array_equal(matrix, np.array([[1, 2], [3, 4]], dtype=np.int32))
