"""Tests for src.geneformer.

Synthetic-data tests cover preprocessing + combine + h5ad write. The full
``EmbExtractor`` is not exercised — the encoding loop is tested implicitly via
``combine_chunk_embeddings`` over fake per-chunk CSVs.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from anndata import AnnData

from src.geneformer import (
    chunk_csv_path,
    combine_chunk_embeddings,
    iter_chunks,
    load_average_gene_embeddings,
    prepare_adata_for_geneformer,
    save_geneformer_h5ad,
)


GENE_SYMBOLS = ["GENEA", "GENEB", "GENEC", "GENED"]
ENSEMBL_MAP = {
    "GENEA": "ENSG00000000001",
    "GENEB": "ENSG00000000002",
    "GENEC": "ENSG00000000003",
}  # GENED intentionally absent → "unknown"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ensembl_dict_path(tmp_path: Path) -> Path:
    p = tmp_path / "ensembl_dict.pkl"
    with open(p, "wb") as f:
        pickle.dump(ENSEMBL_MAP, f)
    return p


@pytest.fixture
def synthetic_adata() -> AnnData:
    rng = np.random.default_rng(0)
    X = rng.poisson(2.0, size=(8, len(GENE_SYMBOLS))).astype(np.float32)
    adata = AnnData(X=X)
    adata.var_names = GENE_SYMBOLS
    return adata


@pytest.fixture
def synthetic_adata_sparse() -> AnnData:
    rng = np.random.default_rng(0)
    X = sp.csr_matrix(rng.poisson(2.0, size=(6, len(GENE_SYMBOLS))).astype(np.float32))
    adata = AnnData(X=X)
    adata.var_names = GENE_SYMBOLS
    adata.obs["n_counts"] = np.array([10, 11, 12, 13, 14, 15], dtype=np.float32)
    return adata


# ---------------------------------------------------------------------------
# prepare_adata_for_geneformer
# ---------------------------------------------------------------------------


class TestPrepareAdata:
    def test_adds_ensembl_id_column(self, synthetic_adata, ensembl_dict_path):
        out = prepare_adata_for_geneformer(synthetic_adata, ensembl_dict_path)
        assert "ensembl_id" in out.var.columns
        assert list(out.var["ensembl_id"]) == [
            "ENSG00000000001",
            "ENSG00000000002",
            "ENSG00000000003",
            "unknown",
        ]

    def test_computes_n_counts_when_missing(self, synthetic_adata, ensembl_dict_path):
        assert "n_counts" not in synthetic_adata.obs.columns
        out = prepare_adata_for_geneformer(synthetic_adata, ensembl_dict_path)
        assert "n_counts" in out.obs.columns
        np.testing.assert_array_equal(
            out.obs["n_counts"].to_numpy(),
            np.asarray(synthetic_adata.X.sum(axis=1)).flatten(),
        )

    def test_preserves_existing_n_counts(self, synthetic_adata_sparse, ensembl_dict_path):
        original = synthetic_adata_sparse.obs["n_counts"].copy()
        out = prepare_adata_for_geneformer(synthetic_adata_sparse, ensembl_dict_path)
        np.testing.assert_array_equal(out.obs["n_counts"].to_numpy(), original.to_numpy())


# ---------------------------------------------------------------------------
# iter_chunks
# ---------------------------------------------------------------------------


class TestIterChunks:
    def test_chunks_cover_all_rows_exactly(self):
        from datasets import Dataset

        ds = Dataset.from_dict({"x": list(range(10))})
        chunks = list(iter_chunks(ds, chunk_size=3))

        assert [i for i, _ in chunks] == [0, 1, 2, 3]
        assert [len(c) for _, c in chunks] == [3, 3, 3, 1]
        assert sum((c["x"] for _, c in chunks), []) == list(range(10))

    def test_exact_division(self):
        from datasets import Dataset

        ds = Dataset.from_dict({"x": list(range(6))})
        chunks = list(iter_chunks(ds, chunk_size=2))
        assert [len(c) for _, c in chunks] == [2, 2, 2]


# ---------------------------------------------------------------------------
# combine_chunk_embeddings
# ---------------------------------------------------------------------------


def _write_chunk_csv(path: Path, df: pd.DataFrame) -> Path:
    df.to_csv(path)
    return path


class TestCombineChunks:
    def test_empty_input_raises(self):
        with pytest.raises(ValueError):
            combine_chunk_embeddings([])

    def test_weighted_average_overlapping_genes(self, tmp_path: Path):
        # Two chunks, two genes. With weights proportional to row counts (both = 1
        # row so weights = 0.5/0.5), the average is the elementwise mean.
        df1 = pd.DataFrame([[2.0, 4.0]], index=["G1"], columns=["e0", "e1"])
        df2 = pd.DataFrame([[4.0, 8.0]], index=["G1"], columns=["e0", "e1"])
        p1 = _write_chunk_csv(tmp_path / "c1.csv", df1)
        p2 = _write_chunk_csv(tmp_path / "c2.csv", df2)

        out = combine_chunk_embeddings([p1, p2])
        np.testing.assert_allclose(out.loc["G1"].to_numpy(), [3.0, 6.0])

    def test_union_with_missing_genes_filled_zero(self, tmp_path: Path):
        # Chunk 1 has G1; chunk 2 has G2. Equal weights → each gene halved.
        df1 = pd.DataFrame([[10.0, 20.0]], index=["G1"], columns=["e0", "e1"])
        df2 = pd.DataFrame([[30.0, 40.0]], index=["G2"], columns=["e0", "e1"])
        p1 = _write_chunk_csv(tmp_path / "c1.csv", df1)
        p2 = _write_chunk_csv(tmp_path / "c2.csv", df2)

        out = combine_chunk_embeddings([p1, p2])

        assert sorted(out.index) == ["G1", "G2"]
        np.testing.assert_allclose(out.loc["G1"].to_numpy(), [5.0, 10.0])
        np.testing.assert_allclose(out.loc["G2"].to_numpy(), [15.0, 20.0])

    def test_weights_proportional_to_chunk_size_by_default(self, tmp_path: Path):
        # Chunk 1 has 3 rows, chunk 2 has 1 row → weights 0.75 / 0.25.
        df1 = pd.DataFrame(
            [[1.0], [1.0], [1.0]], index=["G1", "G2", "G3"], columns=["e0"]
        )
        df2 = pd.DataFrame([[5.0]], index=["G1"], columns=["e0"])
        p1 = _write_chunk_csv(tmp_path / "c1.csv", df1)
        p2 = _write_chunk_csv(tmp_path / "c2.csv", df2)

        out = combine_chunk_embeddings([p1, p2])
        # G1: 0.75 * 1 + 0.25 * 5 = 2.0
        np.testing.assert_allclose(out.loc["G1", "e0"], 2.0)
        # G2: 0.75 * 1 + 0.25 * 0 = 0.75
        np.testing.assert_allclose(out.loc["G2", "e0"], 0.75)

    def test_explicit_weights_override(self, tmp_path: Path):
        df1 = pd.DataFrame([[2.0]], index=["G1"], columns=["e0"])
        df2 = pd.DataFrame([[6.0]], index=["G1"], columns=["e0"])
        p1 = _write_chunk_csv(tmp_path / "c1.csv", df1)
        p2 = _write_chunk_csv(tmp_path / "c2.csv", df2)

        # weights 3:1 → (3*2 + 1*6) / 4 = 3.0
        out = combine_chunk_embeddings([p1, p2], weights=[3.0, 1.0])
        np.testing.assert_allclose(out.loc["G1", "e0"], 3.0)


# ---------------------------------------------------------------------------
# save_geneformer_h5ad → load_average_gene_embeddings round-trip
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_embs_df() -> pd.DataFrame:
    # 3 real genes + 2 special tokens that should be filtered out.
    rng = np.random.default_rng(1)
    data = rng.standard_normal((5, 6)).astype(np.float32)
    return pd.DataFrame(
        data,
        index=[
            "<cls>",
            "<eos>",
            "ENSG00000000001",
            "ENSG00000000002",
            "ENSG00000000003",
        ],
        columns=[f"e{i}" for i in range(6)],
    )


class TestSaveH5ad:
    @pytest.fixture
    def h5ad_path(self, tmp_path: Path, synthetic_embs_df) -> Path:
        path = tmp_path / "geneformer_test.h5ad"
        ensembl_to_symbol = {v: k for k, v in ENSEMBL_MAP.items()}
        save_geneformer_h5ad(
            synthetic_embs_df,
            ensembl_to_symbol=ensembl_to_symbol,
            cell_type="CD20+ B cells",
            output_path=path,
        )
        return path

    def test_returns_anndata(self, h5ad_path):
        assert isinstance(load_average_gene_embeddings(h5ad_path), anndata.AnnData)

    def test_shape_excludes_special_tokens(self, h5ad_path, synthetic_embs_df):
        result = load_average_gene_embeddings(h5ad_path)
        n_real = synthetic_embs_df.shape[0] - 2  # cls, eos
        embsize = synthetic_embs_df.shape[1]
        assert result.X.shape == (n_real, embsize)

    def test_obs_index_is_gene_symbols(self, h5ad_path):
        result = load_average_gene_embeddings(h5ad_path)
        assert list(result.obs_names) == ["GENEA", "GENEB", "GENEC"]

    def test_ensembl_id_column(self, h5ad_path):
        result = load_average_gene_embeddings(h5ad_path)
        assert "ensembl_id" in result.obs.columns
        assert list(result.obs["ensembl_id"]) == [
            "ENSG00000000001",
            "ENSG00000000002",
            "ENSG00000000003",
        ]

    def test_no_special_tokens_in_index(self, h5ad_path):
        result = load_average_gene_embeddings(h5ad_path)
        for tok in ("<cls>", "<eos>", "<pad>", "<mask>"):
            assert tok not in result.obs_names

    def test_cell_type_in_uns(self, h5ad_path):
        result = load_average_gene_embeddings(h5ad_path)
        assert result.uns["cell_type"] == "CD20+ B cells"

    def test_dtype_float32(self, h5ad_path):
        result = load_average_gene_embeddings(h5ad_path)
        assert result.X.dtype == np.float32

    def test_embeddings_are_finite(self, h5ad_path):
        result = load_average_gene_embeddings(h5ad_path)
        assert np.all(np.isfinite(result.X))

    def test_unmapped_ensembl_falls_back_to_id(self, tmp_path: Path):
        df = pd.DataFrame(
            [[1.0, 2.0]],
            index=["ENSG_UNKNOWN"],
            columns=["e0", "e1"],
        )
        path = tmp_path / "fallback.h5ad"
        save_geneformer_h5ad(
            df,
            ensembl_to_symbol={},
            cell_type="X",
            output_path=path,
        )
        result = load_average_gene_embeddings(path)
        assert list(result.obs_names) == ["ENSG_UNKNOWN"]
        assert list(result.obs["ensembl_id"]) == ["ENSG_UNKNOWN"]


# ---------------------------------------------------------------------------
# chunk_csv_path
# ---------------------------------------------------------------------------


class TestChunkCsvPath:
    def test_format(self, tmp_path: Path):
        assert chunk_csv_path(tmp_path, 7) == tmp_path / "chunk_7.csv"
