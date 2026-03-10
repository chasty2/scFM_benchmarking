"""Integration tests for scfm_utils.scgpt — mirrors the get_embeddings notebook workflow."""

from __future__ import annotations

from pathlib import Path

import anndata
import h5py
import numpy as np
import pytest
import scanpy as sc
import torch
from scgpt.model import TransformerModel
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from torch.utils.data import DataLoader

from scfm_utils.constants import N_HVG, SPECIAL_TOKENS
from scfm_utils.scgpt import (
    ScGPTDataset,
    ScGPTModelBundle,
    create_scgpt_dataset,
    encode_scgpt_embeddings_to_h5ad,
    load_average_gene_embeddings,
    load_cls_embeddings,
    load_gene_embeddings,
    load_gene_names,
    load_scgpt_model,
)

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "scGPT_bc"
DATA_PATH = ROOT / "data" / "Immune_ALL_human.h5ad"
MAX_CELLS = 256


# ---------------------------------------------------------------------------
# Fixtures (module-scoped so the model + data are loaded once)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def model_bundle() -> ScGPTModelBundle:
    return load_scgpt_model(MODEL_DIR)


@pytest.fixture(scope="module")
def adata() -> sc.AnnData:
    adata = sc.read(str(DATA_PATH), cache=True)
    adata.obs["celltype"] = adata.obs["final_annotation"].astype(str)
    return adata


@pytest.fixture(scope="module")
def scgpt_dataset(adata, model_bundle) -> ScGPTDataset:
    return create_scgpt_dataset(
        adata,
        model_bundle.vocab,
        model_bundle.gene2idx,
        max_cells=MAX_CELLS,
    )


@pytest.fixture(scope="module")
def h5ad_path(model_bundle, scgpt_dataset, adata, tmp_path_factory) -> Path:
    """Encode 256 cells and write to a temporary h5ad file."""
    path = tmp_path_factory.mktemp("embeddings") / "test.h5ad"
    cell_types = np.array(adata.obs["celltype"].values[:MAX_CELLS])
    encode_scgpt_embeddings_to_h5ad(
        model=model_bundle.model,
        dataloader=scgpt_dataset.dataloader,
        vocab=model_bundle.vocab,
        gene_names=scgpt_dataset.genes_in_vocab,
        cell_types=cell_types,
        output_path=path,
    )
    return path


# ---------------------------------------------------------------------------
# ScGPTModelBundle tests
# ---------------------------------------------------------------------------


class TestScGPTModelBundle:
    def test_types(self, model_bundle):
        assert isinstance(model_bundle.model, TransformerModel)
        assert isinstance(model_bundle.vocab, GeneVocab)
        assert isinstance(model_bundle.gene2idx, dict)
        assert isinstance(model_bundle.config, dict)

    def test_config_keys(self, model_bundle):
        for key in ("embsize", "nheads", "d_hid", "nlayers"):
            assert key in model_bundle.config, f"Missing config key: {key}"

    def test_vocab_has_special_tokens(self, model_bundle):
        for token in SPECIAL_TOKENS:
            assert token in model_bundle.vocab, f"Missing special token: {token}"


# ---------------------------------------------------------------------------
# ScGPTDataset tests
# ---------------------------------------------------------------------------


class TestScGPTDataset:
    def test_types(self, scgpt_dataset):
        assert isinstance(scgpt_dataset.dataloader, DataLoader)
        assert isinstance(scgpt_dataset.genes_in_vocab, list)
        assert all(isinstance(g, str) for g in scgpt_dataset.genes_in_vocab)

    def test_genes_in_vocab_count(self, scgpt_dataset):
        n = len(scgpt_dataset.genes_in_vocab)
        assert 0 < n <= N_HVG

    def test_batch_shapes(self, scgpt_dataset):
        batch = next(iter(scgpt_dataset.dataloader))
        n_genes = len(scgpt_dataset.genes_in_vocab)
        expected_seq_len = n_genes + 1  # +1 for <cls> token

        assert "genes" in batch and "values" in batch
        assert batch["genes"].shape[1] == expected_seq_len
        assert batch["values"].shape[1] == expected_seq_len
        assert batch["genes"].dtype == torch.long
        assert batch["values"].dtype == torch.float32


# ---------------------------------------------------------------------------
# Encode-to-h5ad round-trip tests
# ---------------------------------------------------------------------------


class TestScGPTIO:
    def test_encoding_complete(self, h5ad_path):
        with h5py.File(h5ad_path, "r") as h5f:
            assert h5f.attrs["encoding_complete"] == True

    def test_load_cls_embeddings(self, h5ad_path, model_bundle):
        adata = load_cls_embeddings(h5ad_path)
        embsize = model_bundle.config["embsize"]

        assert isinstance(adata, anndata.AnnData)
        assert adata.X.shape == (MAX_CELLS, embsize)
        assert "celltype" in adata.obs.columns
        assert len(adata.obs) == MAX_CELLS

    def test_load_gene_embeddings_full(self, h5ad_path, model_bundle, scgpt_dataset):
        gene_embs = load_gene_embeddings(h5ad_path)
        embsize = model_bundle.config["embsize"]
        n_genes = len(scgpt_dataset.genes_in_vocab)

        assert gene_embs.shape == (MAX_CELLS, n_genes, embsize)
        assert np.all(np.isfinite(gene_embs))

    def test_load_gene_embeddings_sliced(self, h5ad_path, scgpt_dataset):
        sliced = load_gene_embeddings(h5ad_path, cell_indices=slice(0, 5))
        n_genes = len(scgpt_dataset.genes_in_vocab)

        assert sliced.shape[0] == 5
        assert sliced.shape[1] == n_genes

    def test_load_gene_names(self, h5ad_path, scgpt_dataset):
        names = load_gene_names(h5ad_path)
        assert names == scgpt_dataset.genes_in_vocab
        assert all(isinstance(g, str) for g in names)

    def test_load_average_gene_embeddings(self, h5ad_path, model_bundle, scgpt_dataset, adata):
        avgs, gene_names = load_average_gene_embeddings(h5ad_path)
        embsize = model_bundle.config["embsize"]
        n_genes = len(scgpt_dataset.genes_in_vocab)
        cell_types = np.array(adata.obs["celltype"].values[:MAX_CELLS])
        expected_types = set(np.unique(cell_types))

        assert isinstance(avgs, dict)
        assert set(avgs.keys()) == expected_types
        assert gene_names == scgpt_dataset.genes_in_vocab

        for ct, emb in avgs.items():
            assert emb.shape == (n_genes, embsize)
            assert np.all(np.isfinite(emb))

    def test_average_gene_embeddings_uses_fast_path(self, h5ad_path):
        """Verify pre-computed averages exist in the file."""
        with h5py.File(h5ad_path, "r") as h5f:
            assert "gene_avg" in h5f["embeddings"]
            assert "gene_avg_cell_types" in h5f["embeddings"]
