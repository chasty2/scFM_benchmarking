"""Integration tests for scfm_utils.scgpt — mirrors the get_embeddings notebook workflow."""

from __future__ import annotations

from pathlib import Path

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
    ScGPTEmbeddings,
    ScGPTModelBundle,
    create_scgpt_dataset,
    encode_scgpt_embeddings,
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
def embeddings(model_bundle, scgpt_dataset, adata) -> ScGPTEmbeddings:
    cell_types = np.array(adata.obs["celltype"].values[:MAX_CELLS])
    return encode_scgpt_embeddings(
        model_bundle.model,
        scgpt_dataset.dataloader,
        model_bundle.vocab,
        scgpt_dataset.genes_in_vocab,
        cell_types,
    )


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
# ScGPTEmbeddings tests (end result of model._encode)
# ---------------------------------------------------------------------------


class TestScGPTEmbeddings:
    def test_types(self, embeddings):
        assert isinstance(embeddings.cls_embeddings, np.ndarray)
        assert isinstance(embeddings.gene_embeddings, np.ndarray)

    def test_cls_embeddings_shape(self, embeddings, model_bundle):
        embsize = model_bundle.config["embsize"]
        assert embeddings.cls_embeddings.shape == (MAX_CELLS, embsize)

    def test_gene_embeddings_shape(self, embeddings, model_bundle, scgpt_dataset):
        embsize = model_bundle.config["embsize"]
        n_genes = len(scgpt_dataset.genes_in_vocab)
        assert embeddings.gene_embeddings.shape == (MAX_CELLS, n_genes, embsize)

    def test_gene_names(self, embeddings, scgpt_dataset):
        assert isinstance(embeddings.gene_names, list)
        assert all(isinstance(g, str) for g in embeddings.gene_names)
        n_genes = len(scgpt_dataset.genes_in_vocab)
        assert len(embeddings.gene_names) == n_genes

    def test_embeddings_finite(self, embeddings):
        assert np.all(np.isfinite(embeddings.cls_embeddings))
        assert np.all(np.isfinite(embeddings.gene_embeddings))

    def test_cell_types(self, embeddings):
        assert isinstance(embeddings.cell_types, np.ndarray)
        assert embeddings.cell_types.shape == (MAX_CELLS,)

    def test_average_gene_embeddings(self, embeddings, model_bundle, scgpt_dataset):
        avg = embeddings.average_gene_embeddings()

        assert isinstance(avg, dict)
        expected_keys = set(np.unique(embeddings.cell_types))
        assert set(avg.keys()) == expected_keys

        n_genes = len(scgpt_dataset.genes_in_vocab)
        embsize = model_bundle.config["embsize"]
        for _, emb in avg.items():
            assert emb.shape == (n_genes, embsize)
            assert np.all(np.isfinite(emb))
