"""Integration tests for scfm_utils.scgpt — mirrors the get_embeddings notebook workflow."""

from __future__ import annotations

from pathlib import Path

import anndata
import numpy as np
import pytest
import scanpy as sc
import torch
from scgpt.model import TransformerModel
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from torch.utils.data import DataLoader

from src.constants import SPECIAL_TOKENS
from src.scgpt import (
    ScGPTDataset,
    ScGPTModelBundle,
    create_scgpt_dataset,
    encode_scgpt_embeddings_to_h5ad,
    load_average_gene_embeddings,
    load_scgpt_model,
)

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "scGPT_bc"
DATA_PATH = ROOT / "data" / "Immune_ALL_human.h5ad"
GENE_LIST_FILE = ROOT / "data" / "trrust_genes.txt"


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
        gene_list_file=GENE_LIST_FILE,
    )


@pytest.fixture(scope="module")
def h5ad_path(model_bundle, adata, tmp_path_factory) -> Path:
    """Encode cells of a single cell type and write to a temporary h5ad file."""
    path = tmp_path_factory.mktemp("embeddings") / "test.h5ad"
    cell_type = str(adata.obs["celltype"].iloc[0])
    adata_ct = adata[adata.obs["celltype"] == cell_type].copy()
    ds = create_scgpt_dataset(
        adata_ct,
        model_bundle.vocab,
        model_bundle.gene2idx,
        gene_list_file=GENE_LIST_FILE,
    )
    encode_scgpt_embeddings_to_h5ad(
        model=model_bundle.model,
        dataloader=ds.dataloader,
        vocab=model_bundle.vocab,
        gene_names=ds.genes_in_vocab,
        cell_type=cell_type,
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
        assert len(scgpt_dataset.genes_in_vocab) > 0

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
    def test_returns_anndata(self, h5ad_path):
        assert isinstance(load_average_gene_embeddings(h5ad_path), anndata.AnnData)

    def test_shape(self, h5ad_path, model_bundle):
        result = load_average_gene_embeddings(h5ad_path)
        embsize = model_bundle.config["embsize"]
        assert result.X.shape == (result.n_obs, embsize)

    def test_obs_names_are_gene_strings(self, h5ad_path):
        result = load_average_gene_embeddings(h5ad_path)
        assert len(result.obs_names) > 0
        assert all(isinstance(g, str) for g in result.obs_names)

    def test_cell_type_in_uns(self, h5ad_path, adata):
        result = load_average_gene_embeddings(h5ad_path)
        assert "cell_type" in result.uns
        assert result.uns["cell_type"] == str(adata.obs["celltype"].iloc[0])

    def test_embeddings_are_finite(self, h5ad_path):
        assert np.all(np.isfinite(load_average_gene_embeddings(h5ad_path).X))
