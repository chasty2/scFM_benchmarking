"""Unit tests for scfm_utils.trrust — uses synthetic data, no GPU required."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from scfm_utils.scgpt.encode import ScGPTEmbeddings
from scfm_utils.trrust import (
    REGULATION_LABELS,
    TRRClassifierModel,
    TRRUSTData,
    load_trrust_data,
)
from scfm_utils.trrust.training_data import _deduplicate, _parse_tsv, TRRUSTRecord

EMBSIZE = 4
GENE_NAMES = ["ABL1", "BAX", "BCL2", "MYC", "TP53"]
N_GENES = len(GENE_NAMES)
N_CELLS = 6

TSV_CONTENT = """\
ABL1	BAX	Activation	11753601
ABL1	BCL2	Repression	12345678
BCL2	ABL1	Activation	99999999
MYC	TP53	Unknown	22222222
MYC	NOTINGENES	Activation	33333333
NOTINGENES	ABL1	Repression	44444444
"""

TSV_WITH_CONFLICTS = """\
ABL1	BAX	Activation	11111111
ABL1	BAX	Repression	22222222
BCL2	MYC	Activation	33333333
BCL2	MYC	Activation	44444444
"""


@pytest.fixture()
def tsv_path(tmp_path: Path) -> Path:
    p = tmp_path / "trrust.tsv"
    p.write_text(TSV_CONTENT)
    return p


@pytest.fixture()
def conflict_tsv_path(tmp_path: Path) -> Path:
    p = tmp_path / "trrust_conflict.tsv"
    p.write_text(TSV_WITH_CONFLICTS)
    return p


@pytest.fixture()
def embeddings() -> ScGPTEmbeddings:
    rng = np.random.default_rng(42)
    cell_types = np.array(["TypeA"] * 3 + ["TypeB"] * 3)
    return ScGPTEmbeddings(
        cls_embeddings=rng.standard_normal((N_CELLS, EMBSIZE)),
        gene_embeddings=rng.standard_normal((N_CELLS, N_GENES, EMBSIZE)),
        gene_names=GENE_NAMES,
        cell_types=cell_types,
    )


class TestParseTsv:
    def test_record_count(self, tsv_path):
        records = _parse_tsv(tsv_path)
        assert len(records) == 6

    def test_fields(self, tsv_path):
        records = _parse_tsv(tsv_path)
        r = records[0]
        assert r.tf == "ABL1"
        assert r.target == "BAX"
        assert r.regulation == "Activation"


class TestDeduplicate:
    def test_drops_conflicting_labels(self, conflict_tsv_path):
        records = _parse_tsv(conflict_tsv_path)
        deduped = _deduplicate(records)
        pairs = [(r.tf, r.target) for r in deduped]
        assert ("ABL1", "BAX") not in pairs

    def test_keeps_consistent_labels(self, conflict_tsv_path):
        records = _parse_tsv(conflict_tsv_path)
        deduped = _deduplicate(records)
        pairs = [(r.tf, r.target) for r in deduped]
        assert ("BCL2", "MYC") in pairs

    def test_directionality(self):
        records = [
            TRRUSTRecord(tf="A", target="B", regulation="Activation"),
            TRRUSTRecord(tf="B", target="A", regulation="Repression"),
        ]
        deduped = _deduplicate(records)
        assert len(deduped) == 2


class TestLoadTrrustData:
    def test_return_type(self, tsv_path, embeddings):
        data = load_trrust_data(tsv_path, embeddings)
        assert isinstance(data, TRRUSTData)

    def test_filtered_records_have_valid_genes(self, tsv_path, embeddings):
        data = load_trrust_data(tsv_path, embeddings)
        gene_set = set(GENE_NAMES)
        for rec in data.filtered_records:
            assert rec.tf in gene_set
            assert rec.target in gene_set

    def test_filtered_excludes_missing_genes(self, tsv_path, embeddings):
        data = load_trrust_data(tsv_path, embeddings)
        for rec in data.filtered_records:
            assert rec.tf != "NOTINGENES"
            assert rec.target != "NOTINGENES"

    def test_embedding_shapes(self, tsv_path, embeddings):
        data = load_trrust_data(tsv_path, embeddings)
        n_cell_types = len(np.unique(embeddings.cell_types))
        n_filtered = len(data.filtered_records)
        expected_rows = n_cell_types * n_filtered
        assert data.tf_embeddings.shape == (expected_rows, EMBSIZE)
        assert data.target_embeddings.shape == (expected_rows, EMBSIZE)

    def test_labels_valid(self, tsv_path, embeddings):
        data = load_trrust_data(tsv_path, embeddings)
        assert data.labels.dtype == np.int64
        assert set(data.labels.tolist()).issubset({0, 1, 2})

    def test_label_names_inverse(self, tsv_path, embeddings):
        data = load_trrust_data(tsv_path, embeddings)
        for k, v in REGULATION_LABELS.items():
            assert data.label_names[v] == k

    def test_cell_types_length(self, tsv_path, embeddings):
        data = load_trrust_data(tsv_path, embeddings)
        assert len(data.cell_types) == len(data.labels)

    def test_all_cell_types_present(self, tsv_path, embeddings):
        data = load_trrust_data(tsv_path, embeddings)
        assert set(data.cell_types) == {"TypeA", "TypeB"}

    def test_average_gene_embeddings_stored(self, tsv_path, embeddings):
        data = load_trrust_data(tsv_path, embeddings)
        assert isinstance(data.average_gene_embeddings, dict)
        assert set(data.average_gene_embeddings.keys()) == {"TypeA", "TypeB"}
        for emb in data.average_gene_embeddings.values():
            assert emb.shape == (N_GENES, EMBSIZE)


class TestTRRClassifierModel:
    def test_output_shape(self):
        model = TRRClassifierModel(embsize=EMBSIZE, n_classes=3)
        tf_emb = torch.randn(8, EMBSIZE)
        target_emb = torch.randn(8, EMBSIZE)
        logits = model(tf_emb, target_emb)
        assert logits.shape == (8, 3)

    def test_gradient_flow(self):
        model = TRRClassifierModel(embsize=EMBSIZE, n_classes=3)
        tf_emb = torch.randn(4, EMBSIZE)
        target_emb = torch.randn(4, EMBSIZE)
        logits = model(tf_emb, target_emb)
        loss = logits.sum()
        loss.backward()
        assert model.bilinear.weight.grad is not None
        assert model.bilinear.weight.grad.shape == (3, EMBSIZE, EMBSIZE)
