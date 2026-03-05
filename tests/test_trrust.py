"""Unit tests for scfm_utils.trrust — uses synthetic data, no GPU required."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from scfm_utils.trrust import (
    BINARY_LABELS,
    REGULATION_LABELS,
    REGULATION_LABEL_NAMES,
    TRRClassifierModel,
    TRRUSTData,
    load_binary_trrust_data,
    load_ternary_trrust_data,
    load_trrust_data,
)
from scfm_utils.trrust.training_data import _deduplicate, _parse_tsv

EMBSIZE = 4
GENE_NAMES = ["ABL1", "BAX", "BCL2", "MYC", "TP53"]

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
def gene_embeddings() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(42)
    return {name: rng.standard_normal(EMBSIZE) for name in GENE_NAMES}


class TestParseTsv:
    def test_record_count(self, tsv_path):
        records = _parse_tsv(tsv_path)
        assert len(records) == 6

    def test_fields(self, tsv_path):
        records = _parse_tsv(tsv_path)
        tf, target, regulation = records[0]
        assert tf == "ABL1"
        assert target == "BAX"
        assert regulation == "Activation"


class TestDeduplicate:
    def test_drops_conflicting_labels(self, conflict_tsv_path):
        records = _parse_tsv(conflict_tsv_path)
        deduped = _deduplicate(records)
        pairs = [(r[0], r[1]) for r in deduped]
        assert ("ABL1", "BAX") not in pairs

    def test_keeps_consistent_labels(self, conflict_tsv_path):
        records = _parse_tsv(conflict_tsv_path)
        deduped = _deduplicate(records)
        pairs = [(r[0], r[1]) for r in deduped]
        assert ("BCL2", "MYC") in pairs

    def test_directionality(self):
        records = [
            ("A", "B", "Activation"),
            ("B", "A", "Repression"),
        ]
        deduped = _deduplicate(records)
        assert len(deduped) == 2


class TestLoadTrrustData:
    def test_return_type(self, tsv_path, gene_embeddings):
        data = load_trrust_data(tsv_path, gene_embeddings)
        assert isinstance(data, TRRUSTData)

    def test_filtered_records_have_valid_genes(self, tsv_path, gene_embeddings):
        data = load_trrust_data(tsv_path, gene_embeddings)
        gene_set = set(GENE_NAMES)
        for rec in data.records:
            assert rec.tf in gene_set
            assert rec.target in gene_set

    def test_filtered_excludes_missing_genes(self, tsv_path, gene_embeddings):
        data = load_trrust_data(tsv_path, gene_embeddings)
        for rec in data.records:
            assert rec.tf != "NOTINGENES"
            assert rec.target != "NOTINGENES"

    def test_record_count(self, tsv_path, gene_embeddings):
        # TSV has 4 valid pairs after excluding NOTINGENES rows:
        # ABL1->BAX, ABL1->BCL2, BCL2->ABL1, MYC->TP53
        data = load_trrust_data(tsv_path, gene_embeddings)
        assert len(data.records) == 4

    def test_embedding_shapes(self, tsv_path, gene_embeddings):
        data = load_trrust_data(tsv_path, gene_embeddings)
        n = len(data.records)
        assert data.tf_embeddings.shape == (n, EMBSIZE)
        assert data.target_embeddings.shape == (n, EMBSIZE)

    def test_record_embeddings_match_input(self, tsv_path, gene_embeddings):
        data = load_trrust_data(tsv_path, gene_embeddings)
        for rec in data.records:
            np.testing.assert_array_equal(rec.tf_embedding, gene_embeddings[rec.tf])
            np.testing.assert_array_equal(rec.target_embedding, gene_embeddings[rec.target])

    def test_labels_valid(self, tsv_path, gene_embeddings):
        data = load_trrust_data(tsv_path, gene_embeddings)
        assert data.labels.dtype == np.int64
        assert set(data.labels.tolist()).issubset({0, 1, 2})

    def test_label_names_inverse(self, tsv_path, gene_embeddings):
        for k, v in REGULATION_LABELS.items():
            assert REGULATION_LABEL_NAMES[v] == k


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


class TestGenerateNonePairs:
    def test_correct_count(self, tsv_path, gene_embeddings):
        raw = _parse_tsv(tsv_path)
        raw_pairs = _collect_raw_pairs(raw)
        genes = _trrust_genes(raw, gene_embeddings)
        rng = np.random.default_rng(0)
        pairs = _generate_none_pairs(raw_pairs, gene_embeddings, genes, n=3, none_label=0, rng=rng)
        assert len(pairs) == 3

    def test_no_overlap_with_raw_pairs(self, tsv_path, gene_embeddings):
        raw = _parse_tsv(tsv_path)
        raw_pairs = _collect_raw_pairs(raw)
        genes = _trrust_genes(raw, gene_embeddings)
        rng = np.random.default_rng(0)
        pairs = _generate_none_pairs(raw_pairs, gene_embeddings, genes, n=5, none_label=0, rng=rng)
        for rec in pairs:
            assert (rec.tf, rec.target) not in raw_pairs

    def test_genes_from_trrust_set(self, tsv_path, gene_embeddings):
        raw = _parse_tsv(tsv_path)
        raw_pairs = _collect_raw_pairs(raw)
        genes = _trrust_genes(raw, gene_embeddings)
        gene_set = set(genes)
        rng = np.random.default_rng(0)
        pairs = _generate_none_pairs(raw_pairs, gene_embeddings, genes, n=3, none_label=0, rng=rng)
        for rec in pairs:
            assert rec.tf in gene_set
            assert rec.target in gene_set


class TestLoadBinaryTrrustData:
    def test_labels_are_binary(self, tsv_path, gene_embeddings):
        data = load_binary_trrust_data(tsv_path, gene_embeddings)
        assert set(data.labels.tolist()) == {0, 1}

    def test_equal_class_counts(self, tsv_path, gene_embeddings):
        data = load_binary_trrust_data(tsv_path, gene_embeddings)
        labels = data.labels
        assert (labels == 0).sum() == (labels == 1).sum()

    def test_no_none_pairs_in_raw(self, tsv_path, gene_embeddings):
        raw = _parse_tsv(tsv_path)
        raw_pairs = _collect_raw_pairs(raw)
        data = load_binary_trrust_data(tsv_path, gene_embeddings)
        for rec in data.records:
            if rec.label == BINARY_LABELS["None"]:
                assert (rec.tf, rec.target) not in raw_pairs

    def test_reproducible_with_seed(self, tsv_path, gene_embeddings):
        d1 = load_binary_trrust_data(tsv_path, gene_embeddings, seed=123)
        d2 = load_binary_trrust_data(tsv_path, gene_embeddings, seed=123)
        pairs1 = [(r.tf, r.target) for r in d1.records]
        pairs2 = [(r.tf, r.target) for r in d2.records]
        assert pairs1 == pairs2


class TestLoadTernaryTrrustData:
    def test_labels_are_ternary(self, tsv_path, gene_embeddings):
        data = load_ternary_trrust_data(tsv_path, gene_embeddings)
        assert set(data.labels.tolist()).issubset({0, 1, 2})

    def test_no_unknown_records(self, tsv_path, gene_embeddings):
        # The original data has MYC->TP53 as Unknown; verify it's excluded
        data = load_ternary_trrust_data(tsv_path, gene_embeddings)
        real_pairs = {(r.tf, r.target) for r in data.records if r.label != TERNARY_LABELS["None"]}
        assert ("MYC", "TP53") not in real_pairs

    def test_none_count_approx_one_third(self, tsv_path, gene_embeddings):
        data = load_ternary_trrust_data(tsv_path, gene_embeddings)
        labels = data.labels
        n_none = (labels == TERNARY_LABELS["None"]).sum()
        n_total = len(labels)
        # None should be roughly 1/3 of total (may differ by 1 due to integer division)
        assert abs(n_none / n_total - 1 / 3) < 0.1

    def test_unknown_pairs_excluded_from_none(self, tsv_path, gene_embeddings):
        raw = _parse_tsv(tsv_path)
        raw_pairs = _collect_raw_pairs(raw)
        data = load_ternary_trrust_data(tsv_path, gene_embeddings)
        for rec in data.records:
            if rec.label == TERNARY_LABELS["None"]:
                assert (rec.tf, rec.target) not in raw_pairs

    def test_reproducible_with_seed(self, tsv_path, gene_embeddings):
        d1 = load_ternary_trrust_data(tsv_path, gene_embeddings, seed=99)
        d2 = load_ternary_trrust_data(tsv_path, gene_embeddings, seed=99)
        pairs1 = [(r.tf, r.target) for r in d1.records]
        pairs2 = [(r.tf, r.target) for r in d2.records]
        assert pairs1 == pairs2
