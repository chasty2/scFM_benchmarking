"""Unit tests for scfm_utils.trrust — uses synthetic data, no GPU required."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from src.trrust import (
    BINARY_LABELS,
    TERNARY_LABELS,
    TRRClassifierModel,
    TRRUSTData,
    TRRUSTRecord,
    cross_validate,
    filter_data_by_genes,
    load_binary_trrust_data,
    load_ternary_trrust_data,
)
from src.trrust.training_data import (
    _collect_raw_pairs,
    _deduplicate,
    _generate_none_pairs,
    _parse_tsv,
    _trrust_genes,
)

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


def _synthetic_trrust_data(
    rng: np.random.Generator,
    n_per_class: int,
    n_classes: int,
) -> TRRUSTData:
    records = []
    for label in range(n_classes):
        for i in range(n_per_class):
            records.append(
                TRRUSTRecord(
                    tf=f"TF{label}_{i}",
                    tf_embedding=rng.standard_normal(EMBSIZE).astype(np.float32),
                    target=f"TGT{label}_{i}",
                    target_embedding=rng.standard_normal(EMBSIZE).astype(np.float32),
                    label=label,
                )
            )
    return TRRUSTData(records=records)


class TestFilterDataByGenes:
    def test_keeps_only_allowed_pairs(self, tsv_path, gene_embeddings):
        data = load_binary_trrust_data(tsv_path, gene_embeddings)
        allowed = {"ABL1", "BAX", "BCL2"}
        filtered = filter_data_by_genes(data, allowed)
        for rec in filtered.records:
            assert rec.tf in allowed
            assert rec.target in allowed

    def test_drops_pairs_with_one_missing_gene(self):
        rng = np.random.default_rng(0)
        emb = lambda: rng.standard_normal(EMBSIZE).astype(np.float32)
        data = TRRUSTData(records=[
            TRRUSTRecord(tf="A", tf_embedding=emb(), target="B", target_embedding=emb(), label=0),
            TRRUSTRecord(tf="A", tf_embedding=emb(), target="C", target_embedding=emb(), label=1),
            TRRUSTRecord(tf="D", tf_embedding=emb(), target="B", target_embedding=emb(), label=0),
        ])
        filtered = filter_data_by_genes(data, {"A", "B"})
        kept = {(r.tf, r.target) for r in filtered.records}
        assert kept == {("A", "B")}

    def test_accepts_list_or_set(self, tsv_path, gene_embeddings):
        data = load_binary_trrust_data(tsv_path, gene_embeddings)
        as_list = filter_data_by_genes(data, ["ABL1", "BAX"])
        as_set = filter_data_by_genes(data, {"ABL1", "BAX"})
        assert len(as_list.records) == len(as_set.records)

    def test_returns_new_object(self, tsv_path, gene_embeddings):
        data = load_binary_trrust_data(tsv_path, gene_embeddings)
        original_len = len(data.records)
        filter_data_by_genes(data, {"ABL1"})
        assert len(data.records) == original_len


class TestCrossValidate:
    def test_produces_n_fold_results(self):
        rng = np.random.default_rng(0)
        data = _synthetic_trrust_data(rng, n_per_class=20, n_classes=2)
        result = cross_validate(
            data,
            embsize=EMBSIZE,
            label_map=BINARY_LABELS,
            lr=1e-3,
            epochs=2,
            batch_size=8,
            n_splits=5,
            device="cpu",
            seed=0,
        )
        assert len(result.per_fold) == 5
        assert len(result.fold_accuracies) == 5

    def test_predictions_cover_every_record_once(self):
        rng = np.random.default_rng(1)
        data = _synthetic_trrust_data(rng, n_per_class=20, n_classes=2)
        result = cross_validate(
            data,
            embsize=EMBSIZE,
            label_map=BINARY_LABELS,
            lr=1e-3,
            epochs=2,
            batch_size=8,
            n_splits=5,
            device="cpu",
            seed=0,
        )
        assert len(result.aggregate_predictions) == len(data.records)
        support = result.aggregate_classification_report["macro avg"]["support"]
        assert support == len(data.records)

    def test_config_recorded(self):
        rng = np.random.default_rng(2)
        data = _synthetic_trrust_data(rng, n_per_class=15, n_classes=3)
        result = cross_validate(
            data,
            embsize=EMBSIZE,
            label_map=TERNARY_LABELS,
            lr=5e-4,
            epochs=3,
            batch_size=8,
            use_class_weights=True,
            n_splits=3,
            device="cpu",
            seed=7,
        )
        assert result.config == {
            "lr": 5e-4,
            "epochs": 3,
            "batch_size": 8,
            "use_class_weights": True,
            "n_splits": 3,
            "seed": 7,
        }
        assert set(result.aggregate_predictions["fold"].unique()) == {0, 1, 2}
