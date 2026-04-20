"""Unit tests for src.trrust.training — uses synthetic data, no GPU required."""

from __future__ import annotations

from pathlib import Path

import anndata
import numpy as np
import pandas as pd
import pytest
import torch

from src.trrust import (
    BINARY_LABELS,
    TERNARY_LABELS,
    load_binary_trrust_data,
    load_gene_embeddings,
    load_ternary_trrust_data,
    prepare_train_test_split,
    train_classifier,
)

EMBSIZE = 8
GENE_NAMES = [
    "ABL1", "BAX", "BCL2", "MYC", "TP53", "JUN", "FOS", "EGR1",
    "STAT1", "STAT3", "NFKB1", "RELA", "CREB1", "ATF2", "GATA1", "GATA2",
]

TSV_CONTENT = """\
ABL1\tBAX\tActivation\t11111111
ABL1\tBCL2\tRepression\t22222222
BCL2\tABL1\tActivation\t33333333
MYC\tTP53\tActivation\t44444444
JUN\tFOS\tRepression\t55555555
STAT1\tSTAT3\tActivation\t66666666
NFKB1\tRELA\tActivation\t77777777
CREB1\tATF2\tRepression\t88888888
GATA1\tGATA2\tActivation\t99999999
EGR1\tMYC\tRepression\t10101010
"""


@pytest.fixture()
def tsv_path(tmp_path: Path) -> Path:
    p = tmp_path / "trrust.tsv"
    p.write_text(TSV_CONTENT)
    return p


@pytest.fixture()
def gene_embeddings() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(42)
    return {
        name: rng.standard_normal(EMBSIZE).astype(np.float32) for name in GENE_NAMES
    }


@pytest.fixture()
def binary_data(tsv_path, gene_embeddings):
    return load_binary_trrust_data(tsv_path, gene_embeddings, seed=42)


@pytest.fixture()
def ternary_data(tsv_path, gene_embeddings):
    return load_ternary_trrust_data(tsv_path, gene_embeddings, seed=42)


class TestLoadGeneEmbeddings:
    def test_roundtrip(self, tmp_path):
        gene_names = ["G1", "G2", "G3"]
        X = np.arange(12, dtype=np.float32).reshape(3, 4)
        adata = anndata.AnnData(X=X, obs=pd.DataFrame(index=gene_names))
        path = tmp_path / "test.h5ad"
        adata.write_h5ad(path)

        result = load_gene_embeddings(path)
        assert set(result.keys()) == set(gene_names)
        assert result["G1"].shape == (4,)
        np.testing.assert_array_equal(result["G2"], X[1])


class TestPrepareTrainTestSplit:
    def test_dataset_sizes(self, binary_data):
        n = len(binary_data.records)
        train_ds, test_ds, test_meta = prepare_train_test_split(
            binary_data, test_size=0.2, seed=42
        )
        assert len(train_ds) + len(test_ds) == n
        assert len(test_ds) == len(test_meta)

    def test_metadata_columns(self, binary_data):
        _, _, test_meta = prepare_train_test_split(binary_data, test_size=0.2)
        assert list(test_meta.columns) == ["tf", "target"]

    def test_tensor_shapes(self, binary_data):
        train_ds, test_ds, _ = prepare_train_test_split(binary_data, test_size=0.2)
        tf, tgt, lbl = train_ds[0]
        assert tf.shape == (EMBSIZE,)
        assert tgt.shape == (EMBSIZE,)
        assert lbl.dtype == torch.int64

    def test_stratification(self, binary_data):
        _, test_ds, _ = prepare_train_test_split(
            binary_data, test_size=0.2, seed=42
        )
        test_labels = test_ds.tensors[2].numpy()
        # Binary data is balanced, so test split should contain both classes
        assert set(test_labels.tolist()) == {0, 1}

    def test_reproducibility(self, binary_data):
        _, _, meta1 = prepare_train_test_split(binary_data, test_size=0.2, seed=7)
        _, _, meta2 = prepare_train_test_split(binary_data, test_size=0.2, seed=7)
        pd.testing.assert_frame_equal(meta1, meta2)


class TestTrainClassifierBinary:
    def test_result_fields(self, binary_data):
        train_ds, test_ds, test_meta = prepare_train_test_split(
            binary_data, test_size=0.2, seed=42
        )
        result = train_classifier(
            train_ds, test_ds, test_meta,
            embsize=EMBSIZE,
            label_map=BINARY_LABELS,
            lr=1e-3,
            epochs=3,
            batch_size=16,
            device="cpu",
        )
        assert len(result.train_losses) == 3
        assert len(result.test_losses) == 3
        assert all(isinstance(x, float) for x in result.train_losses)
        assert "accuracy" in result.classification_report
        assert "None" in result.classification_report
        assert "Relationship" in result.classification_report

    def test_predictions_dataframe(self, binary_data):
        train_ds, test_ds, test_meta = prepare_train_test_split(
            binary_data, test_size=0.2, seed=42
        )
        result = train_classifier(
            train_ds, test_ds, test_meta,
            embsize=EMBSIZE,
            label_map=BINARY_LABELS,
            lr=1e-3,
            epochs=2,
            batch_size=16,
            device="cpu",
        )
        df = result.gene_predictions
        assert list(df.columns) == [
            "tf", "target", "true_relationship", "predicted_relationship"
        ]
        assert len(df) == len(test_ds)
        valid_labels = set(BINARY_LABELS.keys())
        assert set(df["true_relationship"]).issubset(valid_labels)
        assert set(df["predicted_relationship"]).issubset(valid_labels)


class TestTrainClassifierTernary:
    def test_weighted(self, ternary_data):
        train_ds, test_ds, test_meta = prepare_train_test_split(
            ternary_data, test_size=0.2, seed=42
        )
        result = train_classifier(
            train_ds, test_ds, test_meta,
            embsize=EMBSIZE,
            label_map=TERNARY_LABELS,
            lr=1e-3,
            epochs=2,
            batch_size=16,
            use_class_weights=True,
            device="cpu",
        )
        assert len(result.train_losses) == 2
        for name in TERNARY_LABELS:
            assert name in result.classification_report

    def test_unweighted(self, ternary_data):
        train_ds, test_ds, test_meta = prepare_train_test_split(
            ternary_data, test_size=0.2, seed=42
        )
        result = train_classifier(
            train_ds, test_ds, test_meta,
            embsize=EMBSIZE,
            label_map=TERNARY_LABELS,
            lr=1e-3,
            epochs=2,
            batch_size=16,
            use_class_weights=False,
            device="cpu",
        )
        assert len(result.train_losses) == 2


class TestReproducibility:
    def test_same_seed_same_losses(self, binary_data):
        train_ds, test_ds, test_meta = prepare_train_test_split(
            binary_data, test_size=0.2, seed=42
        )
        r1 = train_classifier(
            train_ds, test_ds, test_meta,
            embsize=EMBSIZE, label_map=BINARY_LABELS,
            lr=1e-3, epochs=3, batch_size=16, device="cpu", seed=123,
        )
        r2 = train_classifier(
            train_ds, test_ds, test_meta,
            embsize=EMBSIZE, label_map=BINARY_LABELS,
            lr=1e-3, epochs=3, batch_size=16, device="cpu", seed=123,
        )
        assert r1.train_losses == r2.train_losses
