from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REGULATION_LABELS = {"Activation": 0, "Repression": 1, "Unknown": 2}
REGULATION_LABEL_NAMES = {v: k for k, v in REGULATION_LABELS.items()}


@dataclass
class TRRUSTRecord:
    """A single TF → target regulatory relationship with embeddings."""

    tf: str
    tf_embedding: np.ndarray
    target: str
    target_embedding: np.ndarray
    label: int


@dataclass
class TRRUSTData:
    """TRRUST regulatory relationships filtered to genes present in embeddings."""

    records: list[TRRUSTRecord]

    @property
    def tf_embeddings(self) -> np.ndarray:
        return np.stack([r.tf_embedding for r in self.records])

    @property
    def target_embeddings(self) -> np.ndarray:
        return np.stack([r.target_embedding for r in self.records])

    @property
    def labels(self) -> np.ndarray:
        return np.array([r.label for r in self.records], dtype=np.int64)


def load_trrust_data(
    tsv_path: str | Path,
    gene_embeddings: dict[str, np.ndarray],
) -> TRRUSTData:
    """Load TRRUST TSV and create training data from gene embeddings.

    Args:
        tsv_path: Path to the TRRUST TSV file.
        gene_embeddings: Mapping of gene name to embedding vector.

    Returns:
        TRRUSTData containing one TRRUSTRecord per valid (TF, target) pair.
    """
    raw_records = _parse_tsv(Path(tsv_path))
    deduped = _deduplicate(raw_records)

    records = []
    for tf, target, regulation in deduped:
        if tf in gene_embeddings and target in gene_embeddings:
            records.append(
                TRRUSTRecord(
                    tf=tf,
                    tf_embedding=gene_embeddings[tf],
                    target=target,
                    target_embedding=gene_embeddings[target],
                    label=REGULATION_LABELS[regulation],
                )
            )

    return TRRUSTData(records=records)


def _parse_tsv(tsv_path: Path) -> list[tuple[str, str, str]]:
    """Parse the 4-column TRRUST TSV (no header).

    Returns list of (tf, target, regulation) tuples.
    """
    records = []
    with open(tsv_path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 4:
                continue
            tf, target, regulation, _pmids = parts
            records.append((tf, target, regulation))
    return records


def _deduplicate(
    records: list[tuple[str, str, str]],
) -> list[tuple[str, str, str]]:
    """Remove (TF, target) pairs that have conflicting regulation labels.

    Pairs are directional: (A, B) and (B, A) are distinct.
    If the same (TF, target) appears with the same label, keep one copy.
    If it appears with different labels, drop all copies.
    """
    pair_labels: dict[tuple[str, str], set[str]] = defaultdict(set)
    pair_first: dict[tuple[str, str], tuple[str, str, str]] = {}

    for rec in records:
        key = (rec[0], rec[1])
        pair_labels[key].add(rec[2])
        if key not in pair_first:
            pair_first[key] = rec

    return [
        pair_first[key] for key, labels in pair_labels.items() if len(labels) == 1
    ]
