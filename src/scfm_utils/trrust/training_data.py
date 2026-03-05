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


def _collect_raw_pairs(raw_records: list[tuple[str, str, str]]) -> set[tuple[str, str]]:
    """Return all (TF, target) pairs from raw parsed records."""
    return {(r[0], r[1]) for r in raw_records}


def _trrust_genes(
    raw_records: list[tuple[str, str, str]],
    gene_embeddings: dict[str, np.ndarray],
) -> list[str]:
    """Return sorted list of genes appearing in TRRUST that also have embeddings."""
    genes = set()
    for tf, target, _ in raw_records:
        genes.add(tf)
        genes.add(target)
    return sorted(genes & gene_embeddings.keys())


def _generate_none_pairs(
    all_raw_pairs: set[tuple[str, str]],
    gene_embeddings: dict[str, np.ndarray],
    trrust_genes: list[str],
    n: int,
    none_label: int,
    rng: np.random.Generator,
) -> list[TRRUSTRecord]:
    """Sample random gene pairs with no known TRRUST relationship."""
    candidates = [
        (a, b)
        for a in trrust_genes
        for b in trrust_genes
        if a != b and (a, b) not in all_raw_pairs
    ]
    if len(candidates) < n:
        raise ValueError(
            f"Not enough candidate None pairs: need {n}, only {len(candidates)} available"
        )
    indices = rng.choice(len(candidates), size=n, replace=False)
    records = []
    for idx in indices:
        a, b = candidates[idx]
        records.append(
            TRRUSTRecord(
                tf=a,
                tf_embedding=gene_embeddings[a],
                target=b,
                target_embedding=gene_embeddings[b],
                label=none_label,
            )
        )
    return records


def load_binary_trrust_data(
    tsv_path: str | Path,
    gene_embeddings: dict[str, np.ndarray],
    seed: int = 42,
) -> TRRUSTData:
    """Load TRRUST data for binary classification (Relationship vs None).

    All regulation types (Activation, Repression, Unknown) become label 1.
    An equal number of None pairs (label 0) are sampled from gene pairs
    with no known TRRUST relationship.
    """
    raw_records = _parse_tsv(Path(tsv_path))
    all_raw_pairs = _collect_raw_pairs(raw_records)
    deduped = _deduplicate(raw_records)
    genes = _trrust_genes(raw_records, gene_embeddings)

    relationship_records = []
    for tf, target, _regulation in deduped:
        if tf in gene_embeddings and target in gene_embeddings:
            relationship_records.append(
                TRRUSTRecord(
                    tf=tf,
                    tf_embedding=gene_embeddings[tf],
                    target=target,
                    target_embedding=gene_embeddings[target],
                    label=BINARY_LABELS["Relationship"],
                )
            )

    rng = np.random.default_rng(seed)
    none_records = _generate_none_pairs(
        all_raw_pairs, gene_embeddings, genes,
        n=len(relationship_records),
        none_label=BINARY_LABELS["None"],
        rng=rng,
    )
    return TRRUSTData(records=relationship_records + none_records)


def load_ternary_trrust_data(
    tsv_path: str | Path,
    gene_embeddings: dict[str, np.ndarray],
    seed: int = 42,
) -> TRRUSTData:
    """Load TRRUST data for ternary classification (Activation, Repression, None).

    Unknown entries are removed. None pairs (label 2) are sampled to be ~1/3
    of the total dataset. Unknown pairs are excluded from None sampling.
    """
    raw_records = _parse_tsv(Path(tsv_path))
    all_raw_pairs = _collect_raw_pairs(raw_records)
    deduped = _deduplicate(raw_records)
    genes = _trrust_genes(raw_records, gene_embeddings)

    real_records = []
    for tf, target, regulation in deduped:
        if regulation == "Unknown":
            continue
        if tf in gene_embeddings and target in gene_embeddings:
            real_records.append(
                TRRUSTRecord(
                    tf=tf,
                    tf_embedding=gene_embeddings[tf],
                    target=target,
                    target_embedding=gene_embeddings[target],
                    label=TERNARY_LABELS[regulation],
                )
            )

    rng = np.random.default_rng(seed)
    n_none = len(real_records) // 2
    none_records = _generate_none_pairs(
        all_raw_pairs, gene_embeddings, genes,
        n=n_none,
        none_label=TERNARY_LABELS["None"],
        rng=rng,
    )
    return TRRUSTData(records=real_records + none_records)
