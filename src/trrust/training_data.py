from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

BINARY_LABELS = {"None": 0, "Relationship": 1}
BINARY_LABEL_NAMES = {v: k for k, v in BINARY_LABELS.items()}

TERNARY_LABELS = {"Activation": 0, "Repression": 1, "None": 2}
TERNARY_LABEL_NAMES = {v: k for k, v in TERNARY_LABELS.items()}


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


def _build_none_records_from_pairs(
    none_pairs: list[tuple[str, str]],
    gene_embeddings: dict[str, np.ndarray],
    none_label: int,
) -> list[TRRUSTRecord]:
    """Build TRRUSTRecords from a precomputed list of (tf, target) pairs."""
    records = []
    for tf, target in none_pairs:
        if tf not in gene_embeddings or target not in gene_embeddings:
            raise ValueError(
                f"None pair ({tf}, {target}) has a gene missing from gene_embeddings"
            )
        records.append(
            TRRUSTRecord(
                tf=tf,
                tf_embedding=gene_embeddings[tf],
                target=target,
                target_embedding=gene_embeddings[target],
                label=none_label,
            )
        )
    return records


def generate_shared_none_pairs(
    tsv_path: str | Path,
    vocabularies: list[set[str]],
    n: int,
    seed: int = 42,
) -> list[tuple[str, str]]:
    """Sample n (tf, target) pairs with no known TRRUST relationship, drawn from
    the intersection of all provided vocabularies.

    Returns plain string pairs (no embeddings attached) so the same list can be
    reused across scFM embedding dicts. Deterministic given
    (tsv_path, vocabularies, n, seed).
    """
    if not vocabularies:
        raise ValueError("vocabularies must contain at least one set")

    raw_records = _parse_tsv(Path(tsv_path))
    all_raw_pairs = _collect_raw_pairs(raw_records)

    trrust_gene_set: set[str] = set()
    for tf, target, _ in raw_records:
        trrust_gene_set.add(tf)
        trrust_gene_set.add(target)

    candidate_genes = sorted(trrust_gene_set.intersection(*vocabularies))
    candidates = [
        (a, b)
        for a in candidate_genes
        for b in candidate_genes
        if a != b and (a, b) not in all_raw_pairs
    ]
    if len(candidates) < n:
        raise ValueError(
            f"Not enough candidate None pairs: need {n}, only {len(candidates)} available"
        )
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(candidates), size=n, replace=False)
    return [candidates[i] for i in indices]


def count_relationship_pairs(
    tsv_path: str | Path,
    vocab: set[str],
    *,
    exclude_unknown: bool = False,
) -> int:
    """Count deduplicated TRRUST relationship pairs where both TF and target are in ``vocab``.

    Used to size the shared None pair list so class balance matches per-scFM loading.
    """
    raw_records = _parse_tsv(Path(tsv_path))
    deduped = _deduplicate(raw_records)
    count = 0
    for tf, target, regulation in deduped:
        if exclude_unknown and regulation == "Unknown":
            continue
        if tf in vocab and target in vocab:
            count += 1
    return count


def load_binary_trrust_data(
    tsv_path: str | Path,
    gene_embeddings: dict[str, np.ndarray],
    seed: int = 42,
    none_pairs: list[tuple[str, str]] | None = None,
) -> TRRUSTData:
    """Load TRRUST data for binary classification (Relationship vs None).

    All regulation types (Activation, Repression, Unknown) become label 1.
    If ``none_pairs`` is provided, those (tf, target) pairs are used as the
    None class (embeddings looked up from ``gene_embeddings``). Otherwise an
    equal number of None pairs are sampled from gene pairs with no known
    TRRUST relationship.
    """
    raw_records = _parse_tsv(Path(tsv_path))
    deduped = _deduplicate(raw_records)

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

    if none_pairs is None:
        all_raw_pairs = _collect_raw_pairs(raw_records)
        genes = _trrust_genes(raw_records, gene_embeddings)
        rng = np.random.default_rng(seed)
        none_records = _generate_none_pairs(
            all_raw_pairs, gene_embeddings, genes,
            n=len(relationship_records),
            none_label=BINARY_LABELS["None"],
            rng=rng,
        )
    else:
        none_records = _build_none_records_from_pairs(
            none_pairs, gene_embeddings, none_label=BINARY_LABELS["None"]
        )
    return TRRUSTData(records=relationship_records + none_records)


def filter_data_by_genes(
    data: TRRUSTData,
    genes: set[str] | list[str],
) -> TRRUSTData:
    """Return a new TRRUSTData keeping only records where both TF and target are in ``genes``."""
    gene_set = set(genes)
    kept = [r for r in data.records if r.tf in gene_set and r.target in gene_set]
    return TRRUSTData(records=kept)


def load_ternary_trrust_data(
    tsv_path: str | Path,
    gene_embeddings: dict[str, np.ndarray],
    seed: int = 42,
    none_pairs: list[tuple[str, str]] | None = None,
) -> TRRUSTData:
    """Load TRRUST data for ternary classification (Activation, Repression, None).

    Unknown entries are removed. If ``none_pairs`` is provided, those
    (tf, target) pairs are used as the None class. Otherwise None pairs are
    sampled to be ~1/3 of the total dataset. Unknown pairs are excluded from
    None sampling.
    """
    raw_records = _parse_tsv(Path(tsv_path))
    deduped = _deduplicate(raw_records)

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

    if none_pairs is None:
        all_raw_pairs = _collect_raw_pairs(raw_records)
        genes = _trrust_genes(raw_records, gene_embeddings)
        rng = np.random.default_rng(seed)
        n_none = len(real_records) // 2
        none_records = _generate_none_pairs(
            all_raw_pairs, gene_embeddings, genes,
            n=n_none,
            none_label=TERNARY_LABELS["None"],
            rng=rng,
        )
    else:
        none_records = _build_none_records_from_pairs(
            none_pairs, gene_embeddings, none_label=TERNARY_LABELS["None"]
        )
    return TRRUSTData(records=real_records + none_records)
