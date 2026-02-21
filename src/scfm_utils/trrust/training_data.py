from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from scfm_utils.scgpt.encode import ScGPTEmbeddings

REGULATION_LABELS = {"Activation": 0, "Repression": 1, "Unknown": 2}


@dataclass
class TRRUSTRecord:
    """A single TF → target regulatory relationship."""

    tf: str
    target: str
    regulation: str


@dataclass
class TRRUSTData:
    """TRRUST regulatory relationships filtered to genes present in embeddings.

    Attributes:
        records: All deduplicated TRRUST records from the TSV.
        filtered_records: Records where both TF and target are in the embeddings.
        tf_embeddings: TF gene embeddings per sample, shape (n_samples, embsize).
        target_embeddings: Target gene embeddings per sample, shape (n_samples, embsize).
        labels: Integer-encoded regulation labels, shape (n_samples,).
        label_names: Mapping from integer label to regulation string.
        cell_types: Cell type for each sample, shape (n_samples,).
        average_gene_embeddings: Average gene embeddings per cell type,
            mapping cell type → (n_genes, embsize).
    """

    records: list[TRRUSTRecord]
    filtered_records: list[TRRUSTRecord]
    tf_embeddings: np.ndarray
    target_embeddings: np.ndarray
    labels: np.ndarray
    label_names: dict[int, str]
    cell_types: np.ndarray
    average_gene_embeddings: dict[str, np.ndarray]


def load_trrust_data(
    tsv_path: str | Path,
    embeddings: ScGPTEmbeddings,
) -> TRRUSTData:
    """Load TRRUST TSV and create training data from gene embeddings.

    For each cell type in the embeddings, creates one training sample per
    (TF, target) pair where both genes exist in the embedding vocabulary.
    Features are the concatenation of the TF and target average gene
    embeddings. Labels are the integer-encoded regulation type.

    Pairs are directional: (A, B) and (B, A) are treated as distinct.
    Pairs with conflicting regulation labels are dropped.
    """
    records = _parse_tsv(Path(tsv_path))
    records = _deduplicate(records)

    gene_to_idx = {g: i for i, g in enumerate(embeddings.gene_names)}

    filtered = [
        r for r in records if r.tf in gene_to_idx and r.target in gene_to_idx
    ]

    avg_embeddings = embeddings.average_gene_embeddings()

    tf_list = []
    target_list = []
    labels_list = []
    cell_types_list = []

    for cell_type, gene_embs in avg_embeddings.items():
        for rec in filtered:
            tf_list.append(gene_embs[gene_to_idx[rec.tf]])
            target_list.append(gene_embs[gene_to_idx[rec.target]])
            labels_list.append(REGULATION_LABELS[rec.regulation])
            cell_types_list.append(cell_type)

    label_names = {v: k for k, v in REGULATION_LABELS.items()}
    embsize = next(iter(avg_embeddings.values())).shape[1] if avg_embeddings else 0

    return TRRUSTData(
        records=records,
        filtered_records=filtered,
        tf_embeddings=np.stack(tf_list) if tf_list else np.empty((0, embsize)),
        target_embeddings=np.stack(target_list) if target_list else np.empty((0, embsize)),
        labels=np.array(labels_list, dtype=np.int64),
        label_names=label_names,
        cell_types=np.array(cell_types_list),
        average_gene_embeddings=avg_embeddings,
    )


def _parse_tsv(tsv_path: Path) -> list[TRRUSTRecord]:
    """Parse the 4-column TRRUST TSV (no header)."""
    records = []
    with open(tsv_path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 4:
                continue
            tf, target, regulation, _pmids = parts
            records.append(TRRUSTRecord(tf=tf, target=target, regulation=regulation))
    return records


def _deduplicate(records: list[TRRUSTRecord]) -> list[TRRUSTRecord]:
    """Remove (TF, target) pairs that have conflicting regulation labels.

    Pairs are directional: (A, B) and (B, A) are distinct.
    If the same (TF, target) appears with the same label, keep one copy.
    If it appears with different labels, drop all copies.
    """
    pair_labels: dict[tuple[str, str], set[str]] = defaultdict(set)
    pair_first: dict[tuple[str, str], TRRUSTRecord] = {}

    for rec in records:
        key = (rec.tf, rec.target)
        pair_labels[key].add(rec.regulation)
        if key not in pair_first:
            pair_first[key] = rec

    return [
        pair_first[key] for key, labels in pair_labels.items() if len(labels) == 1
    ]
