from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from src.trrust.classifier import TRRClassifierModel
from src.trrust.training_data import TRRUSTData


@dataclass
class TrainingResult:
    """Outputs of a single train/test run of a TRRUST classifier."""

    train_losses: list[float]
    test_losses: list[float]
    classification_report: dict
    gene_predictions: pd.DataFrame


def load_gene_embeddings(h5ad_path: str | Path) -> dict[str, np.ndarray]:
    """Load per-gene average embeddings from an h5ad file into a dict.

    Compatible with both scGPT and Geneformer encoded outputs: each stores
    gene symbols in ``obs_names`` and the embedding matrix in ``.X``.
    """
    adata = anndata.read_h5ad(h5ad_path)
    return {gene: adata.X[i] for i, gene in enumerate(adata.obs_names)}


def prepare_train_test_split(
    data: TRRUSTData,
    test_size: float = 0.1,
    seed: int = 42,
) -> tuple[TensorDataset, TensorDataset, pd.DataFrame]:
    """Stratified train/test split over a TRRUSTData object.

    Returns ``(train_ds, test_ds, test_metadata)`` where ``test_metadata`` is a
    DataFrame with ``tf`` and ``target`` columns aligned to test_ds row order,
    so ``train_classifier`` can label predictions with gene names.
    """
    tf_tensor = torch.from_numpy(data.tf_embeddings).float()
    tgt_tensor = torch.from_numpy(data.target_embeddings).float()
    label_tensor = torch.from_numpy(data.labels).long()

    indices = np.arange(len(data.records))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        stratify=data.labels,
        random_state=seed,
    )

    train_ds = TensorDataset(
        tf_tensor[train_idx], tgt_tensor[train_idx], label_tensor[train_idx]
    )
    test_ds = TensorDataset(
        tf_tensor[test_idx], tgt_tensor[test_idx], label_tensor[test_idx]
    )
    test_metadata = pd.DataFrame(
        {
            "tf": [data.records[i].tf for i in test_idx],
            "target": [data.records[i].target for i in test_idx],
        }
    )
    return train_ds, test_ds, test_metadata


def _compute_class_weights(labels: torch.Tensor, n_classes: int) -> torch.Tensor:
    class_counts = torch.bincount(labels, minlength=n_classes).float()
    return len(labels) / (n_classes * class_counts)


def train_classifier(
    train_ds: TensorDataset,
    test_ds: TensorDataset,
    test_metadata: pd.DataFrame,
    *,
    embsize: int,
    label_map: dict[str, int],
    lr: float,
    epochs: int,
    batch_size: int = 64,
    use_class_weights: bool = False,
    device: str | torch.device | None = None,
    seed: int = 42,
) -> TrainingResult:
    """Train a ``TRRClassifierModel`` on the provided train/test splits.

    ``label_map`` maps string labels → integer indices (e.g. ``BINARY_LABELS``).
    When ``use_class_weights=True``, inverse-frequency weights are computed
    from the training-split labels and passed to ``CrossEntropyLoss``.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    torch.manual_seed(seed)

    n_classes = len(label_map)
    label_names = {v: k for k, v in label_map.items()}

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    classifier = TRRClassifierModel(embsize=embsize, n_classes=n_classes).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)

    if use_class_weights:
        train_labels = train_ds.tensors[2]
        weights = _compute_class_weights(train_labels, n_classes).to(device)
        criterion = torch.nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    train_losses: list[float] = []
    test_losses: list[float] = []
    n_train = len(train_ds)
    n_test = len(test_ds)

    for _epoch in range(epochs):
        classifier.train()
        epoch_loss = 0.0
        for tf_b, tgt_b, lbl_b in train_loader:
            tf_b, tgt_b, lbl_b = tf_b.to(device), tgt_b.to(device), lbl_b.to(device)
            logits = classifier(tf_b, tgt_b)
            loss = criterion(logits, lbl_b)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(lbl_b)
        train_losses.append(epoch_loss / n_train)

        classifier.eval()
        test_loss = 0.0
        with torch.no_grad():
            for tf_b, tgt_b, lbl_b in test_loader:
                tf_b, tgt_b, lbl_b = tf_b.to(device), tgt_b.to(device), lbl_b.to(device)
                logits = classifier(tf_b, tgt_b)
                test_loss += criterion(logits, lbl_b).item() * len(lbl_b)
        test_losses.append(test_loss / n_test)

    classifier.eval()
    all_preds = []
    all_true = []
    with torch.no_grad():
        for tf_b, tgt_b, lbl_b in test_loader:
            logits = classifier(tf_b.to(device), tgt_b.to(device))
            all_preds.append(logits.argmax(dim=1).cpu())
            all_true.append(lbl_b)
    preds = torch.cat(all_preds).numpy()
    true = torch.cat(all_true).numpy()

    target_names = [label_names[i] for i in range(n_classes)]
    report = classification_report(
        true, preds, target_names=target_names, output_dict=True, zero_division=0
    )

    predictions_df = test_metadata.reset_index(drop=True).copy()
    predictions_df["true_relationship"] = [label_names[int(t)] for t in true]
    predictions_df["predicted_relationship"] = [label_names[int(p)] for p in preds]

    return TrainingResult(
        train_losses=train_losses,
        test_losses=test_losses,
        classification_report=report,
        gene_predictions=predictions_df,
    )
