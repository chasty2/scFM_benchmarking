from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from anndata import AnnData
from scgpt.preprocess import Preprocessor
from scgpt.tokenizer.gene_tokenizer import GeneVocab, tokenize_and_pad_batch
from torch.utils.data import DataLoader, Dataset

from scfm_utils.constants import N_BINS, N_HVG, PAD_TOKEN, PAD_VALUE


class SeqDataset(Dataset):
    def __init__(self, data: dict):
        self.data = data

    def __len__(self):
        return self.data["genes"].shape[0]

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()}


@dataclass
class ScGPTDataset:
    dataloader: DataLoader
    genes_in_vocab: list[str]


def create_scgpt_dataset(
    adata: AnnData,
    vocab: GeneVocab,
    gene2idx: dict[str, int],
    *,
    n_hvg: int = N_HVG,
    n_bins: int = N_BINS,
    batch_key: str = "batch",
    data_is_raw: bool = False,
    max_cells: int | None = None,
    batch_size: int = 64,
) -> ScGPTDataset:
    """Preprocess an AnnData object and create a DataLoader for scGPT encoding."""
    # Preprocess
    preprocessor = Preprocessor(
        use_key="X",
        filter_gene_by_counts=3,
        filter_cell_by_counts=False,
        normalize_total=1e4,
        result_normed_key="X_normed",
        log1p=data_is_raw,
        result_log1p_key="X_log1p",
        subset_hvg=n_hvg,
        hvg_flavor="cell_ranger",
        binning=n_bins,
        result_binned_key="X_binned",
    )
    preprocessor(adata, batch_key=batch_key)

    # Filter genes to those in vocab
    genes_in_vocab = [g for g in adata.var.index if g in gene2idx]
    gene_ids = np.array([gene2idx[g] for g in genes_in_vocab])

    gene_mask = adata.var.index.isin(genes_in_vocab)
    counts = adata.layers["X_binned"][:, gene_mask]
    if hasattr(counts, "toarray"):
        counts = counts.toarray()
    counts = counts.astype(np.float32)

    # Tokenize and pad
    max_seq_len = len(genes_in_vocab) + 1  # +1 for <cls> token
    tokenized_data = tokenize_and_pad_batch(
        counts,
        gene_ids,
        max_len=max_seq_len,
        vocab=vocab,
        pad_token=PAD_TOKEN,
        pad_value=PAD_VALUE,
        append_cls=True,
        include_zero_gene=True,
    )

    # Optional cell subset
    if max_cells is not None:
        tokenized_data = {k: v[:max_cells] for k, v in tokenized_data.items()}

    dataset = SeqDataset(tokenized_data)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=4,
        pin_memory=True,
    )

    return ScGPTDataset(dataloader=dataloader, genes_in_vocab=genes_in_vocab)
