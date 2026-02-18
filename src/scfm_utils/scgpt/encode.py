from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scgpt.model import TransformerModel
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from torch.utils.data import DataLoader

from scfm_utils.constants import PAD_TOKEN


@dataclass
class ScGPTEmbeddings:
    cls_embeddings: np.ndarray  # (n_cells, embsize)
    gene_embeddings: np.ndarray  # (n_cells, n_genes, embsize)


def encode_scgpt_embeddings(
    model: TransformerModel,
    dataloader: DataLoader,
    vocab: GeneVocab,
    device: torch.device | None = None,
) -> ScGPTEmbeddings:
    """Run scGPT encoder over a DataLoader and return CLS and gene embeddings."""
    if device is None:
        device = next(model.parameters()).device

    all_cls = []
    all_gene = []

    with torch.no_grad():
        for batch in dataloader:
            src = batch["genes"].to(device)
            values = batch["values"].to(device)
            src_key_padding_mask = src.eq(vocab[PAD_TOKEN]).to(device)

            output = model._encode(
                src=src,
                values=values,
                src_key_padding_mask=src_key_padding_mask,
            )

            all_cls.append(output[:, 0, :].cpu())
            all_gene.append(output[:, 1:, :].cpu())

    return ScGPTEmbeddings(
        cls_embeddings=torch.cat(all_cls, dim=0).numpy(),
        gene_embeddings=torch.cat(all_gene, dim=0).numpy(),
    )
