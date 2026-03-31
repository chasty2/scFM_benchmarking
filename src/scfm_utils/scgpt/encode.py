from __future__ import annotations

from pathlib import Path

import anndata
import numpy as np
import pandas as pd
import torch
from scgpt.model import TransformerModel
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from torch.utils.data import DataLoader
from tqdm import tqdm

from scfm_utils.constants import PAD_TOKEN


def encode_scgpt_embeddings_to_h5ad(
    model: TransformerModel,
    dataloader: DataLoader,
    vocab: GeneVocab,
    gene_names: list[str],
    cell_type: str,
    output_path: str | Path,
    device: torch.device | None = None,
    compression: str | None = "gzip",  # type: ignore[assignment]
) -> None:
    """Encode scGPT gene embeddings and write average embeddings to an h5ad file.

    h5ad file layout:

    /
    ├── X                      (n_genes, embsize) float32 — average gene embeddings
    ├── obs/                   gene names (obs index)
    └── uns/
        └── cell_type          string — the cell type label
    """
    if device is None:
        device = next(model.parameters()).device

    output_path = Path(output_path)
    n_genes = len(gene_names)
    embsize = model.d_model

    sum_acc = np.zeros((n_genes, embsize), dtype=np.float64)
    n_cells = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Encoding embeddings"):
            src = batch["genes"].to(device)
            values = batch["values"].to(device)
            src_key_padding_mask = src.eq(vocab[PAD_TOKEN]).to(device)

            output = model._encode(
                src=src,
                values=values,
                src_key_padding_mask=src_key_padding_mask,
            )

            gene_np = output[:, 1:, :].cpu().numpy()
            sum_acc += gene_np.sum(axis=0)
            n_cells += output.shape[0]

    avg = (sum_acc / n_cells).astype(np.float32)

    adata = anndata.AnnData(
        X=avg,
        obs=pd.DataFrame(index=gene_names),
    )
    adata.uns["cell_type"] = cell_type
    adata.write_h5ad(output_path, compression=compression)


def load_average_gene_embeddings(path: str | Path) -> anndata.AnnData:
    """Load average gene embeddings from an h5ad file."""
    return anndata.read_h5ad(path)
