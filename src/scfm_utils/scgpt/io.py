from __future__ import annotations

from pathlib import Path

import anndata
import h5py
import numpy as np
import pandas as pd
import torch
from scgpt.model import TransformerModel
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from torch.utils.data import DataLoader
from tqdm import tqdm

from scfm_utils.constants import PAD_TOKEN
from scfm_utils.scgpt.encode import ScGPTEmbeddings


def encode_scgpt_embeddings_to_h5ad(
    model: TransformerModel,
    dataloader: DataLoader,
    vocab: GeneVocab,
    gene_names: list[str],
    cell_types: np.ndarray,
    output_path: str | Path,
    device: torch.device | None = None,
    compression: str | None = "gzip",
) -> None:
    """Stream scGPT embeddings to an h5ad file batch-by-batch.

    CLS embeddings are stored in the AnnData X slot (n_cells, embsize).
    Gene embeddings are stored as a custom h5py dataset at /gene_embeddings
    (n_cells, n_genes, embsize).
    """
    if device is None:
        device = next(model.parameters()).device

    output_path = Path(output_path)
    n_cells = len(dataloader.dataset)
    n_genes = len(gene_names)
    embsize = model.d_model

    # Step 1: Write AnnData scaffold with placeholder X
    adata = anndata.AnnData(
        X=np.zeros((n_cells, embsize), dtype=np.float32),
        obs=pd.DataFrame(
            {"celltype": cell_types},
            index=[str(i) for i in range(n_cells)],
        ),
    )
    adata.uns["gene_names"] = np.array(gene_names, dtype=object)
    adata.write_h5ad(output_path)
    del adata

    # Step 2: Reopen with h5py, replace X, add gene_embeddings, stream batches
    batch_size = dataloader.batch_size or 64
    with h5py.File(output_path, "a") as h5f:
        del h5f["X"]
        cls_ds = h5f.create_dataset(
            "X",
            shape=(n_cells, embsize),
            dtype="float32",
            chunks=(min(batch_size, n_cells), embsize),
            compression=compression,
        )
        gene_ds = h5f.create_dataset(
            "gene_embeddings",
            shape=(n_cells, n_genes, embsize),
            dtype="float32",
            chunks=(min(batch_size, n_cells), n_genes, embsize),
            compression=compression,
        )

        row = 0
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

                bs = output.shape[0]
                cls_ds[row : row + bs] = output[:, 0, :].cpu().numpy()
                gene_ds[row : row + bs] = output[:, 1:, :].cpu().numpy()
                row += bs

        h5f.attrs["encoding_complete"] = True


def load_scgpt_embeddings(path: str | Path) -> ScGPTEmbeddings:
    """Load full embeddings from h5ad into ScGPTEmbeddings.

    Note: This loads gene_embeddings entirely into RAM. For large datasets,
    prefer load_cls_embeddings() and load_average_gene_embeddings().
    """
    adata = anndata.read_h5ad(path)
    with h5py.File(path, "r") as h5f:
        gene_embs = h5f["gene_embeddings"][:]
        gene_names = list(h5f["uns"]["gene_names"][:])
    return ScGPTEmbeddings(
        cls_embeddings=np.asarray(adata.X),
        gene_embeddings=gene_embs,
        gene_names=gene_names,
        cell_types=np.asarray(adata.obs["celltype"].values),
    )


def load_cls_embeddings(path: str | Path) -> anndata.AnnData:
    """Load cls_embeddings as AnnData (X = cls_embeddings, obs = celltype)."""
    return anndata.read_h5ad(path)


def load_gene_embeddings(
    path: str | Path, cell_indices: np.ndarray | slice | None = None
) -> np.ndarray:
    """Load gene embeddings (or a slice) via h5py.

    Args:
        path: Path to the h5ad file.
        cell_indices: Optional indices/slice to load a subset of cells.

    Returns:
        Array of shape (n_cells, n_genes, embsize) or subset thereof.
    """
    with h5py.File(path, "r") as h5f:
        if cell_indices is not None:
            return h5f["gene_embeddings"][cell_indices]
        return h5f["gene_embeddings"][:]


def load_gene_names(path: str | Path) -> list[str]:
    """Read gene names from h5ad."""
    with h5py.File(path, "r") as h5f:
        return list(h5f["uns"]["gene_names"][:])


def load_average_gene_embeddings(
    path: str | Path, chunk_size: int = 256
) -> tuple[dict[str, np.ndarray], list[str]]:
    """Stream gene embeddings and compute per-cell-type averages.

    Memory usage: O(n_cell_types * n_genes * embsize) instead of loading
    the full gene_embeddings array.

    Returns:
        Tuple of (averages dict mapping cell type -> (n_genes, embsize), gene_names list).
    """
    with h5py.File(path, "r") as h5f:
        # Read cell types from obs — anndata stores categoricals with codes/categories
        obs_group = h5f["obs"]
        celltype_group = obs_group["celltype"]
        if "categories" in celltype_group:
            categories = celltype_group["categories"][:]
            codes = celltype_group["codes"][:]
            cell_types = np.array([categories[c] for c in codes])
        else:
            cell_types = celltype_group[:]

        # Decode bytes to str if needed
        if cell_types.dtype.kind == "S" or cell_types.dtype.kind == "O":
            cell_types = np.array(
                [x.decode() if isinstance(x, bytes) else str(x) for x in cell_types]
            )

        gene_ds = h5f["gene_embeddings"]
        n_cells = gene_ds.shape[0]

        unique_types = np.unique(cell_types)
        sums = {
            ct: np.zeros(gene_ds.shape[1:], dtype=np.float64) for ct in unique_types
        }
        counts = {ct: 0 for ct in unique_types}

        for start in range(0, n_cells, chunk_size):
            end = min(start + chunk_size, n_cells)
            chunk = gene_ds[start:end]
            chunk_types = cell_types[start:end]
            for ct in unique_types:
                mask = chunk_types == ct
                if mask.any():
                    sums[ct] += chunk[mask].sum(axis=0)
                    counts[ct] += int(mask.sum())

        gene_names = list(h5f["uns"]["gene_names"][:])
        gene_names = [
            g.decode() if isinstance(g, bytes) else str(g) for g in gene_names
        ]

        averages = {
            ct: (sums[ct] / counts[ct]).astype(np.float32) for ct in unique_types
        }
    return averages, gene_names
