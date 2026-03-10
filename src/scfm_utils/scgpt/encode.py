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

    h5ad file layout:

    /
    ├── X                      (n_cells, embsize) float32 — CLS token embeddings
    ├── obs/                   AnnData obs group
    │   ├── _index             cell index strings ("0", "1", ...)
    │   └── celltype           categorical with codes/categories
    ├── uns/
    │   └── gene_names         (n_genes,) byte strings — gene identifiers
    ├── embeddings/
    │   ├── gene               (n_cells, n_genes, embsize) float32 — per-gene embeddings
    │   ├── gene_avg           (n_cell_types, n_genes, embsize) float32 — per-cell-type averages
    │   └── gene_avg_cell_types  string array of cell type labels (axis 0 of gene_avg)
    └── attrs:
        └── encoding_complete  bool flag set to True after streaming finishes

    The embeddings/ group is a custom HDF5 group (not an anndata slot) since
    anndata does not natively support 3D arrays. All read functions use h5py
    directly rather than anndata.read_h5ad.
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
        emb_grp = h5f.require_group("embeddings")
        gene_ds = emb_grp.create_dataset(
            "gene",
            shape=(n_cells, n_genes, embsize),
            dtype="float32",
            chunks=(min(batch_size, n_cells), n_genes, embsize),
            compression=compression,
        )

        # Accumulators for per-cell-type average gene embeddings
        unique_types = np.unique(cell_types)
        sums = {ct: np.zeros((n_genes, embsize), dtype=np.float64) for ct in unique_types}
        counts = {ct: 0 for ct in unique_types}

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
                cls_np = output[:, 0, :].cpu().numpy()
                gene_np = output[:, 1:, :].cpu().numpy()
                cls_ds[row : row + bs] = cls_np
                gene_ds[row : row + bs] = gene_np

                # Accumulate per-cell-type sums
                batch_types = cell_types[row : row + bs]
                for ct in unique_types:
                    mask = batch_types == ct
                    if mask.any():
                        sums[ct] += gene_np[mask].sum(axis=0)
                        counts[ct] += int(mask.sum())

                row += bs

        # Write pre-computed per-cell-type average gene embeddings
        sorted_types = sorted(unique_types)
        avg_array = np.stack(
            [(sums[ct] / counts[ct]).astype(np.float32) for ct in sorted_types]
        )
        emb_grp.create_dataset(
            "gene_avg", data=avg_array, compression=compression,
        )
        emb_grp.create_dataset(
            "gene_avg_cell_types",
            data=np.array(sorted_types, dtype=h5py.string_dtype()),
        )

        h5f.attrs["encoding_complete"] = True


def _read_cell_types(obs_group: h5py.Group) -> np.ndarray:
    """Read cell types from an anndata obs group, handling categoricals."""
    celltype_group = obs_group["celltype"]
    if "categories" in celltype_group:
        categories = celltype_group["categories"][:]
        codes = celltype_group["codes"][:]
        cell_types = np.array([categories[c] for c in codes])
    else:
        cell_types = celltype_group[:]

    if cell_types.dtype.kind in ("S", "O"):
        cell_types = np.array(
            [x.decode() if isinstance(x, bytes) else str(x) for x in cell_types]
        )
    return cell_types


def load_cls_embeddings(path: str | Path) -> anndata.AnnData:
    """Load cls_embeddings as AnnData (X = cls_embeddings, obs = celltype)."""
    with h5py.File(path, "r") as h5f:
        cls_embs = h5f["X"][:]
        cell_types = _read_cell_types(h5f["obs"])
        obs_index = h5f["obs"]["_index"][:]
    obs_index = [x.decode() if isinstance(x, bytes) else str(x) for x in obs_index]
    return anndata.AnnData(
        X=cls_embs,
        obs=pd.DataFrame({"celltype": cell_types}, index=obs_index),
    )


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
            return h5f["embeddings"]["gene"][cell_indices]
        return h5f["embeddings"]["gene"][:]


def load_gene_names(path: str | Path) -> list[str]:
    """Read gene names from h5ad."""
    with h5py.File(path, "r") as h5f:
        raw = h5f["uns"]["gene_names"][:]
    return [g.decode() if isinstance(g, bytes) else str(g) for g in raw]


def load_average_gene_embeddings(
    path: str | Path, chunk_size: int = 256
) -> tuple[dict[str, np.ndarray], list[str]]:
    """Load per-cell-type average gene embeddings.

    If pre-computed averages exist in the file (written by
    encode_scgpt_embeddings_to_h5ad), reads them directly.
    Otherwise falls back to streaming computation.

    Returns:
        Tuple of (averages dict mapping cell type -> (n_genes, embsize), gene_names list).
    """
    with h5py.File(path, "r") as h5f:
        gene_names = list(h5f["uns"]["gene_names"][:])
        gene_names = [
            g.decode() if isinstance(g, bytes) else str(g) for g in gene_names
        ]

        # Fast path: pre-computed averages
        emb_grp = h5f.get("embeddings")
        if emb_grp is not None and "gene_avg" in emb_grp:
            avg_array = emb_grp["gene_avg"][:]
            raw_types = emb_grp["gene_avg_cell_types"][:]
            cell_type_labels = [
                t.decode() if isinstance(t, bytes) else str(t) for t in raw_types
            ]
            averages = {ct: avg_array[i] for i, ct in enumerate(cell_type_labels)}
            return averages, gene_names

        # Fallback: stream through full gene embeddings
        cell_types = _read_cell_types(h5f["obs"])
        gene_ds = h5f["embeddings"]["gene"]
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

        averages = {
            ct: (sums[ct] / counts[ct]).astype(np.float32) for ct in unique_types
        }
    return averages, gene_names
