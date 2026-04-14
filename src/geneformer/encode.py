from __future__ import annotations

import functools
import shutil
import tempfile
from pathlib import Path
from typing import Sequence

import anndata
import numpy as np
import pandas as pd

SPECIAL_TOKENS = ("<cls>", "<eos>", "<pad>", "<mask>")


def chunk_csv_path(chunks_dir: str | Path, chunk_index: int) -> Path:
    """Canonical CSV path for a chunk index — used by both extraction and resume logic."""
    return Path(chunks_dir) / f"chunk_{chunk_index}.csv"


def extract_chunk_gene_embeddings(
    emb_extractor,
    chunk_dataset,
    model_directory: str | Path,
    *,
    chunks_dir: str | Path,
    chunk_index: int,
) -> pd.DataFrame:
    """Run ``EmbExtractor.extract_embs`` on a single chunk and return the gene
    embeddings DataFrame. The CSV is written to ``chunk_csv_path(chunks_dir, chunk_index)``
    so callers can resume by checking that path before calling this function.
    """
    chunks_dir = Path(chunks_dir)
    chunks_dir.mkdir(parents=True, exist_ok=True)

    tmp_dir = Path(tempfile.mkdtemp(prefix=f"gf_chunk_{chunk_index}_"))
    try:
        chunk_path = tmp_dir / "chunk.dataset"
        chunk_dataset.save_to_disk(str(chunk_path))

        embs_df = emb_extractor.extract_embs(
            model_directory=str(model_directory),
            input_data_file=str(chunk_path),
            output_directory=str(chunks_dir),
            output_prefix=f"chunk_{chunk_index}",
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return embs_df


def combine_chunk_embeddings(
    chunk_csv_paths: Sequence[str | Path],
    *,
    weights: Sequence[float] | None = None,
) -> pd.DataFrame:
    """Weighted average across per-chunk gene embedding DataFrames over the union
    of gene IDs. Genes absent from a chunk contribute ``0`` to the average for
    that chunk (matches the prototype notebook's ``reindex(fill_value=0.0)``
    behavior).

    If ``weights`` is ``None``, weights are proportional to each chunk's row
    count.
    """
    if not chunk_csv_paths:
        raise ValueError("chunk_csv_paths is empty")

    chunk_dfs = [pd.read_csv(p, index_col=0) for p in chunk_csv_paths]

    if weights is None:
        sizes = np.array([df.shape[0] for df in chunk_dfs], dtype=np.float64)
    else:
        sizes = np.asarray(weights, dtype=np.float64)
        if len(sizes) != len(chunk_dfs):
            raise ValueError("weights length must match number of chunks")
    norm_weights = sizes / sizes.sum()

    all_genes = sorted(
        functools.reduce(lambda a, b: a | b, (set(df.index) for df in chunk_dfs))
    )

    combined = sum(
        w * df.reindex(all_genes, fill_value=0.0)
        for w, df in zip(norm_weights, chunk_dfs)
    )
    return combined


def save_geneformer_h5ad(
    embs_df: pd.DataFrame,
    *,
    ensembl_to_symbol: dict[str, str],
    cell_type: str,
    output_path: str | Path,
    compression: str | None = "gzip",
) -> None:
    """Write average gene embeddings to an h5ad file matching the scGPT layout.

    h5ad file layout:

    /
    ├── X                      (n_genes, embsize) float32 — average gene embeddings
    ├── obs/                   gene symbols (obs index) + ensembl_id column
    └── uns/
        └── cell_type          string — the cell type label
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = embs_df[~embs_df.index.isin(SPECIAL_TOKENS)]

    ensembl_ids = df.index.tolist()
    gene_symbols = [ensembl_to_symbol.get(eid, eid) for eid in ensembl_ids]

    obs = pd.DataFrame({"ensembl_id": ensembl_ids}, index=gene_symbols)

    adata = anndata.AnnData(
        X=df.values.astype(np.float32),
        obs=obs,
    )
    adata.uns["cell_type"] = cell_type
    adata.write_h5ad(output_path, compression=compression)


def load_average_gene_embeddings(path: str | Path) -> anndata.AnnData:
    """Load average gene embeddings from an h5ad file."""
    return anndata.read_h5ad(path)
