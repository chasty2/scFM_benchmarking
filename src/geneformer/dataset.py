from __future__ import annotations

import pickle
from collections.abc import Iterator
from pathlib import Path

import numpy as np
from anndata import AnnData
from datasets import Dataset, load_from_disk


def prepare_adata_for_geneformer(
    adata: AnnData,
    ensembl_dict_path: str | Path | None = None,
) -> AnnData:
    """Add ``var["ensembl_id"]`` and ``obs["n_counts"]`` so the AnnData is ready for
    ``TranscriptomeTokenizer``.

    Genes whose symbol is not in the dictionary are assigned ``"unknown"``.
    """
    if ensembl_dict_path is None:
        from geneformer import ENSEMBL_DICTIONARY_FILE

        ensembl_dict_path = ENSEMBL_DICTIONARY_FILE

    with open(ensembl_dict_path, "rb") as f:
        gene_name_to_ensembl: dict[str, str] = pickle.load(f)

    adata.var["ensembl_id"] = [
        gene_name_to_ensembl.get(g, "unknown") for g in adata.var_names
    ]

    if "n_counts" not in adata.obs.columns:
        adata.obs["n_counts"] = np.asarray(adata.X.sum(axis=1)).flatten()

    return adata


def tokenize_adata(
    input_dir: str | Path,
    output_dir: str | Path,
    output_prefix: str,
    *,
    custom_attr_name_dict: dict[str, str] | None = None,
    nproc: int = 4,
    file_format: str = "h5ad",
) -> Path:
    """Tokenize every h5ad in ``input_dir`` and save the resulting HF dataset."""
    from geneformer import TranscriptomeTokenizer

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tk = TranscriptomeTokenizer(
        custom_attr_name_dict=custom_attr_name_dict,
        nproc=nproc,
    )
    tk.tokenize_data(
        data_directory=str(input_dir),
        output_directory=str(output_dir),
        output_prefix=output_prefix,
        file_format=file_format,
    )
    return output_dir / f"{output_prefix}.dataset"


def filter_tokenized_by_celltype(
    dataset_path: str | Path,
    cell_type: str,
    *,
    attr: str = "cell_type",
    num_proc: int = 4,
    sort_by_length: bool = True,
) -> Dataset:
    """Load a tokenized dataset from disk and filter to a single cell type."""
    ds = load_from_disk(str(dataset_path))
    ds = ds.filter(lambda x: x[attr] == cell_type, num_proc=num_proc)
    if sort_by_length and "length" in ds.column_names:
        ds = ds.sort("length", reverse=True)
    return ds


def iter_chunks(dataset: Dataset, chunk_size: int) -> Iterator[tuple[int, Dataset]]:
    """Yield ``(chunk_index, chunk_dataset)`` pairs of size ``chunk_size``."""
    n = len(dataset)
    for i, start in enumerate(range(0, n, chunk_size)):
        end = min(start + chunk_size, n)
        yield i, dataset.select(range(start, end))
