"""Encode gene embeddings for a single cell type using Geneformer.

Resumable: per-chunk CSVs are written to ``CHUNKS_DIR`` and any chunk whose CSV
already exists is skipped, so the script can be re-run after a crash without
losing completed work. After all chunks are present the script writes the final
weighted-average embeddings as an h5ad file matching the scGPT layout.

Configure by editing the constants below, then::

    uv run python notebooks/geneformer_encoding.py
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import anndata
from geneformer import EmbExtractor

# Allow `from src...` when running this file directly with `python`.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.geneformer import (  # noqa: E402
    chunk_csv_path,
    combine_chunk_embeddings,
    extract_chunk_gene_embeddings,
    filter_tokenized_by_celltype,
    iter_chunks,
    save_geneformer_h5ad,
)

# ---------------------------------------------------------------------------
# Configuration — edit these for each run.
# ---------------------------------------------------------------------------

TOKENIZED_DATASET = _REPO_ROOT / "data/geneformer_tokenized/immune_human.dataset"
PREPARED_ADATA = _REPO_ROOT / "data/geneformer_prep/Immune_ALL_human.h5ad"
MODEL_NAME = "ctheodoris/Geneformer"
CELL_TYPE = "NKT cells"
CHUNK_SIZE = 50
CHUNKS_DIR = _REPO_ROOT / "data/geneformer_output/nkt_chunks"
OUTPUT_H5AD = _REPO_ROOT / "data/embeddings/geneformer_nkt.h5ad"

# EmbExtractor settings — match the prototype notebook.
FORWARD_BATCH_SIZE = 2
EMB_LAYER = -1
SUMMARY_STAT = "mean"

# Smoke-test cap. Set to ``None`` for a full run; e.g. ``2`` to encode only the
# first two chunks end-to-end.
MAX_CHUNKS: int | None = None


def main() -> None:
    print(f"[{time.strftime('%H:%M:%S')}] loading tokenized dataset", flush=True)
    ds = filter_tokenized_by_celltype(TOKENIZED_DATASET, CELL_TYPE)
    n_cells = len(ds)
    n_chunks = math.ceil(n_cells / CHUNK_SIZE)
    print(
        f"[{time.strftime('%H:%M:%S')}] {CELL_TYPE}: {n_cells} cells → "
        f"{n_chunks} chunks of {CHUNK_SIZE}",
        flush=True,
    )

    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

    embex = EmbExtractor(
        model_type="Pretrained",
        num_classes=0,
        emb_mode="gene",
        max_ncells=None,
        emb_layer=EMB_LAYER,
        forward_batch_size=FORWARD_BATCH_SIZE,
        summary_stat=SUMMARY_STAT,
    )

    chunks_to_process = iter_chunks(ds, CHUNK_SIZE)
    for i, chunk in chunks_to_process:
        if MAX_CHUNKS is not None and i >= MAX_CHUNKS:
            print(f"[{time.strftime('%H:%M:%S')}] hit MAX_CHUNKS={MAX_CHUNKS}, stopping", flush=True)
            break

        csv_path = chunk_csv_path(CHUNKS_DIR, i)
        if csv_path.exists():
            print(
                f"[{time.strftime('%H:%M:%S')}] chunk {i + 1}/{n_chunks}: "
                f"skip (csv exists at {csv_path.name})",
                flush=True,
            )
            continue

        t0 = time.time()
        extract_chunk_gene_embeddings(
            embex,
            chunk,
            model_directory=MODEL_NAME,
            chunks_dir=CHUNKS_DIR,
            chunk_index=i,
        )
        elapsed = time.time() - t0
        print(
            f"[{time.strftime('%H:%M:%S')}] chunk {i + 1}/{n_chunks}: "
            f"{len(chunk)} cells, {elapsed:.1f}s → {csv_path.name}",
            flush=True,
        )

    expected = n_chunks if MAX_CHUNKS is None else min(MAX_CHUNKS, n_chunks)
    chunk_paths = sorted(
        CHUNKS_DIR.glob("chunk_*.csv"),
        key=lambda p: int(p.stem.split("_")[-1]),
    )
    if len(chunk_paths) < expected:
        print(
            f"[{time.strftime('%H:%M:%S')}] only {len(chunk_paths)}/{expected} "
            f"chunks present; skipping combine",
            flush=True,
        )
        return

    print(f"[{time.strftime('%H:%M:%S')}] combining {len(chunk_paths)} chunks", flush=True)
    embs_df = combine_chunk_embeddings(chunk_paths)
    print(f"[{time.strftime('%H:%M:%S')}] combined shape: {embs_df.shape}", flush=True)

    print(f"[{time.strftime('%H:%M:%S')}] loading prepared AnnData for symbol map", flush=True)
    prep = anndata.read_h5ad(PREPARED_ADATA)
    ensembl_to_symbol = {
        eid: sym
        for sym, eid in zip(prep.var_names, prep.var["ensembl_id"])
        if eid != "unknown"
    }

    save_geneformer_h5ad(
        embs_df,
        ensembl_to_symbol=ensembl_to_symbol,
        cell_type=CELL_TYPE,
        output_path=OUTPUT_H5AD,
    )
    print(f"[{time.strftime('%H:%M:%S')}] wrote {OUTPUT_H5AD}", flush=True)


if __name__ == "__main__":
    main()
