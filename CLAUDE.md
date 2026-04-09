# scFM Benchmarking

Benchmarking gene embeddings from single-cell foundation models to evaluate performance on gene regulatory network predictions.

## Stack

- **Package manager**: `uv` — use `uv run` to execute scripts and `uv sync` to install dependencies
- **ML framework**: PyTorch
- **Testing**: pytest
- **Notebooks**: Jupyter Lab

## Directory Structure

- `data/` — ML inputs and outputs (h5ad files, TSVs, gene lists)
- `models/` — saved model weights and configs
- `notebooks/` — Jupyter notebooks for experiments and analysis
- `reports/` — saved plots and figures
- `src/` — all source code imported into notebooks and tests
- `tests/` — pytest test files

## Source Code Layout

`src/` is a Python package. Import with `from src.<module> import ...`.

- `src/constants.py` — shared constants (token names, bin counts, etc.)
- `src/scgpt/` — scGPT model loading, dataset creation, and embedding encoding
- `src/trrust/` — TRRUST classifier model and training data utilities

## Running Tests

```bash
# Run all tests
uv run pytest

# Run a specific test file
uv run pytest tests/test_trrust.py -v
```

`tests/test_trrust.py` uses only synthetic data and runs without any external files.
`tests/test_scgpt.py` requires model weights in `models/scGPT_bc/` and data at `data/Immune_ALL_human.h5ad`.
