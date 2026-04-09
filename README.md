# scFM_benchmarking
Benchmarking gene embeddings from single cell foundation models to evaluate their performance on gene regulatory network predictions

## System Dependencies

### uv

Install [uv](https://docs.astral.sh/uv/getting-started/installation/):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Geneformer

Geneformer requires the following system-level packages:

```bash
sudo apt install python3-dev git-lfs
git lfs install
```

- `python3-dev`: Provides Python C headers needed to compile native extensions
- `git-lfs`: Required to pull the large model weight files from Hugging Face

## Installation

```bash
uv sync
```

## Testing

```bash
uv run pytest tests/ -v
```
