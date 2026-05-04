# scFM_benchmarking

The purpose of this investigation is to evaluate how well scFMs represent relationships between genes. To do this, we trained simple classifier models from the outputs of two different scFMs: scGPT and Geneformer. The predictive accuracy of these models was benchmarked against known ground truths of transcriptional regulatory relationships using the TRRUST database. These models showed promise in predicting high-level relationships between genes (Relationship vs. No Relationship), but struggled to predict more nuanced relationships (Activation vs. Repression vs. None). Future work to train classifiers on larger datasets and fine-tuned scFM could potentially improve model performance.

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
