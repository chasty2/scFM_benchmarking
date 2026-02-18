from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import torch
from scgpt.model import TransformerModel
from scgpt.tokenizer.gene_tokenizer import GeneVocab

from scfm_utils.constants import N_BINS, PAD_VALUE, SPECIAL_TOKENS

logger = logging.getLogger(__name__)


@dataclass
class ScGPTModelBundle:
    model: TransformerModel
    vocab: GeneVocab
    gene2idx: dict[str, int]
    config: dict


def load_scgpt_model(
    model_dir: str | Path,
    device: torch.device | None = None,
) -> ScGPTModelBundle:
    """Load an scGPT model from a directory containing args.json, best_model.pt, and vocab.json."""
    model_dir = Path(model_dir)
    vocab_file = model_dir / "vocab.json"
    config_file = model_dir / "args.json"
    model_file = model_dir / "best_model.pt"

    # Load vocabulary
    vocab = GeneVocab.from_file(vocab_file)
    for s in SPECIAL_TOKENS:
        if s not in vocab:
            vocab.append_token(s)

    # Load model config
    with open(config_file) as f:
        config = json.load(f)

    embsize = config["embsize"]
    nhead = config["nheads"]
    d_hid = config["d_hid"]
    nlayers = config["nlayers"]

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate model
    ntokens = len(vocab)
    model = TransformerModel(
        ntokens,
        embsize,
        nhead,
        d_hid,
        nlayers,
        vocab=vocab,
        pad_value=PAD_VALUE,
        n_input_bins=N_BINS,
    )

    # Load weights with partial-load fallback
    try:
        model.load_state_dict(torch.load(model_file, map_location=device))
        logger.info("Loaded all model params from %s", model_file)
    except Exception:
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_file, map_location=device)
        pretrained_dict = {
            k: v
            for k, v in pretrained_dict.items()
            if k in model_dict and v.shape == model_dict[k].shape
        }
        for k, v in pretrained_dict.items():
            logger.info("Loading params %s with shape %s", k, v.shape)
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    model.to(device)
    model.eval()

    gene2idx = vocab.get_stoi()

    return ScGPTModelBundle(
        model=model,
        vocab=vocab,
        gene2idx=gene2idx,
        config=config,
    )
