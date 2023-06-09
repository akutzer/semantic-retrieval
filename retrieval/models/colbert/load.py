#!/usr/bin/env python3
from typing import Tuple, Optional

from retrieval.configs import BaseConfig
from retrieval.models.colbert.colbert import ColBERT
from retrieval.models.colbert.tokenizer import ColBERTTokenizer


def load_colbert_and_tokenizer(
    directory: str, device: str = "cpu", config: Optional[BaseConfig] = None
) -> Tuple[ColBERT, ColBERTTokenizer]:
    tokenizer = ColBERTTokenizer.from_pretrained(directory, config=config)
    colbert = ColBERT.from_pretrained(directory, device=device, config=config)
    colbert.register_tokenizer(tokenizer)

    return colbert, tokenizer


def get_colbert_and_tokenizer(
    config: BaseConfig, device: str = "cpu"
) -> Tuple[ColBERT, ColBERTTokenizer]:
    tokenizer = ColBERTTokenizer(config)
    colbert = ColBERT(config, device=device)
    colbert.register_tokenizer(tokenizer)

    return colbert, tokenizer
