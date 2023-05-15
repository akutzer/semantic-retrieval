from retrieval.configs import BaseConfig
from retrieval.models.colbert.colbert import ColBERT
from retrieval.models.colbert.tokenizer import ColBERTTokenizer


def load_colbert_and_tokenizer(self, directory: str, device: str = "cpu"):
    tokenizer = ColBERTTokenizer.from_pretrained(directory)
    colbert = ColBERT.from_pretrained(directory, device=device)
    colbert.register_tokenizer(tokenizer)

    return colbert, tokenizer


def get_colbert_and_tokenizer(self, config: BaseConfig, device: str = "cpu"):
    tokenizer = ColBERTTokenizer(config)
    colbert = ColBERT(config, device=device)
    colbert.register_tokenizer(tokenizer)

    return colbert, tokenizer
