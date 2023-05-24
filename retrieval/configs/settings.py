#!/usr/bin/env python3
from dataclasses import dataclass


@dataclass
class TokenizerSettings:
    tok_name_or_path: str = "bert-base-uncased" # pr "../data/colbertv2.0/" or "roberta-base"
    query_token: str = "[Q]"
    doc_token: str = "[D]"


@dataclass
class ModelSettings:
    backbone_name_or_path: str = "bert-base-uncased" # pr "../data/colbertv2.0/" or "roberta-base"
    hidden_size: int = 768          # requires: <= 768
    num_hidden_layers: int = 12     # requires: <= 12
    num_attention_heads: int = 12   # requires: <= 12
    intermediate_size: int = 3072
    hidden_act: str = "gelu"
    dropout: float = 0.1
    skip_punctuation: bool = True
    similarity: str = "cosine" # "L2" or "cosine"
    intra_batch_similarity: bool = False
    normalize: bool = True


@dataclass
class DocSettings:
    dim: int = 128
    doc_maxlen: int = 220
    mask_punctuation: bool = True


@dataclass
class QuerySettings:
    query_maxlen: int = 32
    ignore_mask_tokens : bool = False
    interaction: str = "colbert"


@dataclass
class TrainingSettings:
    epochs: int = 10
    bucket_size: int = 128*10
    batch_size: int = 128
    accum_steps: int = 16
    passages_per_query: int = 1
    drop_last: bool = False


@dataclass
class IndexerSettings:
    n_clusters: int = 100
