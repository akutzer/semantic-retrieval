#!/usr/bin/env python3= 
from dataclasses import dataclass, asdict
from retrieval.configs.settings import *


@dataclass
class BaseConfig(TokenizerSettings, DocSettings, QuerySettings, DataLoaderSettings, TrainingSettings, ModelSettings):
    def asdict(self) -> dict:
        return asdict(self)




FANDOM_CONFIG_DEFAULT = BaseConfig(
    # TokenizerSettings
    tok_name_or_path = "bert-base-uncased", # "bert-base-uncased" or "../data/colbertv2.0/" or "roberta-base"

    # ModelSettings
    backbone_name_or_path = "bert-base-uncased", # "bert-base-uncased" or "../data/colbertv2.0/" or "roberta-base"
    num_hidden_layers = 12,     # requires: <= 12
    num_attention_heads = 12,   # requires: <= 12
    dropout = 0.1,
    dim = 128,
    skip_punctuation = True,
    similarity = "cosine", # "L2" or "cosine"
    normalize = True,

    # DocSettings
    doc_maxlen = 220,
    ignore_mask_tokens = True,

    # QuerySettings:
    query_maxlen = 32,

    # DataLoaderSettings:
    bucket_size = 24,
    batch_size = 24,
    accum_steps = 1,
    passages_per_query = -1, # not used by QPP-style datasets
    shuffle = True,
    drop_last = True,
    pin_memory = True,
    num_workers = 4,

    # TrainingSettings:
    epochs = 10,
    lr_warmup_epochs = 2,
    lr_warmup_decay = 1/3,
    use_amp = True,
)