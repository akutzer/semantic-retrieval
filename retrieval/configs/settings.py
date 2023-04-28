from dataclasses import dataclass



@dataclass
class TokenizerSettings:
    tok_name_or_path: str = "../data/colbertv2.0/"
    query_token_id: str = "[unused0]"
    doc_token_id: str = "[unused1]"
    query_token: str = "[Q]"
    doc_token: str = "[D]"


@dataclass
class ModelSettings:
    backbone_name_or_path: str = "../data/colbertv2.0/" # "bert-base-uncased"
    dim: int = 69
    skip_punctuation: bool = True
    similarity: str = "cosine" # "L2" or "cosine"


@dataclass
class DocSettings:
    dim: int = 128
    doc_maxlen: int = 220
    mask_punctuation: bool = True


@dataclass
class QuerySettings:
    query_maxlen: int = 32
    attend_to_mask_tokens : bool = False
    interaction: str = "colbert"


@dataclass
class TrainingSettings:
    batch_size: int = 128
    accum_steps: int = 16
    passages_per_query: int = 1
    drop_last: bool = False