from dataclasses import dataclass



@dataclass
class TokenizerSettings:
    tok_name_or_path: str = "../data/colbertv2.0/"
    query_token_id: str = "[unused0]"
    doc_token_id: str = "[unused1]"
    query_token: str = "[Q]"
    doc_token: str = "[D]"


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


