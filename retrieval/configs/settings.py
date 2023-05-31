#!/usr/bin/env python3
from dataclasses import dataclass
import torch


@dataclass
class TokenizerSettings:
    tok_name_or_path: str = "bert-base-uncased"  # "bert-base-uncased" or "../data/colbertv2.0/" or "roberta-base"
    query_token: str = "[Q]"
    doc_token: str = "[D]"


@dataclass
class ModelSettings:
    backbone_name_or_path: str = "bert-base-uncased"  # "bert-base-uncased" or "../data/colbertv2.0/" or "roberta-base"
    hidden_size: int = 768  # requires: <= 768
    num_hidden_layers: int = 12  # requires: <= 12
    num_attention_heads: int = 12  # requires: <= 12
    intermediate_size: int = 3072
    dim: int = 128
    hidden_act: str = "gelu"
    dropout: float = 0.1
    skip_punctuation: bool = True
    similarity: str = "cosine"  # "L2" or "cosine"
    intra_batch_similarity: bool = (
        False  # should always be deactivated when using QQP-style datasets
    )
    normalize: bool = True


@dataclass
class DocSettings:
    doc_maxlen: int = 220
    ignore_mask_tokens: bool = True


@dataclass
class QuerySettings:
    query_maxlen: int = 32


@dataclass
class DataLoaderSettings:
    bucket_size: int = 128
    batch_size: int = 128
    accum_steps: int = 16
    passages_per_query: int = 10  # only used by QPP-style datasets
    shuffle: bool = False
    drop_last: bool = False
    pin_memory: bool = False
    # num_workers: int = 0


@dataclass
class TrainingSettings:
    epochs: int = 10
    lr: float = 5e-6
    warmup_epochs: int = 2
    warmup_start_factor: float = 1 / 10
    use_amp: bool = False
    num_gpus: int = 0

    def __post_init__(self):
        if self.num_gpus < 0:
            raise ValueError("num_gpus cannot be negative")

        available_gpus = torch.cuda.device_count()
        if self.num_gpus > available_gpus:
            print(
                f"Lowering number of GPUs from {self.num_gpus} to the maximal "
                "amount of available GPUs {available_gpus}"
            )
            self.num_gpus = available_gpus

    # @property
    # def num_gpus(self) -> int:
    #     return self.__num_gpus

    # @num_gpus.setter
    # def num_gpus(self, num_gpus: int):
    #     available_gpus = torch.cuda.device_count()
    #     if num_gpus > available_gpus:
    #         print(
    #             f"Lowering number of GPUs from {num_gpus} to the maximal "
    #             "amount of available GPUs {available_gpus}"
    #         )
    #         num_gpus = available_gpus
    #     self.__num_gpus = num_gpus


@dataclass
class IndexerSettings:
    n_clusters: int = 100
    interaction: str = "colbert"
