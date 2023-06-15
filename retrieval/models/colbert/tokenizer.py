#!/usr/bin/env python3
import os
from typing import Union, List, Optional
import logging

import torch
from transformers import AutoTokenizer

from retrieval.configs import BaseConfig, save_config, load_config
from retrieval.models.colbert.utils import _split_into_batches


class ColBERTTokenizer:
    def __init__(self, config: BaseConfig, tok=None):
        self.config = config

        if tok is not None:
            self.tok = tok
        else:
            self.tok = AutoTokenizer.from_pretrained(config.tok_name_or_path)

        self.query_maxlen = self.config.query_maxlen
        self.doc_maxlen = self.config.doc_maxlen

        self._init_special_tokens()

    def __len__(self):
        return len(self.tok)

    def tokenize(
        self,
        text: Union[str, List[str]],
        mode: str,
        add_special_tokens: bool = False,
        truncate: bool = False,
    ) -> Union[List[str], List[List[str]]]:
        """
        Splits the input sequence(s) into a list of tokens represented as substrings.
        """
        assert isinstance(text, str) or (
            isinstance(text, (list, tuple)) and all(isinstance(t, str) for t in text)
        )
        assert isinstance(mode, str) and mode in ["query", "doc"]

        is_single_str = isinstance(text, str)
        if is_single_str:
            text = [text]

        # tokenize the strings, adding " " to the start of each string
        # to ensure that the first token is treated as the start of a word
        tokens = [
            self.tok.tokenize(f" {seq}", add_special_tokens=False) for seq in text
        ]

        if truncate:
            maxlen = self.query_maxlen if mode == "query" else self.doc_maxlen
            if add_special_tokens:
                maxlen -= 3  # account for [CLS], [Q]/[D], and [SEP] tokens
            tokens = [tok_seq[:maxlen] for tok_seq in tokens]

        if add_special_tokens:
            if mode == "query":
                prefix, suffix = [self.cls_token, self.Q_marker_token], [self.sep_token]
                # if a query is shorter than the max_length, then we will pad it
                # up to this length using [MASK] tokens
                tokens = [
                    prefix
                    + tok_seq
                    + suffix
                    + [self.mask_token] * (self.query_maxlen - (len(tok_seq) + 3))
                    for tok_seq in tokens
                ]
            else:
                prefix, suffix = [self.cls_token, self.D_marker_token], [self.sep_token]
                tokens = [prefix + tok_seq + suffix for tok_seq in tokens]

        return tokens[0] if is_single_str else tokens

    def encode(
        self,
        text: Union[str, List[str]],
        mode: str,
        add_special_tokens: bool = False,
        truncate: bool = False,
    ) -> Union[List[int], List[List[int]]]:
        """
        Splits the input sequence(s) into a list of tokens represented by their ids.
        """
        assert isinstance(text, str) or (
            isinstance(text, (list, tuple)) and all(isinstance(t, str) for t in text)
        )
        assert isinstance(mode, str) and mode in ["query", "doc"]

        is_single_str = isinstance(text, str)
        if is_single_str:
            text = [text]

        # tokenize the strings; the " " is added so the first token is viewed
        # as the beginning of a word and not a continuation of a previous word
        text = [f" {seq}" for seq in text]
        ids = self.tok(
            text,
            add_special_tokens=False,
            return_attention_mask=False
        )["input_ids"]

        if truncate:
            maxlen = self.query_maxlen if mode == "query" else self.doc_maxlen
            if add_special_tokens:
                maxlen -= 3  # account for [CLS], [Q]/[D], and [SEP] tokens
            ids = [id_seq[:maxlen] for id_seq in ids]

        if add_special_tokens:
            if mode == "query":
                prefix, suffix = [self.cls_token_id, self.Q_marker_token_id], [self.sep_token_id]
                # if a query is shorter than the max_length, then we will pad it
                # up to this length using [MASK] tokens
                ids = [
                    prefix
                    + id_seq
                    + suffix
                    + [self.mask_token_id] * (self.query_maxlen - (len(id_seq) + 3))
                    for id_seq in ids
                ]
            else:
                prefix, suffix = [self.cls_token_id, self.D_marker_token_id], [self.sep_token_id]
                ids = [
                    prefix
                    + id_seq
                    + suffix
                    for id_seq in ids
                ]

        return ids[0] if is_single_str else ids

    def tensorize(
        self,
        text: Union[str, List[str]],
        mode: str,
        bsize: Union[None, int] = None,
        return_tensors: str = "pt",
        pin_memory: bool = False,
    ) -> torch.IntTensor:
        """
        Tokenizes and pads the input sequence(s) and returns them as a Tensor if no bsize is given or
        as a List of Tensors if a bsize was given.
        Note: The output will always be a 2-d Tensor of shape [B, min(L, L_max)]
        B     - batch size
        L     - length of the tokenized sequence
        L_max - maximal length of a tokenized sequence
        """

        assert isinstance(text, str) or (
            isinstance(text, (list, tuple)) and all(isinstance(t, str) for t in text)
        )
        assert isinstance(mode, str) and mode in ["query", "doc"]

        is_single_str = isinstance(text, str)
        if is_single_str:
            text = [text]

        if mode == "query":
            marker_id = self.Q_marker_token_id
            maxlen = self.query_maxlen
            padding = "max_length"

        else:
            marker_id = self.D_marker_token_id
            maxlen = self.doc_maxlen
            padding = "longest"

        # add placehold for the [Q]/[D] marker
        text = [f". {seq}" for seq in text]

        encoding = self.tok(
            text,
            padding=padding,
            truncation=True,
            return_tensors=return_tensors,
            max_length=maxlen,
        )

        ids, mask = encoding["input_ids"], encoding["attention_mask"]

        # postprocessing for the [Q]/[D]
        ids[:, 1] = marker_id
        # postprocessing for the [MASK] augmentation
        if mode == "query":
            bool_mask = ids == self.pad_token_id
            ids[bool_mask] = self.mask_token_id
            if not self.config.ignore_mask_tokens:
                mask[bool_mask] = 1

        if pin_memory:
            ids, mask = ids.pin_memory(), mask.pin_memory()

        if bsize:
            return _split_into_batches(ids, mask, bsize)

        # return (ids[0], mask[0]) if is_single_str else (ids, mask)
        return (ids, mask)

    def decode(self, ids: Union[List[int], torch.IntTensor], **kwargs) -> str:
        return self.tok.decode(ids, **kwargs)

    def batch_decode(
        self, ids: Union[List[List[int]], torch.IntTensor], **kwargs
    ) -> List[str]:
        return self.tok.batch_decode(ids, **kwargs)

    def save(self, save_directory: str, store_config: bool = True):
        # create the directory if it doesn't exist
        os.makedirs(save_directory, exist_ok=True)

        # save the tokenizer
        self.tok.save_pretrained(save_directory)

        # save the model's config if available
        if store_config:
            config_path = os.path.join(save_directory, "colbert_config.json")
            save_config(self.config, config_path)

    @classmethod
    def from_pretrained(cls, directory: str, config: Optional[BaseConfig] = None) -> "ColBERTTokenizer":
        # load the model's config if necessary
        if not isinstance(config, BaseConfig):
            config_path = os.path.join(directory, "colbert_config.json")
            config = load_config(config_path)
        if not config:
            logging.basicConfig(level=logging.WARNING, format="[%(asctime)s][%(levelname)s] %(message)s", datefmt="%y-%m-%d %H:%M:%S")
            logging.warning("colbert_config.json does not exist, loading default config.")
            config = BaseConfig()

        # load the tokenizers's parameters if available
        tok = AutoTokenizer.from_pretrained(directory, use_auth_token=False)
        tokenizer = cls(config, tok=tok)
        tokenizer._init_special_tokens()

        return tokenizer

    def __str__(self):
        return str(self.tok)

    def __repr__(self):
        return repr(self.tok)
    
    def _init_special_tokens(self):
        is_bert = "bert-base" in self.config.tok_name_or_path.lower() or "colbertv2" in self.config.tok_name_or_path.lower()
        if is_bert:
            logging.info("Detected BERT Tokenizer. Using unused tokens for [Q]/[D] tokens")
            # if loading the BERT or ColBERTv2 weights, the unused tokens 0 and 1
            # are used for the [Q]/[D] token
            self.Q_marker_token = self.config.query_token
            self.Q_marker_token_id = self.tok.convert_tokens_to_ids("[unused0]")

            self.D_marker_token = self.config.doc_token
            self.D_marker_token_id = self.tok.convert_tokens_to_ids("[unused1]")
        else:
            # instead of using already existing tokens for the [Q]/[D] token,
            # we add those as new tokens; however it is important to expand the
            # embedding matrix of the model using this tokenizer by calling:
            # `model.resize_token_embeddings(len(tokenizer))`
            self.tok.add_tokens(
                [self.config.query_token, self.config.doc_token],
                special_tokens=True
            )

            self.Q_marker_token = self.config.query_token
            self.Q_marker_token_id = self.tok.convert_tokens_to_ids(
                self.config.query_token
            )
            self.D_marker_token = self.config.doc_token
            self.D_marker_token_id = self.tok.convert_tokens_to_ids(
                self.config.doc_token
            )

        self.cls_token, self.cls_token_id = self.tok.cls_token, self.tok.cls_token_id
        self.sep_token, self.sep_token_id = self.tok.sep_token, self.tok.sep_token_id
        self.mask_token, self.mask_token_id = self.tok.mask_token, self.tok.mask_token_id
        self.pad_token, self.pad_token_id = self.tok.pad_token, self.tok.pad_token_id


if __name__ == "__main__":
    sentences = [
        "Query?",
        "1. Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.",
        "2.. Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.",
        "3.. Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.",
        "4. Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum. Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.",
    ]

    base_tokenizers = [
        "bert-base-uncased",
        "roberta-base",
        # "../../../data/colbertv2.0/",
    ]
    config = BaseConfig(tok_name_or_path=base_tokenizers[0])

    tokenizer = ColBERTTokenizer(config)

    # tokenizer.save("testchen")
    # tokenizer = ColBERTTokenizer.from_pretrained("../../../checkpoints/harry_potter_bert_2023-05-31T15:10:52/epoch1_2_loss0.1793_mrr0.9658_acc93.171/")

    # testing the bsize attribute in the tensorize methode
    batches = tokenizer.tensorize(sentences, mode="query", bsize=1)
    for b in batches:
        ids, mask = b
        print(b[0].shape, tokenizer.decode(b[0][0]))
    print()

    IDX = 1
    # single string tokenization for a query
    tokenize = tokenizer.tokenize(
        sentences[IDX], mode="query", add_special_tokens=True, truncate=True
    )
    encode = tokenizer.encode(
        sentences[IDX], mode="query", add_special_tokens=True, truncate=True
    )
    tensorize, _ = tokenizer.tensorize(sentences[IDX], mode="query")

    print("=" * 50)
    print("MODE: query")
    print("=" * 50)
    print(f"Tokenize: (len={len(tokenize)})", tokenize, sep="\n", end="\n\n")
    print(f"Encode: (len={len(encode)})", encode, sep="\n", end="\n\n")
    print(f"Tensorize: (len={len(tensorize[0])})", tensorize, sep="\n", end="\n\n")
    print("Decoded:", tokenizer.decode(tensorize[0]), end="\n\n\n")

    # batch tokenization for a document
    tokenize = tokenizer.tokenize(
        sentences, mode="doc", add_special_tokens=True, truncate=True
    )
    encode = tokenizer.encode(
        sentences, mode="doc", add_special_tokens=True, truncate=True
    )
    tensorize = tokenizer.tensorize(sentences, mode="doc")[0]

    print("=" * 50)
    print("MODE: doc")
    print("=" * 50)
    print(f"Tokenize: (len={len(tokenize[IDX])})", tokenize[IDX], sep="\n", end="\n\n")
    print(f"Encode: (len={len(encode[IDX])})", encode[IDX], sep="\n", end="\n\n")
    print(
        f"Tensorize: (len={len(tensorize[IDX])})", tensorize[IDX], sep="\n", end="\n\n"
    )
    print("Decoded:", tokenizer.decode(tensorize[IDX]), end="\n\n")
