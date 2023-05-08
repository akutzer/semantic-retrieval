#!/usr/bin/env python3
from transformers import AutoTokenizer
from retrieval.configs import BaseConfig
from retrieval.models.colbert.utils import _split_into_batches


class ColBERTTokenizer():
    def __init__(self, config: BaseConfig):
        self.config = config
        self.tok = AutoTokenizer.from_pretrained(config.tok_name_or_path)    

        self.query_maxlen = self.config.query_maxlen
        self.doc_maxlen = self.config.doc_maxlen

        self.tok.add_tokens([self.config.query_token, self.config.doc_token], special_tokens=True)

        self.Q_marker_token, self.Q_marker_token_id = self.config.query_token, self.tok.convert_tokens_to_ids(self.config.query_token)
        self.D_marker_token, self.D_marker_token_id = self.config.doc_token, self.tok.convert_tokens_to_ids(self.config.doc_token)

        self.cls_token, self.cls_token_id = self.tok.cls_token, self.tok.cls_token_id
        self.sep_token, self.sep_token_id = self.tok.sep_token, self.tok.sep_token_id
        self.mask_token, self.mask_token_id = self.tok.mask_token, self.tok.mask_token_id
        self.pad_token,self.pad_token_id = self.tok.pad_token,self.tok.pad_token_id

    def tokenize(self, batch_text, mode, add_special_tokens=False, truncate=False):
        assert type(batch_text) in [list, tuple], (type(batch_text))
        assert isinstance(mode, str) and mode in ["query", "doc"]

        tokens = [self.tok.tokenize(" " + seq, add_special_tokens=add_special_tokens) for seq in batch_text]

        if truncate:
            maxlen = self.query_maxlen if mode == "query" else self.doc_maxlen
            tokens = [list(tok_seq[:maxlen - 3]) for tok_seq in tokens]

        if not add_special_tokens:
            return tokens

        if mode == "query":
            prefix, suffix = [self.cls_token, self.Q_marker_token], [self.sep_token]
            tokens = [prefix + tok_seq + suffix + [self.mask_token] * (self.query_maxlen - (len(tok_seq) + 3)) for tok_seq in tokens]

        else:
            prefix, suffix = [self.cls_token, self.D_marker_token], [self.sep_token]
            tokens = [prefix + tok_seq + suffix for tok_seq in tokens]

        return tokens
    
    def encode(self, batch_text, mode="query", add_special_tokens=False, truncate=False):
        #assert type(batch_text) in [list, tuple], (type(batch_text))
        assert isinstance(mode, str) and mode in ["query", "doc"]

        string = False
        if isinstance(batch_text, str):
            batch_text = [batch_text]
            string = True
        

        batch_text = [" " + seq for seq in batch_text]
        ids = self.tok(batch_text, add_special_tokens=add_special_tokens, return_attention_mask=False)["input_ids"]

        if truncate:
            maxlen = self.query_maxlen if mode == "query" else self.doc_maxlen
            ids = [list(id_seq[:maxlen - 3]) for id_seq in ids]

        if not add_special_tokens:
            if string:
                return ids[0]

            return ids
        
        if mode == "query":
            prefix, suffix = [self.cls_token_id, self.Q_marker_token_id], [self.sep_token_id]
            ids = [prefix + id_seq + suffix + [self.mask_token_id] * (self.query_maxlen - (len(id_seq) + 3)) for id_seq in ids]

        else:
            prefix, suffix = [self.cls_token_id, self.D_marker_token_id], [self.sep_token_id]
            ids = [prefix + id_seq + suffix for id_seq in ids]

        if string:
            return ids[0]

        return ids

    def tensorize(self, batch_text, mode, bsize=None, return_tensors="pt"):
        assert type(batch_text) in [list, tuple], (type(batch_text))
        assert isinstance(mode, str) and mode in ["query", "doc"]

        if mode == "query":
            marker_id = self.Q_marker_token_id
            maxlen = self.query_maxlen
            padding = "max_length"
        
        else:
            marker_id = self.D_marker_token_id
            maxlen = self.doc_maxlen
            padding = "longest"

        # add placehold for the [Q]/[D] marker
        batch_text = [". " + seq for seq in batch_text]

        batch_encoding = self.tok(batch_text, padding=padding, truncation=True,
                       return_tensors=return_tensors, max_length=maxlen
                       )

        ids, mask = batch_encoding["input_ids"], batch_encoding["attention_mask"]

        # postprocess for the [Q]/[D] marker and the [MASK] augmentation
        ids[:, 1] = marker_id
        if mode == "query":
            ids[ids == self.pad_token_id] = self.mask_token_id

        if self.config.ignore_mask_tokens:
            mask[ids == self.mask_token_id] = 1
            assert mask.sum().item() == mask.size(0) * mask.size(1), mask
            
        if bsize:
            return _split_into_batches(ids, mask, bsize)

        return ids, mask
    
    def decode(self, batch_ids, **kwargs):
        return self.tok.decode(batch_ids, **kwargs)
    
    def __len__(self):
        return len(self.tok)


if __name__ == "__main__":

    sentences = [
        "1. Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.",
        "2.. Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.", 
        "3.. Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.",
        "4. Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum. Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.",
    ]

    base_tokenizers = ["bert-base-uncased", "roberta-base", "../../../data/colbertv2.0/"]
    config = BaseConfig(
        tok_name_or_path=base_tokenizers[2]
    )

    tokenizer = ColBERTTokenizer(config)
    IDX = 0

    batches = tokenizer.tensorize(sentences, mode="doc", bsize=2)
    for b in batches:
        ids, mask = b
        print(b)
        print(tokenizer.decode(b[0][0]))
    # exit(0)
    
    print("="*50)
    print("MODE: query")
    print("="*50)
    tokenize = tokenizer.tokenize(sentences, mode="query", add_special_tokens=True, truncate=True)
    # encode = tokenizer.encode(sentences, mode="query", add_special_tokens=True, truncate=True)
    tensorize = tokenizer.tensorize(sentences, mode="query")[0]

    print(f"Tokenize: (len={len(tokenize[IDX])})", tokenize[IDX], sep="\n", end="\n\n")
    # print(f"Encode: (len={len(encode[IDX])})", encode[IDX], sep="\n", end="\n\n")
    print(f"Tensorize: (len={len(tensorize[IDX])})", tensorize[IDX], sep="\n", end="\n\n")
    print("Decode:", tokenizer.decode(tensorize[IDX]), end="\n\n\n")

    print("="*50)
    print("MODE: doc")
    print("="*50)
    tokenize = tokenizer.tokenize(sentences, mode="doc", add_special_tokens=True, truncate=True)
    # encode = tokenizer.encode(sentences, mode="doc", add_special_tokens=True, truncate=True)
    tensorize = tokenizer.tensorize(sentences, mode="doc")[0]

    print(f"Tokenize: (len={len(tokenize[IDX])})", tokenize[IDX], sep="\n", end="\n\n")
    # print(f"Encode: (len={len(encode[IDX])})", encode[IDX], sep="\n", end="\n\n")
    print(f"Tensorize: (len={len(tensorize[IDX])})", tensorize[IDX], sep="\n", end="\n\n")
    print("Decode:", tokenizer.decode(tensorize[IDX]), end="\n\n")
