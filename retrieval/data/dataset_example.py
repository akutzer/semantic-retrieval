from tqdm import tqdm
import numpy as np
np.random.seed(125)

from retrieval.configs import BaseConfig
from retrieval.models import ColBERTTokenizer
from retrieval.data.dataset import TripleDataset
from retrieval.data.dataloader import BucketIterator



def load_ms_marco_v1_1_QP():
    config = BaseConfig(
        bucket_size=16*4,
        batch_size=32,
        accum_steps=2,
        passages_per_query=1,
        doc_maxlen=300)

    triples_path = "../../data/ms_marco_v1.1/train/triples.train.tsv"
    queries_path = "../../data/ms_marco_v1.1/train/queries.train.tsv"
    passages_path = "../../data/ms_marco_v1.1/train/passages.train.tsv"

    dataset = TripleDataset(config, triples_path, queries_path, passages_path, mode="QPP")
    tokenizer = ColBERTTokenizer(config)
    bucket_iter = BucketIterator(config, dataset, tokenizer)
    
    return tokenizer, bucket_iter


def load_ms_marco_v2_1_QPP():
    config = BaseConfig(
        bucket_size=16*4,
        batch_size=32,
        accum_steps=2,
        passages_per_query=10,
        doc_maxlen=300)

    triples_path = "../../data/ms_marco_v2.1/train/triples.train.tsv"
    queries_path = "../../data/ms_marco_v2.1/train/queries.train.tsv"
    passages_path = "../../data/ms_marco_v2.1/train/passages.train.tsv"

    dataset = TripleDataset(config, triples_path, queries_path, passages_path, mode="QPP")
    tokenizer = ColBERTTokenizer(config)
    bucket_iter = BucketIterator(config, dataset, tokenizer)
    
    return tokenizer, bucket_iter


def load_harry_potter_QQP():
    config = BaseConfig(
        bucket_size=16*4,
        batch_size=32,
        accum_steps=2,
        passages_per_query=10)

    triples_path = "../../data/fandoms_qa/harry_potter/triples.tsv"
    queries_path = "../../data/fandoms_qa/harry_potter/queries.tsv"
    passages_path = "../../data/fandoms_qa/harry_potter/passages.tsv"

    dataset = TripleDataset(config, triples_path, queries_path, passages_path, mode="QQP")
    tokenizer = ColBERTTokenizer(config)
    bucket_iter = BucketIterator(config, dataset, tokenizer)
    
    return tokenizer, bucket_iter



# tokenizer, dataloader = load_ms_marco_v1_1_QP()
tokenizer, dataloader = load_ms_marco_v2_1_QPP()
# tokenizer, dataloader = load_harry_potter_QQP()


dataloader.shuffle()
for i, bucket in enumerate(tqdm(dataloader)):
    for batch in bucket:
        Q, P = batch
        (q_tokens, q_masks), (p_tokens, p_masks) = Q, P
        if p_tokens.shape[1] >= dataloader.config.doc_maxlen:
            print(p_tokens.shape)
        # print(q_tokens.shape, p_tokens.shape)
        # if dataloader.dataset.is_qpp():
        #     print(tokenizer.decode(q_tokens[0]), tokenizer.decode(p_tokens[0]), tokenizer.decode(p_tokens[-1]), sep="\n\n")
        # else:
        #     print(tokenizer.decode(q_tokens[0]), tokenizer.decode(q_tokens[1]), tokenizer.decode(p_tokens[0]), sep="\n\n")
        # exit(0)
