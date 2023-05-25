#!/usr/bin/env python3
import math
import numpy as np

from torch.utils.data import Dataset, DataLoader
from retrieval.configs import BaseConfig
from retrieval.data.dataset import TripleDataset
from retrieval.models import ColBERTTokenizer



class TokenizedTripleDataset(Dataset):
    def __init__(self, config: BaseConfig, dataset: TripleDataset, tokenizer: ColBERTTokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        if self.dataset.is_qqp():
            *qids, pids = self.dataset[index]
            pids = [pids]
        else:
            qids, *pids = self.dataset[index]
            qids = [qids]
            pids = pids[:self.config.passages_per_query]
        
        queries =  self.dataset.qid2string(qids)
        passages = self.dataset.pid2string(pids)

        return queries, passages

    def collate_fn(self, batch):
        queries_flatten, passages_flatten = [], []
        for queries, passages in batch:
            queries_flatten.extend(queries)
            passages_flatten.extend(passages)

        Q_batches = self.tokenizer.tensorize(queries_flatten, mode="query")
        P_batches = self.tokenizer.tensorize(passages_flatten, mode="doc")

        return Q_batches, P_batches


def get_pytorch_dataloader(config: BaseConfig, dataset: TripleDataset, tokenizer: ColBERTTokenizer):
    token_dataset = TokenizedTripleDataset(config, dataset, tokenizer)

    sub_batch_size = math.ceil(config.batch_size / config.accum_steps)
    dataloader = DataLoader(
        dataset=token_dataset,
        batch_size=sub_batch_size,
        shuffle=config.shuffle,
        collate_fn=token_dataset.collate_fn,
        pin_memory=config.pin_memory,
        drop_last=config.drop_last,
        num_workers=config.num_workers
    )

    return dataloader



if __name__ == "__main__":
    from tqdm import tqdm
    import torch

    SEED = 125
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    config = BaseConfig(
        batch_size=32,
        accum_steps=2,
        passages_per_query=10,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        num_workers=4
    )

    triples_path = "../../data/ms_marco_v2.1/train/triples.train.tsv"
    queries_path = "../../data/ms_marco_v2.1/train/queries.train.tsv"
    passages_path = "../../data/ms_marco_v2.1/train/passages.train.tsv"
    dataset = TripleDataset(config, triples_path, queries_path, passages_path, mode="QPP")

    tokenizer = ColBERTTokenizer(config)
    dataloader = get_pytorch_dataloader(config, dataset, tokenizer)

    for i, batch in enumerate(tqdm(dataloader)):
        Q, P = batch
        (q_tokens, q_masks), (p_tokens, p_masks) = Q, P
        # q_tokens, q_masks, p_tokens, p_masks = q_tokens.to("cuda:0", non_blocking=True), q_masks.to("cuda:0", non_blocking=True), p_tokens.to("cuda:0", non_blocking=True), p_masks.to("cuda:0", non_blocking=True)

        # print(q_tokens.shape, q_masks.shape, p_tokens.shape, p_masks.shape)
        # print(q_tokens.is_pinned())
        # print(q_tokens[0], p_tokens[0])
        # print(tokenizer.decode(q_tokens[0]))
        # print(tokenizer.decode(q_tokens[1]))
        # print(tokenizer.decode(p_tokens[0]))
        # print(tokenizer.decode(p_tokens[1]))
        # exit(0)


