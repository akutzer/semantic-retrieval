#!/usr/bin/env python3
import math
import numpy as np

from retrieval.configs import BaseConfig
from retrieval.data.dataset import TripleDataset
from retrieval.models import ColBERTTokenizer




class BucketIterator():
    def __init__(self, config: BaseConfig, dataset: TripleDataset, tokenizer: ColBERTTokenizer):
        self.config = config
        self.bucket_size = config.bucket_size
        self.batch_size = config.batch_size
        assert self.bucket_size >= self.batch_size, "Bucket can't be smaller than the batch size"
        assert self.bucket_size % self.batch_size == 0, "Bucket must be a multiple of the batch size"

        self.tokenizer = tokenizer
        self.dataset = dataset

        self.drop_last = config.drop_last
        self.pin_memory = config.pin_memory
        self.position = 0
        self.index_order = np.arange(0, len(self.dataset))
        self.reset()

    def __iter__(self):
        return self

    def __len__(self):
        length = math.ceil(len(self.dataset) / self.bucket_size)
        if self.drop_last:
            # if the last bucket is smaller than the batch size it will be dropped
            last_bucket_size = len(self.dataset) % self.bucket_size
            if last_bucket_size < self.batch_size:
                length -= 1
        return length

    def __next__(self):
        offset, endpos = self.position, min(self.position + self.bucket_size, len(self.dataset))
        self.position = endpos

        # drops the last incomplete batch
        if self.drop_last and offset + self.bucket_size > len(self.dataset):
            remain = len(self.dataset) - offset
            # if the remaining datapoints are not enough to fill a single batch
            # stop the iterator
            if remain < self.batch_size:
                raise StopIteration
            else:
                endpos -= remain % self.batch_size

        if offset >= len(self.dataset):
            raise StopIteration
        
        qry_batch, psg_batch = [], []
        # for idx in range(offset, endpos):
        for idx in self.index_order[offset:endpos]:
            if self.dataset.is_qqp():
                *qids,  pids = self.dataset[idx]
                pids = [pids]
            else:   # mode == "qpp"
                qids, *pids = self.dataset[idx]
                qids = [qids]
                pids = pids[:self.config.passages_per_query]
            
            qry_batch.extend(self.dataset.qid2string(qids))
            psg_batch.extend(self.dataset.pid2string(pids))
        
        return self.collate_fn(qry_batch, psg_batch)
    
    def collate_fn(self, queries, passages):
        subbatch_size = math.ceil(self.batch_size / self.config.accum_steps)

        if self.dataset.is_qqp():
            q_batch_size = subbatch_size * 2
            p_batch_size = subbatch_size
        else:   # mode == "qpp"
            q_batch_size = subbatch_size
            p_batch_size = subbatch_size * self.config.passages_per_query

        # if self.drop_last:
        #     if self.dataset.is_qqp() and passages % p_batch_size !=:

        #     print(len(queries), len(passages), q_batch_size, p_batch_size)
        #     exit(0)

        Q_batches = self.tokenizer.tensorize(queries, mode="query", bsize=q_batch_size, pin_memory=self.pin_memory)
        P_batches = self.tokenizer.tensorize(passages, mode="doc", bsize=p_batch_size, pin_memory=self.pin_memory)

        #print(Q_batches[0].shape, Q_batches[1].shape, P_batches[0].shape, P_batches[1].shape)

        return zip(Q_batches, P_batches)
    
    def collate_fn_sort(self, queries, passages):
        raise DeprecationWarning

        size = len(queries)

        # tokenize
        q_tokens, q_masks = self.tokenizer.tensorize(queries, mode="query")
        p_tokens, p_masks = self.tokenizer.tensorize(passages, mode="doc")

        # sort by paragraph length
        sorted_indices = p_masks.sum(dim=-1).sort(descending=True).indices
        q_tokens, q_masks = q_tokens[sorted_indices], q_masks[sorted_indices]
        p_tokens, p_masks = p_tokens[sorted_indices], p_masks[sorted_indices]

        # split into sub-batches, while also removing unnecessary padding
        batch_p_maxlen = p_masks[::self.batch_size].sum(dim=-1)
        q_tokens = [q_tokens[i:i+self.batch_size] for i in range(0, size, self.batch_size)]
        q_masks = [q_masks[i:i+self.batch_size] for i in range(0, size, self.batch_size)]
        p_tokens = [p_tokens[i:i+self.batch_size, :batch_p_maxlen[i//self.batch_size]] for i in range(0, size, self.batch_size)]
        p_masks = [p_masks[i:i+self.batch_size, :batch_p_maxlen[i//self.batch_size]] for i in range(0, size, self.batch_size)]

        return zip(q_tokens, q_masks, p_tokens, p_masks)

    def shuffle(self, reset_index=False):
        np.random.shuffle(self.index_order)
        # self.dataset.shuffle(reset_index=False)
    
    def reset(self):
        self.position = 0
        if self.config.shuffle:
            self.shuffle()


if __name__ == "__main__":
    from tqdm import tqdm

    np.random.seed(125)

    config = BaseConfig(
        bucket_size=32,
        batch_size=32,
        accum_steps=2,
        passages_per_query=10,
        drop_last=True,
        pin_memory=True
    )

    triples_path = "../../data/ms_marco_v2.1/train/triples.train.tsv"
    queries_path = "../../data/ms_marco_v2.1/train/queries.train.tsv"
    passages_path = "../../data/ms_marco_v2.1/train/passages.train.tsv"

    dataset = TripleDataset(config, triples_path, queries_path, passages_path, mode="QPP")
    tokenizer = ColBERTTokenizer(config)
    data_iter = BucketIterator(config, dataset, tokenizer)
    data_iter.shuffle()

    for i, bucket in enumerate(tqdm(data_iter)):
        for batch in bucket:
            Q, P = batch
            (q_tokens, q_masks), (p_tokens, p_masks) = Q, P

            # q_tokens, q_masks, p_tokens, p_masks = q_tokens.to("cuda:0", non_blocking=True), q_masks.to("cuda:0", non_blocking=True), p_tokens.to("cuda:0", non_blocking=True), p_masks.to("cuda:0", non_blocking=True)

            # print(q_tokens.shape, q_masks.shape, p_tokens.shape, p_masks.shape)
            # print(q_tokens.is_pinned())
            # print(q_tokens[0], p_tokens[0])
            # print(data_iter.tokenizer.decode(q_tokens[0]))
            # print(data_iter.tokenizer.decode(p_tokens[0]))
            # exit(0)
