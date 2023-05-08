#!/usr/bin/env python3
import math

from retrieval.configs import BaseConfig
from retrieval.data.dataset import TripleDataset
from retrieval.models import ColBERTTokenizer




class BucketIterator():
    def __init__(self, config: BaseConfig, dataset: TripleDataset):
        self.config = config
        self.bucket_size = config.bucket_size
        self.batch_size = config.batch_size

        self.tokenizer = ColBERTTokenizer(config)
        self.dataset = dataset

        self.drop_last = config.drop_last
        self.position = 0

    def __iter__(self):
        return self

    def __len__(self):
        if self.drop_last:
            math.floor(len(self.dataset) / self.bucket_size)
        return math.ceil(len(self.dataset) / self.bucket_size)

    def __next__(self):
        offset, endpos = self.position, min(self.position + self.bucket_size, len(self.dataset))
        self.position = endpos

        # drops the last incomplete batch
        if self.drop_last and offset + self.bucket_size > len(self.dataset):
            raise StopIteration
        
        if offset >= len(self.dataset):
            raise StopIteration
        
        qry_batch, psg_batch = [], []
        for i in range(offset, endpos):
            if self.dataset.is_qqp():
                *qids,  pids = self.dataset[i]
                pids = [pids]
            else:   # mode == "qpp"
                qids, *pids = self.dataset[i]
                qids = [qids]
                pids = pids[:self.config.passages_per_query]
            
            qry_batch.extend(self.dataset.qid2string(qids))
            psg_batch.extend(self.dataset.pid2string(pids))
        
        return self.collate_fn(qry_batch, psg_batch)
    
    def collate_fn(self, queries, passages):
        size = len(queries)

        if self.dataset.is_qqp():
            q_batch_size = 2 * self.batch_size
            p_batch_size = self.batch_size
        else:   # mode == "qpp"
            q_batch_size = self.batch_size
            p_batch_size = self.batch_size * self.config.passages_per_query

        Q_batches = self.tokenizer.tensorize(queries, mode="query", bsize=q_batch_size)
        P_batches = self.tokenizer.tensorize(passages, mode="doc", bsize=p_batch_size)

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
        # TODO: don't shuffle dataset, shuffle the iterator indicies
        self.dataset.shuffle(reset_index=False)
    
    def reset(self):
        self.position = 0



if __name__ == "__main__":
    from tqdm import tqdm

    config = BaseConfig(batch_size=16, bucket_size=16*4, passages_per_query=1)
    triples_path = "../../data/fandom-qa/witcher_qa/triples.train.tsv"
    queries_path = "../../data/fandom-qa/witcher_qa/queries.train.tsv"
    passages_path = "../../data/fandom-qa/witcher_qa/passages.train.tsv"

    dataset = TripleDataset(config, triples_path, queries_path, passages_path, mode="QPP")
    data_iter = BucketIterator(config, dataset)

    data_iter.shuffle()
    for i, bucket in enumerate(tqdm(data_iter)):
        for batch in bucket:
            Q, P = batch
            (q_tokens, q_masks), (p_tokens, p_masks) = Q, P
            
            # print(Q[0][0], P[0][0])
            # print(data_iter.tokenizer.decode(Q[0][0]))
            # print(data_iter.tokenizer.decode(P[0][0]))
            # exit(0)
