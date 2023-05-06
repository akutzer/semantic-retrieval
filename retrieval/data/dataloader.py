#!/usr/bin/env python3
import math

from retrieval.configs import BaseConfig
from retrieval.data.dataset import TripleDataset
from retrieval.models import ColBERTTokenizer



class DataIterator():
    def __init__(self, config: BaseConfig, dataset: TripleDataset):
        self.config = config
        self.batch_size = config.batch_size
        self.accum_steps = config.accum_steps
        self.psgs_per_qry = config.passages_per_query
        self.drop_last = config.drop_last

        self.tokenizer = ColBERTTokenizer(config)
        self.dataset = dataset

        self.position = 0

    def __iter__(self):
        return self

    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size)

    def __next__(self):
        offset, endpos = self.position, min(self.position + self.batch_size, len(self.dataset))
        self.position = endpos

        # drops the last incomplete batch
        if self.drop_last and offset + self.batch_size > len(self.dataset):
            raise StopIteration
        
        if offset >= len(self.dataset):
            raise StopIteration
        
        qry_batch, psg_batch = [], []
        for i in range(offset, endpos):
            qid, *pids = self.dataset[i]
            qry_batch.append(self.dataset.qid2string(qid))
            psg_batch.extend(self.dataset.pid2string(pids))
        
        return self.collate_fn(qry_batch, psg_batch)
    
    def collate_fn(self, queries, passages):
        size = len(queries)

        assert self.accum_steps > 0
        subbatch_size = math.ceil(self.batch_size / self.accum_steps)

        # tokenize
        q_tokens, q_masks = self.tokenizer.tensorize(queries, mode="query") # self.qry_tokenizer.tensorize(queries)
        p_tokens, p_masks = self.tokenizer.tensorize(passages, mode="doc") # self.doc_tokenizer.tensorize(passages)

        # sort by paragraph length
        sorted_indices = p_masks.sum(dim=-1).sort(descending=True).indices
        q_tokens, q_masks = q_tokens[sorted_indices], q_masks[sorted_indices]
        p_tokens, p_masks = p_tokens[sorted_indices], p_masks[sorted_indices] 

        # split into sub-batches, while also removing unnecessary padding
        subbatch_p_maxlen = p_masks[::subbatch_size].sum(dim=-1)
        q_tokens = [q_tokens[i:i+subbatch_size] for i in range(0, size, subbatch_size)]
        q_masks = [q_masks[i:i+subbatch_size] for i in range(0, size, subbatch_size)]
        p_tokens = [p_tokens[i:i+subbatch_size, :subbatch_p_maxlen[i//subbatch_size]] for i in range(0, size, subbatch_size)]
        p_masks = [p_masks[i:i+subbatch_size, :subbatch_p_maxlen[i//subbatch_size]] for i in range(0, size, subbatch_size)]

        return zip(q_tokens, q_masks, p_tokens, p_masks)

    def collate_fn_(self, queries, passages):
        size = len(queries)

        q_tokens, q_masks = self.tokenizer.tensorize(queries, mode="query") # self.qry_tokenizer.tensorize(queries)
        p_tokens, p_masks = self.tokenizer.tensorize(passages, mode="doc") # self.doc_tokenizer.tensorize(passages)

        assert self.accum_steps > 0
        subbatch_size = self.batch_size // self.accum_steps

        # split into sub-batches
        q_tokens = [q_tokens[i:i+subbatch_size] for i in range(0, size, subbatch_size)]
        q_masks = [q_masks[i:i+subbatch_size] for i in range(0, size, subbatch_size)]
        p_tokens = [p_tokens[i:i+subbatch_size] for i in range(0, size, subbatch_size)]
        p_masks = [p_masks[i:i+subbatch_size] for i in range(0, size, subbatch_size)]

        return zip(q_tokens, q_masks, p_tokens, p_masks)

    def shuffle(self, reset_index=False):
        self.dataset.shuffle(reset_index=False)
    
    def reset(self):
        self.position = 0
    
if __name__ == "__main__":
    from tqdm import tqdm

    config = BaseConfig(passages_per_query=1)
    triples_path = "../../data/fandom-qa/witcher_qa/triples.train.tsv"
    queries_path = "../../data/fandom-qa/witcher_qa/queries.train.tsv"
    passages_path = "../../data/fandom-qa/witcher_qa/passages.train.tsv"

    dataset = TripleDataset(config, triples_path, queries_path, passages_path, mode="QPP")
    data_iter = DataIterator(config, dataset)

    data_iter.shuffle()
    for i, batch in enumerate(tqdm(data_iter)):
        for sub_batch in batch:
            q_tokens, q_masks, p_tokens, p_masks = sub_batch
            Q, P = (q_tokens, q_masks), (p_tokens, p_masks)
            
            # print(Q[0][0], P[0][0])
            # print(data_iter.tokenizer.decode(Q[0][0]))
            # print(data_iter.tokenizer.decode(P[0][0]))
            # exit(0)
