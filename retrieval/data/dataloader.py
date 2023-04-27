import math

from retrieval.tokenization import QueryTokenizer, DocTokenizer
from retrieval.data.queries import Queries
from retrieval.data.passages import Passages
from retrieval.data.triples import Triples



class DataIterator():
    def __init__(self, config, triples_path, queries_path, passages_path):
        self.batch_size = config.batch_size
        self.accum_steps = config.accum_steps
        self.psgs_per_qry = config.passages_per_query
        self.drop_last = config.drop_last
        
        self.qry_tokenizer = QueryTokenizer(config)
        self.doc_tokenizer = DocTokenizer(config)
        self.position = 0

        self.triples = Triples(triples_path)
        self.queries = Queries(queries_path)
        self.passages = Passages(passages_path)


    def __iter__(self):
        return self

    def __len__(self):
        return math.ceil(len(self.triples) / self.batch_size)

    def __next__(self):
        offset, endpos = self.position, min(self.position + self.batch_size, len(self.triples))
        self.position = endpos

        # drops the last incomplete batch
        if self.drop_last and offset + self.batch_size > len(self.triples):
            raise StopIteration
        
        if offset >= len(self.triples):
            raise StopIteration
        
        qry_batch, psg_batch = [], []
        for i in range(offset, endpos):
            qid, *pids = self.triples[i]
            qry_batch.append(self.queries[qid])
            psg_batch.extend([self.passages[pid] for pid in pids if pid >= 0])
        
        return self.collate_fn(qry_batch, psg_batch)
    
    def collate_fn(self, queries, passages):
        size = len(queries)

        q_tokens, q_masks = self.qry_tokenizer.tensorize(queries)
        p_tokens, p_masks = self.doc_tokenizer.tensorize(passages)

        assert self.accum_steps > 0
        subbatch_size = self.batch_size // self.accum_steps

        # split into sub-batches
        q_tokens = [q_tokens[i:i+subbatch_size] for i in range(0, size, subbatch_size)]
        q_masks = [q_masks[i:i+subbatch_size] for i in range(0, size, subbatch_size)]
        p_tokens = [p_tokens[i:i+subbatch_size] for i in range(0, size, subbatch_size)]
        p_masks = [p_masks[i:i+subbatch_size] for i in range(0, size, subbatch_size)]

        return zip(q_tokens, q_masks, p_tokens, p_masks)

    def collate_fn_old(self, queries, passages):
        size = len(queries)

        q_tokens, q_masks = self.qry_tokenizer.tensorize(queries)
        p_tokens, p_masks = self.doc_tokenizer.tensorize(passages)

        assert self.accum_steps > 0
        subbatch_size = self.batch_size // self.accum_steps

        # split into sub-batches
        q_tokens = [q_tokens[i:i+subbatch_size] for i in range(0, size, subbatch_size)]
        q_masks = [q_masks[i:i+subbatch_size] for i in range(0, size, subbatch_size)]
        p_tokens = [p_tokens[i:i+subbatch_size] for i in range(0, size, subbatch_size)]
        p_masks = [p_masks[i:i+subbatch_size] for i in range(0, size, subbatch_size)]

        return list(zip(q_tokens, q_masks, p_tokens, p_masks))