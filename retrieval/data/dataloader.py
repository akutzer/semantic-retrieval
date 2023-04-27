import csv
import math

from retrieval.tokenization import QueryTokenizer, DocTokenizer, tensorize_triples


class DataIterator():
    def __init__(self, config, triples_path, queries_path, passages_path, drop_last=False):
        self.batch_size = config.batch_size
        self.accum_steps = config.accum_steps
        self.psgs_per_qry = config.passages_per_query
        self.drop_last = drop_last
        
        self.qry_tokenizer = QueryTokenizer(config)
        self.doc_tokenizer = DocTokenizer(config)
        #self.tensorize_triples = partial(tensorize_triples, self.query_tokenizer, self.doc_tokenizer)
        self.position = 0

        self.triples = []
        with open(triples_path, mode="r", encoding="utf-8", newline="") as triples_f:
            reader = csv.reader(triples_f, delimiter="\t")
            for triplet in reader:
                self.triples.append(list(map(int, triplet)))
        
        self.queries = {}
        with open(queries_path, mode="r", encoding="utf-8", newline="") as queries_f:
            reader = csv.reader(queries_f, delimiter="\t")
            for qid, query in reader:
                self.queries[int(qid)] = query
        
        self.passages = {}
        with open(passages_path, mode="r", encoding="utf-8", newline="") as passages_f:
            reader = csv.reader(passages_f, delimiter="\t")
            for pid, passage in reader:
                self.passages[int(pid)] = passage

        

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

        return list(zip(q_tokens, q_masks, p_tokens, p_masks))