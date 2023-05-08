import torch
from retrieval.configs import BaseConfig
from retrieval.data.triples import Triples
from retrieval.data.queries import Queries
from retrieval.data.passages import Passages


def _is_nested_list(obj):
    if isinstance(obj, list):
        if all(isinstance(elem, list) for elem in obj) and len(obj) > 0:
            return True
        else:
            return False
    else:
        return None


class TripleDataset(torch.utils.data.Dataset):
    def __init__(self, config: BaseConfig, triples_path: str, queries_path: str, passages_path: str, mode: str):
        self.config = config
        mode = mode.lower()
        assert mode in ["qqp", "qpp"], f"Mode must be either `QQP` or `QPP`, but was given as: {mode}"
        self.mode = mode
        self.output_string = False

        self.triples = Triples(triples_path, mode=mode, psgs_per_qry=config.passages_per_query)
        self.queries = Queries(queries_path)
        self.passages = Passages(passages_path)
    
    def __getitem__(self, index):
        triple = self.triples[index]

        if self.output_string:
            triple = self.ids2strings(triple)

        return triple
    
    def __len__(self):
        return len(self.triples)
    
    def queries_items(self):
        return self.queries.items()
    
    def passages_items(self):
        return self.passages.items()    
    
    def id2string(self, triple):
        if self.mode == "qqp":
            q_pos, *q_neg, p = triple
            q_pos = self.queries[q_pos]
            q_neg = self.queries[q_neg]
            p = self.passages[p]
            triple = (q_pos, *q_neg, p)

        elif self.mode == "qpp":
            q, p_pos, *p_neg = triple
            q = self.queries[q]
            p_pos = self.passages[p_pos]
            p_neg = self.passages[p_neg]
            triple = (q, p_pos, *p_neg)
        
        return triple
    
    def ids2strings(self, triples):
        is_nested = _is_nested_list(triples)

        if is_nested:
            triples = [self.id2string(triple) for triple in triples]
        else:
            triples = self.id2string(triples)
        
        return triples
    
    def qid2string(self, qid):
        return self.queries.qid2string(qid)
    
    def pid2string(self, pid):
        return self.passages.pid2string(pid)
        
    def shuffle(self, reset_index=False):
        self.triples.shuffle(reset_index=reset_index)
    
    def output_strings(self):
        self.output_string = True
    
    def output_ids(self):
        self.output_string = False
    
    def is_qqp(self):
        return self.mode == "qpp"
    
    def is_qpp(self):
        return self.mode == "qpp"



if __name__ == "__main__":
    from tqdm import tqdm

    config = BaseConfig(passages_per_query=1)
    triples_path = "../../data/fandom-qa/witcher_qa/triples.train.tsv"
    queries_path = "../../data/fandom-qa/witcher_qa/queries.train.tsv"
    passages_path = "../../data/fandom-qa/witcher_qa/passages.train.tsv"
    dataset = TripleDataset(config, triples_path, queries_path, passages_path, mode="QPP")

    #dataset.output_strings()
    for i, triple in enumerate(tqdm(dataset)):
        qid, pid_pos, *pid_neg = triple
        query, psg_pos, *psg_neg = dataset.id2string(triple)
        # print(triple, query, psg_pos, psg_neg, sep="\n", end="\n\n")