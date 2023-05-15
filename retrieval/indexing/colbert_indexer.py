import faiss
import time
from tqdm import tqdm
import torch
import torch.nn.functional as F
from collections import defaultdict
from typing import Union, List

from retrieval.configs import BaseConfig
from retrieval.data import Passages
from retrieval.models import ColBERTTokenizer, ColBERTInference
from retrieval.indexing.indexer import IndexerInterface



class ColBERTIndexer(IndexerInterface):
    def __init__(self, config, device="cpu"):
        self.config = config
        self.device = device
        self.load_model()
        
        self.similarity = self.config.similarity
        self.embeddings = torch.tensor([], device=device)
        self.iid2pid = dict()
        self.pid2iid = defaultdict(list)
        self.next_iid = 0
        
    def load_model(self):
        # TODO: implement correctly
        self.tokenizer = ColBERTTokenizer(self.config)
        self.inference = ColBERTInference(self.config, self.tokenizer, device=self.device)
        
    def index(self, path_to_passages: str, bsize: int = 16, dtype=torch.float32):
        passages = Passages(path_to_passages)
        data = passages.values().tolist()
        pids = passages.keys().tolist()

        embeddings = self.inference.doc_from_text(data, bsize=bsize, show_progress=True)
        assert len(embeddings) == len(pids)
        
        # update the iid2pid and pid2iid mappings
        for pid, emb in zip(pids, embeddings):
            # adds new iids in the range [next_iid, next_iid + n_embeddings)
            start_iid, end_iid = self.next_iid, self.next_iid + emb.shape[0]
            new_iids = range(start_iid, end_iid)
            self.pid2iid[pid].extend(new_iids)
            self.iid2pid.update({iid: pid for iid in new_iids})
            self.next_iid = end_iid
        
        # concatenate the new embeddings onto the previous embedding matrix
        # TODO: this does not remove old embeddings for the same PID
        self.embeddings = torch.cat([self.embeddings, *embeddings], dim=0)
    
    def search(self, query: torch.Tensor, k: int):
        # add batch dimension if query is a single vector
        if query.dim() == 1:
            query = query[None]
        # query shape: (B * L_q, D)
        
        
        if self.similarity == "l2":
            query = query[:, None]  # shape : (B * L_q, 1, D)
            # print(query.shape, self.embeddings.shape)
            sim = -1.0 * (query - self.embeddings).pow(2).sum(dim=-1) # shape: (B * L_q, N_embs)
            # sim = -1.0 * torch.norm(query - self.embeddings, ord=2, dim=-1) # shape: (B * L_q, N_embs)

        elif self.similarity == "cosine":
            # print(query.shape, self.embeddings.shape)
            sim = query @ self.embeddings.T # shape: (B * L_q, N_embs)
        else:
            raise ValueError()
        
        topk_sim, topk_iids = sim.topk(k)
        return topk_sim, topk_iids
    
    def iids_to_pids(self, iids, bsize: int = 1):
        iids = iids.reshape(bsize, -1)
        pids = []

        for query_iids in iids:
            query_pids = []
            for iid in query_iids:
                pid = self.iid2pid[iid.item()]
                if pid not in query_pids:
                    query_pids.append(pid)
            pids.append(query_pids)

        return pids

    def get_pid_embedding(self, pids: Union[int, List[int]]):
        is_single_pid = isinstance(pids, int)
        if is_single_pid:
            pids = [pids]
            
        embs = [torch.stack([self.embeddings[iid] for iid in self.pid2iid[pid]], dim=0) for pid in pids]
        if is_single_pid:
            embs = embs[0]

        return embs

    def save(self, path):
        parameters = {
            "iid2pid": self.iid2pid,
            "pid2iid": self.pid2iid,
            "embeddings": self.embeddings.cpu(),
        }
        torch.save(parameters, path)
    
    def load(self, path):
        parameters = torch.load(path)
        self.iid2pid = parameters["iid2pid"]
        self.pid2iid = parameters["pid2iid"]
        self.embeddings = parameters["embeddings"].to(self.device)
    
    
    
    


if __name__ == "__main__":
    import random
    import numpy as np

    SEED = 125
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)



    MODEL_PATH = "../../data/colbertv2.0/" # "../../../data/colbertv2.0/" or "bert-base-uncased" or "roberta-base"

    config = BaseConfig(
        tok_name_or_path=MODEL_PATH,
        backbone_name_or_path=MODEL_PATH,
        similarity="cosine",
        dim = 128,
        batch_size = 16,
        accum_steps = 1,
    )
    PATH = "../../data/fandom-qa/witcher_qa/passages.train.tsv"
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

    INDEX_PATH = "../../data/fandom-qa/witcher_qa/passages.train.indices.pt"
    IDX = 3



    indexer = ColBERTIndexer(config, device="cuda:0")
    # since we are not sorting by length, small batch sizes seem to be more efficient,
    # because there is less padding
    # indexer.index(PATH, bsize=8)
    # indexer.save(INDEX_PATH)

    # print(indexer.pid2iid)
    # print(indexer.iid2pid[IDX], indexer.embeddings[IDX])
    # print(indexer.get_pid_embedding(indexer.iid2pid[IDX]))
    indexer.load(INDEX_PATH)
    # indexer.similarity = config.similarity

    query = "Who is the author of 'The Witcher'?" #"Who do NPCs react if it rains?" #"What is the largest island of the Skellige Islands?" # "Where can I find NPCs if it rains?" #"Who was Cynthia?"

    Q = indexer.inference.query_from_text(query)
    if Q.dim() == 2:
        Q = Q[None]
    print(Q.shape)

    B, L_q, D = Q.shape
    Q = Q.reshape(B*L_q, -1)
    sim, iids = indexer.search(Q, k=10)
    pids = indexer.iids_to_pids(iids, bsize=B)
    print(pids)

    passages = Passages(PATH)
    print(passages[2])
    # for pid in pids:
    #     print(passages[pid].values)
    
    embs = [(batch_pids, indexer.get_pid_embedding(batch_pids)) for batch_pids in pids]

    for pids, topk_embs in embs:
        sims = []
        for pid_emb in topk_embs:
            # print(Q.shape, pid_emb.shape)
            out = Q @ pid_emb.T
            sim = out.max(dim=-1).values.sum()
            sims.append(sim)
        values, indices = torch.sort(torch.tensor(sims), descending=True)
        sorted_pids = torch.tensor(pids)[indices]
        print(pids, values, indices)
        print(sorted_pids)
    
        for sim, pid in zip(values, sorted_pids[:10]):
            print(round(sim.item(), 3), pid.item(),  passages[pid.item()])
    
    # e
    # print(len(embs), len(embs[0]))
    

    # print(indexer.iid2pid[IDX], indexer.embeddings[IDX])
    # print(indexer.get_pid_embedding(indexer.iid2pid[IDX]))