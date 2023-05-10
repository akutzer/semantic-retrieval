import faiss
import time
from tqdm import tqdm
import torch
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
        
    def index(self, path_to_passages: str, bsize: int = 16):
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
        # add batch dimension if query is a 2-d Tensor
        if query.dim() == 2:
            query = query[None]

        # query shape: (B, L_q, D)    
        
        if self.similarity == "l2":
            query = query[:, :, None]  # shape : (B, L_q, 1, D)
            print(query.shape, self.embeddings.shape)
            sim = -1.0 * (query - self.embeddings).pow(2).sum(dim=-1)

        elif self.similarity == "cosine":
            print(query.shape, self.embeddings.shape)
            sim = query @ self.embeddings.T
        else:
            raise ValueError()
        
        k_hat = 2 * k
        B, Q, D = sim.shape
        top_iids = sim.reshape(B, -1).topk(k_hat).indices  # shape: (B, k_hat)
        top_iids %= D
        top_iids = top_iids.tolist()
        pids = [[self.iid2pid[iid] for iid in batch_iids] for batch_iids in top_iids]

        # remove duplicates in pids while mainting the order of the list
        cleaned_pids = []
        for batch_pids in pids:
            cleaned_batch_pids = []
            for pid in batch_pids:
                if pid not in cleaned_batch_pids:
                    cleaned_batch_pids.append(pid)
                if len(cleaned_batch_pids) == k:
                    break
            cleaned_pids.append(cleaned_batch_pids)

        return cleaned_pids
        # embs = get_pid_embedding(top_iids)
        # return top_iids, pids
    

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


    config = BaseConfig(
        dim = 32,
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

    # exit(0)
    # indexer.save(INDEX_PATH)
    # # print(indexer.pid2iid)
    # print(indexer.iid2pid[IDX], indexer.embeddings[IDX])
    # print(indexer.get_pid_embedding(indexer.iid2pid[IDX]))

    indexer = ColBERTIndexer(config, device="cpu")
    indexer.load(INDEX_PATH)
    indexer.similarity = "cosine"
    pids = indexer.search(torch.randn(16, 24, config.dim), k=4)
    print(pids)
    embs = [indexer.get_pid_embedding(batch_pids) for batch_pids in pids]
    print(len(embs), len(embs[0]))
    

    # print(indexer.iid2pid[IDX], indexer.embeddings[IDX])
    # print(indexer.get_pid_embedding(indexer.iid2pid[IDX]))