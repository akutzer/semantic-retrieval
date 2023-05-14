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
    
    # TODO: upadate to also use list of string for path_to_passages
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
        # add batch dimension
        if query.dim() == 2:
            query = query[None]
        # query shape: (B, L_q, D)

        # TODO: use similarity from model config
        if self.similarity == "l2":
            sim = -1.0 * (query - self.embeddings).pow(2).sum(dim=-1) # shape: (B * L_q, N_embs)
            # sim = -1.0 * torch.norm(query - self.embeddings, ord=2, dim=-1) # shape: (B * L_q, N_embs)

        elif self.similarity == "cosine":
            sim = query @ self.embeddings.mT # shape: (B, L_q, N_embs)
        else:
            raise ValueError()
        
        topk_sim, topk_iids = sim.topk(k, dim=-1) # both shapes: (B, L_q, k)
        return topk_sim, topk_iids
    

    
    def iids_to_pids(self, iids: torch.IntTensor):
        # iids shape: (B, L_q, k)
        B = iids.shape[0]
        iids = iids.reshape(B, -1)
        pids = []

        for query_iids in iids.tolist():
            query_pids = list(set(self.iid2pid[iid] for iid in query_iids))
            pids.append(query_pids)

        return pids
    
    def get_pid_embedding(self, pids, pad=False):

        is_single_pid = isinstance(pids, int)
        if is_single_pid:
            pids = [pids]
        
        iids = [self.pid2iid[pid] for pid in pids]
        max_iids = max(len(iid_list) for iid_list in iids)
        embs = torch.zeros((len(pids), max_iids, self.embeddings.shape[-1]), device=self.device)  # Initialize tensor to store embeddings
        for i, iid_list in enumerate(iids):
            embs[i, :len(iid_list)] = self.embeddings[iid_list]
        
        if pad:
            mask = torch.arange(max_iids, device=self.device)[None, :] < torch.tensor([len(iid_list) for iid_list in iids], device=self.device)[:, None]
            embs = embs[:, :mask.sum(dim=1).max()]  # Trim the tensor to the maximum sequence length

        if is_single_pid:
            embs = embs[0]
            if pad:
                mask = mask[0]
        
        return embs, mask if pad else embs

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
    # print(Q.shape)

    sim, iids = indexer.search(Q, k=10)
    # print(sim.shape, iids.shape)
    pids = indexer.iids_to_pids(iids)
    # print(pids)

    passages = Passages(PATH)
    # print(passages[2])
    # for pid in pids:
    #     print(passages[pid].values)
    
    embs = [(query_best_pids, *indexer.get_pid_embedding(query_best_pids, pad=True)) for query_best_pids in pids]
    print(len(embs), embs[0][1].shape, embs[0][2].shape)
    # exit(0)

    for q, (pids, topk_embs, mask) in zip(Q, embs):
        print(Q.shape, topk_embs.shape, mask.shape)
        # topk_embs @ Q.mT instead of Q @ topk_embs.mT because of the masking later on
        sim = topk_embs @ Q.mT # (N_doc, L_d, L_q)

        # replace the similarity results for padding vectors
        sim[~mask] = -torch.inf

        # calculate the sum of max similarities
        sms = sim.max(dim=1).values.sum(dim=-1)
        print(sim.shape, sms.shape)

        values, indices = torch.sort(sms, descending=True)
        sorted_pids = torch.tensor(pids, device=indices.device)[indices]
        # print(pids, values, indices)
        # print(sorted_pids)

        for sim, pid in zip(values, sorted_pids[:10]):
            print(round(sim.item(), 3), pid.item(),  passages[pid.item()])
        

        # sims = []
        # for pid_emb in topk_embs:
        #     # print(Q.shape, pid_emb.shape)
        #     out = Q @ pid_emb.T
        #     sim = out.max(dim=-1).values.sum()
        #     sims.append(sim)
        # values, indices = torch.sort(torch.tensor(sims), descending=True)
        # sorted_pids = torch.tensor(pids)[indices]
        # print(pids, values, indices)
        # print(sorted_pids)
    
        # for sim, pid in zip(values, sorted_pids[:10]):
        #     print(round(sim.item(), 3), pid.item(),  passages[pid.item()])
    
    # e
    # print(len(embs), len(embs[0]))
    

    # print(indexer.iid2pid[IDX], indexer.embeddings[IDX])
    # print(indexer.get_pid_embedding(indexer.iid2pid[IDX]))