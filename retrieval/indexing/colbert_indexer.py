
import time
from tqdm import tqdm
import torch
import torch.nn.functional as F
from collections import defaultdict
from typing import Union, List, Tuple

from retrieval.configs import BaseConfig
from retrieval.data import Passages
from retrieval.models import ColBERTTokenizer, ColBERTInference, get_colbert_and_tokenizer
from retrieval.indexing.indexer import IndexerInterface



class ColBERTIndexer(IndexerInterface):
    def __init__(self, inference, device="cpu"):
        self.inference = inference
        self.inference.to(device)
        self.device = device
        self.similarity = self.inference.colbert.config.similarity

        self.embeddings = torch.tensor([], device=self.device)
        self.iid2pid = torch.empty(0, device=self.device, dtype=torch.int64)
        self.pid2iid = torch.empty((0, 0), device=self.device, dtype=torch.int64)
        self.next_iid = 0
            
    def index(self, passages: List[str], pids: List[str], bsize: int = 16, dtype: torch.dtype = torch.float32) -> None:
        batch_passages, batch_pids = self._new_passages(passages, pids)
        # if there are no passages which haven't been indexed yet, return
        if len(passages) == 0:
            return

        with torch.inference_mode():
            psgs_embedded = self.inference.doc_from_text(batch_passages, bsize=bsize, show_progress=True)
            assert len(psgs_embedded) == len(batch_pids)

            # calculate the new width of the pid2iid matrix
            # (aka the maximal number of IIDs which are assigned to a single PID)
            max_iids_per_pid = max(max(emb.shape[0] for emb in psgs_embedded), self.pid2iid.shape[-1])

            # calculate the new height of the pid2iid matrix (aka the new maximal PIDs)
            max_pid = max(max(pids) + 1, self.pid2iid.shape[0])

            # extends the pid2iid matrix (padded with -1)
            pid2iid = torch.full((max_pid, max_iids_per_pid), -1, dtype=torch.int64, device=self.device)
            pid2iid[:self.pid2iid.shape[0], :self.pid2iid.shape[1]] = self.pid2iid

            # extends the iid2pid vector
            num_new_iids = sum(emb.shape[0] for emb in psgs_embedded)
            iid2pid = torch.empty((self.iid2pid.shape[0] + num_new_iids,), dtype=torch.int64, device=self.device)
            iid2pid[:self.iid2pid.shape[0]] = self.iid2pid
            
            # update the pid2iid matrix and iid2pid vector for the new embeddings
            next_iid_ = self.iid2pid.shape[0]
            for pid, emb in zip(batch_pids, psgs_embedded):
                # adds new iids in the range [next_iid_, next_iid_ + n_embeddings)
                new_iids = torch.arange(next_iid_, next_iid_ + emb.shape[0], device=self.device)
                pid2iid[pid, :emb.shape[0]] = new_iids
                iid2pid[new_iids] = pid
                next_iid_ += emb.shape[0]
            
            # Update the mappings
            self.pid2iid = pid2iid
            self.iid2pid = iid2pid
            
            # Concatenate the new embeddings onto the previous embedding matrix
            self.embeddings = torch.cat([self.embeddings, *psgs_embedded], dim=0)
    
    def search(self, query: torch.Tensor, k: int) -> Tuple[torch.IntTensor, torch.IntTensor]:
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
            
    def iids_to_pids(self, batch_iids: torch.IntTensor) -> List[torch.IntTensor]:
        # add batch dimension
        if batch_iids.dim() == 0:
            batch_iids = batch_iids[None, None]
        elif batch_iids.dim() == 1:
            batch_iids = batch_iids[None]

        batch_pids = []
        for iids in self.iid2pid[batch_iids]:
            # TODO torch.unique is major bottleneck of the retriever
            batch_pids.append(torch.unique(iids))

        return batch_pids
    
    def get_pid_embedding(self, batch_pids: List[torch.IntTensor]) -> Tuple[List[torch.Tensor], List[torch.BoolTensor]]:
        batch_embs, batch_masks = [], []
        for pids in batch_pids:
            iids = self.pid2iid[pids]
            embs = self.embeddings[iids]
            mask = iids != -1
            embs[~mask] = 0

            batch_embs.append(embs)
            batch_masks.append(mask)

        return batch_embs, batch_masks

    def save(self, path: str):
        parameters = {
            "iid2pid": self.iid2pid.cpu(),
            "pid2iid": self.pid2iid.cpu(),
            "embeddings": self.embeddings.cpu(),
        }
        torch.save(parameters, path)
    
    def load(self, path: str):
        parameters = torch.load(path)
        self.iid2pid = parameters["iid2pid"].to(self.device)
        self.pid2iid = parameters["pid2iid"].to(self.device)
        self.embeddings = parameters["embeddings"].to(self.device)
        
    def _new_passages(self, passages: List[str], pids: List[str]) -> Tuple[List[str], List[str]]:
        passages_, pids_ = [], []

        # if the indexer is uninitialized use all passages
        if self.pid2iid.numel() == 0:
            passages_, pids_ = passages, pids

        # else use only passages from unseen pids
        else:
            for pid, passage in zip(pids, passages):
                if pid > self.pid2iid.shape[0] - 1 or self.pid2iid[pid][0] == -1:
                    passages_.append(passage)
                    pids_.append(pid)

        return passages_, pids_
    


if __name__ == "__main__":
    import random
    import numpy as np

    SEED = 125
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    PASSAGES_PATH = "../../data/fandom-qa/witcher_qa/passages.train.tsv"
    INDEX_PATH = "../../data/fandom-qa/witcher_qa/passages.train.indices.pt"
    MODEL_PATH = "../../data/colbertv2.0/" # "../../../data/colbertv2.0/" or "bert-base-uncased" or "roberta-base"

    config = BaseConfig(
        tok_name_or_path=MODEL_PATH,
        backbone_name_or_path=MODEL_PATH,
        similarity="cosine",
        dim = 128,
        batch_size = 16,
        accum_steps = 1,
    )
    colbert, tokenizer = get_colbert_and_tokenizer(config)
    inference = ColBERTInference(colbert, tokenizer, device=DEVICE)
    indexer = ColBERTIndexer(inference, device=DEVICE)

    passages = Passages(PASSAGES_PATH)
    data = passages.values().tolist()
    pids = passages.keys().tolist()
    
    # test indexing of already seen data
    indexer.index(data[:1], pids[:1], bsize=8)
    indexer.index(data[1:2], pids[1:2], bsize=8)
    indexer.index(data[:3], pids[:3], bsize=8)
    print(indexer.embeddings.shape)
    print(indexer.iid2pid)
    print(indexer.pid2iid)

    # test some other methods
    test_iids = torch.arange(0, 10).reshape(5, 2).T[:, None]
    test_pids = indexer.iids_to_pids(test_iids)
    test_embs = indexer.get_pid_embedding(test_pids)

    # index the entire data
    indexer.index(data, pids, bsize=8)
    indexer.save(INDEX_PATH)
    indexer.load(INDEX_PATH)
    print(indexer.embeddings.shape)


    # test retrieval
    queries = ["Who is the author of 'The Witcher'?", "How does an NPC react if it starts raining?", "Who the hell is Cynthia?"]

    Qs = indexer.inference.query_from_text(queries)
    if Qs.dim() == 2:
        Qs = Qs[None]
    # print(Qs.shape)
    
    batch_sim, batch_iids = indexer.search(Qs, k=10)
    # print(batch_sim.shape, batch_iids.shape)
    batch_pids = indexer.iids_to_pids(batch_iids)
    # print(batch_pids)
    batch_embs, batch_masks = indexer.get_pid_embedding(batch_pids)
    # print(batch_embs, batch_masks)
    
    for i, (Q, pids, topk_embs, mask) in enumerate(zip(Qs, batch_pids, batch_embs, batch_masks)):
        # print(Q.shape, pids.shape, topk_embs.shape, mask.shape)

        # topk_embs @ Q.mT instead of Q @ topk_embs.mT because of the masking later on
        sim = topk_embs @ Q.mT # (N_doc, L_d, L_q)

        # replace the similarity results for padding vectors
        sim[~mask] = -torch.inf

        # calculate the sum of max similarities
        sms = sim.max(dim=1).values.sum(dim=-1)
        # print(sim.shape, sms.shape)

        values, indices = torch.sort(sms, descending=True)
        sorted_pids = pids[indices]
        # print(pids, values, indices)
        # print(values, sorted_pids)

        print("=" * 150)
        print(f"Query: {queries[i]}")
        print("=" * 150)
        for sim, pid in zip(values, sorted_pids[:10]):
            print(round(sim.item(), 3), pid.item(),  passages[pid.item()])
        print(end="\n\n\n")
