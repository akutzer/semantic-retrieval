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
        self.embeddings = torch.tensor([], device=self.device)
        self.iid2pid = torch.empty(0, device=self.device, dtype=torch.int64)
        self.pid2iid = torch.empty((0, 0), device=self.device, dtype=torch.int64)
        self.next_iid = 0
        
    def load_model(self):
        # TODO: implement correctly
        self.tokenizer = ColBERTTokenizer(self.config)
        self.inference = ColBERTInference(self.config, self.tokenizer, device=self.device)
    
    def _new_passages(self, passages: List[str], pids: List[str]):
        passages_, pids_ = [], []

        # if the indexer is uninitialized use all passages
        if self.pid2iid.shape == torch.Size([0, 0]):
            passages_, pids_ = passages, pids

        # else use only passages from unseen pids
        else:
            for pid, passage in zip(pids, passages):
                if pid > self.pid2iid.shape[0] - 1 or self.pid2iid[pid][0] == -1:
                    passages_.append(passage)
                    pids_.append(pid)

        return passages_, pids_

    
    def index(self, passages: List[str], pids: List[str], bsize: int = 16, dtype=torch.float32):
        
        passages, pids = self._new_passages(passages, pids)
        # if there are no passages which haven't been indexed yet, return
        if len(passages) == 0:
            return

        with torch.inference_mode():
            embeddings = self.inference.doc_from_text(passages, bsize=bsize, show_progress=True)
            assert len(embeddings) == len(pids)

            # calculate the new width of the pid2iid matrix
            # (aka the maximal number of IIDs which are assigned to a single PID)
            max_iids_per_pid = max(max(emb.shape[0] for emb in embeddings), self.pid2iid.shape[-1])

            # calculate the new height of the pid2iid matrix (aka the new maximal PIDs)
            max_pid = max(max(pids) + 1, self.pid2iid.shape[0])

            # extends the pid2iid matrix (padded with -1)
            pid2iid = torch.full((max_pid, max_iids_per_pid), -1, dtype=torch.int64, device=self.device)
            pid2iid[:self.pid2iid.shape[0], :self.pid2iid.shape[1]] = self.pid2iid

            # extends the iid2pid vector
            num_new_iids = sum(emb.shape[0] for emb in embeddings)
            iid2pid = torch.empty((self.iid2pid.shape[0] + num_new_iids,), dtype=torch.int64, device=self.device)
            iid2pid[:self.iid2pid.shape[0]] = self.iid2pid
            
            # update the pid2iid matrix and iid2pid vector for the new embeddings
            next_iid_ = self.iid2pid.shape[0]
            for i, (pid, emb) in enumerate(zip(pids, embeddings)):
                # adds new iids in the range [next_iid_, next_iid_ + n_embeddings)
                new_iids = torch.arange(next_iid_, next_iid_ + emb.shape[0], device=self.device)
                pid2iid[pid, :emb.shape[0]] = new_iids
                iid2pid[new_iids] = pid
                next_iid_ += emb.shape[0]
            
            # Update the mappings
            self.pid2iid = pid2iid
            self.iid2pid = iid2pid
            
            # Concatenate the new embeddings onto the previous embedding matrix
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
        # add batch dimension
        if iids.dim() == 2:
            iids = iids[None]
        # iids shape: (B, L_q, k)

        B = iids.shape[0]
        iids = iids.reshape(B, -1)
        pids = []

        for batch_pids in self.iid2pid[iids]:
            pids.append(torch.unique(batch_pids))

        return pids
    
    def get_pid_embedding(self, batch_pids: List[torch.IntTensor], pad=False):
        # is_single_pid = isinstance(pids, int)
        # if is_single_pid:
        #     pids = [pids]
        
        # print(pids)
        # print(self.pid2iid[pids])
        batch_embs, batch_masks = [], []
        for pids in batch_pids:
            iids = self.pid2iid[pids]
            # print(iids)
            embs = self.embeddings[iids]
            mask = iids != -1
            embs[~mask] = 0
            # emb_pids = self.pid2iid[pids]
            # mask = emb_pids != -1
            # print(embs, mask)
            batch_embs.append(embs)
            batch_masks.append(mask)
        
        return batch_embs, batch_masks
        # # exit(0)
        # iids = [self.pid2iid[pid] for pid in pids]
        # max_iids = max(len(iid_list) for iid_list in iids)
        # embs = torch.zeros((len(pids), max_iids, self.embeddings.shape[-1]), device=self.device)  # Initialize tensor to store embeddings
        # for i, iid_list in enumerate(iids):
        #     embs[i, :len(iid_list)] = self.embeddings[iid_list]
        
        # if pad:
        #     mask = torch.arange(max_iids, device=self.device)[None, :] < torch.tensor([len(iid_list) for iid_list in iids], device=self.device)[:, None]
        #     embs = embs[:, :mask.sum(dim=1).max()]  # Trim the tensor to the maximum sequence length

        # # if is_single_pid:
        # #     embs = embs[0]
        # #     if pad:
        # #         mask = mask[0]
        
        # print(embs, mask)
        # exit(0)

        # return embs, mask if pad else embs

    def save(self, path):
        parameters = {
            "iid2pid": self.iid2pid.cpu(),
            "pid2iid": self.pid2iid.cpu(),
            "embeddings": self.embeddings.cpu(),
        }
        torch.save(parameters, path)
    
    def load(self, path):
        parameters = torch.load(path)
        self.iid2pid = parameters["iid2pid"].to(self.device)
        self.pid2iid = parameters["pid2iid"].to(self.device)
        self.embeddings = parameters["embeddings"].to(self.device)
    

    # def iids_to_pids_and_embedding(self, iids: torch.IntTensor, pad=False):
    #     # iids shape: (B, L_q, k)
    #     B, L_q, k = iids.shape
    #     iids = iids.reshape(B * L_q * k)

    #     unique_iids, inverse_indices = torch.unique(iids, return_inverse=True)

    #     unique_pids = []
    #     for iid in unique_iids:
    #         if iid in self.iid2pid:
    #             unique_pids.append(self.iid2pid[iid])
    #     unique_pids = torch.unique(torch.tensor(unique_pids, device=self.device))

    #     pid_to_index = {pid: index for index, pid in enumerate(unique_pids)}
    #     pids = []
    #     for query_iids in iids.reshape(B, L_q, k):
    #         query_pids = [pid_to_index[self.iid2pid[iid]] if iid in self.iid2pid else None for iid in query_iids]
    #         pids.append(query_pids)

    #     embs = self.embeddings[unique_iids]

    #     if pad:
    #         max_iids = max(len(iid_list) for iid_list in pids)
    #         mask = torch.arange(max_iids, device=self.device)[None, :] < torch.tensor([len(iid_list) for iid_list in pids], device=self.device)[:, None]
    #         embs = embs[torch.unique(inverse_indices)]
    #         embs = embs[:, :mask.sum(dim=1).max()]

    #         pids_padded = []
    #         for query_pids in pids:
    #             query_pids_padded = query_pids + [None] * (max_iids - len(query_pids))
    #             pids_padded.append(query_pids_padded)

    #         pids = pids_padded

    #     return embs.reshape(B, L_q, -1), pids if pad else embs.reshape(B, L_q, -1)





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

    passages = Passages(PATH)
    data = passages.values().tolist()
    pids = passages.keys().tolist()
    # since we are not sorting by length, small batch sizes seem to be more efficient,
    # because there is less padding
    indexer.index(data[:1], pids[:1], bsize=8)
    # print(indexer.embeddings[:10])
    # print(indexer.iid2pid)
    # print(indexer.pid2iid)
    indexer.index(data[1:2], pids[1:2], bsize=8)
    # print(indexer.embeddings[:10])
    # print(indexer.iid2pid)
    # print(indexer.pid2iid)
    indexer.index(data[:3], pids[:3], bsize=8)
    # print(indexer.embeddings[:10])
    # print(indexer.iid2pid)
    # print(indexer.pid2iid)
    # exit(0)

    # indexer.index(data, pids, bsize=8)
    # indexer.save(INDEX_PATH)

    
    indexer.load(INDEX_PATH)

    queries = ["Who is the author of 'The Witcher'?", "How does an NPC react if it starts raining?"] #"What is the largest island of the Skellige Islands?" # "Where can I find NPCs if it rains?" #"Who was Cynthia?"
    # queries = queries[0]

    Qs = indexer.inference.query_from_text(queries)
    if Qs.dim() == 2:
        Qs = Qs[None]
    # print(Qs.shape)
    
    batch_sim, batch_iids = indexer.search(Qs, k=10)
    print(batch_sim.shape, batch_iids.shape)
    batch_pids = indexer.iids_to_pids(batch_iids)
    # print(batch_pids)

    batch_embs, batch_masks = indexer.get_pid_embedding(batch_pids)
    # print(batch_embs, batch_masks)
    

    for Q, pids, topk_embs, mask in zip(Qs, batch_pids, batch_embs, batch_masks):
        print(Q.shape, pids.shape, topk_embs.shape, mask.shape)

        # topk_embs @ Q.mT instead of Q @ topk_embs.mT because of the masking later on
        sim = topk_embs @ Q.mT # (N_doc, L_d, L_q)

        # replace the similarity results for padding vectors
        sim[~mask] = -torch.inf

        # calculate the sum of max similarities
        sms = sim.max(dim=1).values.sum(dim=-1)
        print(sim.shape, sms.shape)

        values, indices = torch.sort(sms, descending=True)
        sorted_pids = pids[indices]
        # print(pids, values, indices)
        # print(values, sorted_pids)

        print("="*50 + "  Best Results  " + "="*50)
        for sim, pid in zip(values, sorted_pids[:10]):
            print(round(sim.item(), 3), pid.item(),  passages[pid.item()])
        print(end="\n\n\n")

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