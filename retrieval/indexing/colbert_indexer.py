#!/usr/bin/env python3
import logging
from typing import Union, List, Tuple
import torch

from retrieval.models import ColBERTInference
from retrieval.indexing.indexer import IndexerInterface


logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")


class ColBERTIndexer(IndexerInterface):
    def __init__(self, inference: ColBERTInference, device: Union[str, torch.device]="cpu", dtype: torch.dtype = torch.float32):
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device
        self.dtype = dtype

        self.inference = inference
        self.inference.to(device)
        self.similarity = self.inference.colbert.config.similarity

        self.embeddings = torch.tensor([], device=self.device, dtype=self.dtype)
        self.iid2pid = torch.empty(0, device=self.device, dtype=torch.int32)
        self.pid2iid = torch.empty((0, 0), device=self.device, dtype=torch.int32)
        self.next_iid = 0
        self.offset = None
            
    def index(self, passages: List[str], pids: List[str], bsize: int = 16) -> None:
        batch_passages, batch_pids = self._new_passages(passages, pids)
        # if there are no passages which haven't been indexed yet, return
        if len(batch_passages) == 0:
            return

        with torch.inference_mode():
            with torch.autocast(self.device.type):
                psgs_embedded = self.inference.doc_from_text(batch_passages, bsize=bsize, show_progress=True)
            assert len(psgs_embedded) == len(batch_pids)

            # calculate the new width of the pid2iid matrix
            # (aka the maximal number of IIDs which are assigned to a single PID)
            max_iids_per_pid = max(max(emb.shape[0] for emb in psgs_embedded), self.pid2iid.shape[-1])

            # calculate the new height of the pid2iid matrix (aka the new maximal PIDs)
            max_pid = max(max(batch_pids), self.pid2iid.shape[0] + (self.offset if self.offset is not None else 0) - 1)
            if self.offset is not None:
                new_offset = min(self.offset, min(batch_pids)) 
                diff_offset = self.offset - new_offset
            else:
                new_offset = min(batch_pids)
                diff_offset = 0
            print(new_offset, diff_offset)

            # extends the pid2iid matrix (padded with -1)
            pid2iid = torch.full((max_pid + 1 - new_offset, max_iids_per_pid), -1, dtype=torch.int32, device=self.device)
            print("pid2iid", pid2iid.shape, max_pid, new_offset, diff_offset)
            print(diff_offset, self.pid2iid.shape[0])
            pid2iid[diff_offset:diff_offset + self.pid2iid.shape[0], :self.pid2iid.shape[1]] = self.pid2iid

            # extends the iid2pid vector
            num_new_iids = sum(emb.shape[0] for emb in psgs_embedded)
            iid2pid = torch.empty((self.iid2pid.shape[0] + num_new_iids,), dtype=torch.int32, device=self.device)
            iid2pid[:self.iid2pid.shape[0]] = self.iid2pid
            
            # update the pid2iid matrix and iid2pid vector for the new embeddings
            next_iid_ = self.iid2pid.shape[0]
            for pid, emb in zip(batch_pids, psgs_embedded):
                # adds new iids in the range [next_iid_, next_iid_ + n_embeddings)
                new_iids = torch.arange(next_iid_, next_iid_ + emb.shape[0], device=self.device)
                pid2iid[pid - new_offset, :emb.shape[0]] = new_iids
                iid2pid[new_iids] = pid
                next_iid_ += emb.shape[0]
            
            # Update the mappings
            self.pid2iid = pid2iid
            self.iid2pid = iid2pid
            self.offset = new_offset
            
            # Concatenate the new embeddings onto the previous embedding matrix
            # this is done on the cpu to save GPU-RAM
            cat = torch.cat([self.embeddings.cpu(), *map(lambda x: x.cpu(), psgs_embedded)], dim=0)
            del psgs_embedded
            self.embeddings = cat.to(device=self.device, dtype=self.dtype)
            # self.embeddings = torch.cat([self.embeddings, *psgs_embedded], dim=0)
    
    def search(self, query: torch.Tensor, k: int) -> Tuple[torch.IntTensor, torch.IntTensor]:
        # add batch dimension
        if query.dim() == 2:
            query = query[None]
        query = query.to(dtype=self.dtype)
        # query shape: (B, L_q, D)

        # TODO: use similarity from model config
        if self.similarity == "l2":
            sim = -1.0 * (query.unsqueeze(-2) - self.embeddings.unsqueeze(-3)).pow(2).sum(dim=-1) # shape: (B * L_q, N_embs)
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
            iids = self.pid2iid[pids - self.offset]
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
            "offset": self.offset
        }
        torch.save(parameters, path)
    
    def load(self, path: str):
        parameters = torch.load(path)
        self.iid2pid = parameters["iid2pid"].to(self.device)
        self.pid2iid = parameters["pid2iid"].to(self.device)
        self.embeddings = parameters["embeddings"].to(self.device)
        self.offset = parameters["offset"]
        self.dtype = self.embeddings.dtype
        logging.info(f"Successfully loaded the precomputed indices. Changed dtype to {self.dtype}!")
        
    def _new_passages(self, passages: List[str], pids: List[str]) -> Tuple[List[str], List[str]]:
        passages_, pids_ = [], []

        # if the indexer is uninitialized use all passages
        if self.pid2iid.numel() == 0:
            passages_, pids_ = passages, pids

        # else use only passages from unseen pids
        else:
            current_max_pid = self.pid2iid.shape[0] + self.offset - 1
            current_min_pid = 0 if self.offset is None else self.offset
            for pid, passage in zip(pids, passages):
                if pid > current_max_pid or pid < current_min_pid or self.pid2iid[pid - self.offset][0] == -1:
                    passages_.append(passage)
                    pids_.append(pid)

        return passages_, pids_
    
    def to(self, device: Union[str, torch.device]) -> None:
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device

        self.inference.to(self.device)
        self.embeddings = self.embeddings.to(self.device)
        self.iid2pid = self.iid2pid.to(self.device)
        self.pid2iid = self.pid2iid.to(self.device)

    


if __name__ == "__main__":
    import random
    import numpy as np

    from retrieval.configs import BaseConfig
    from retrieval.data import Passages
    from retrieval.models import load_colbert_and_tokenizer

    SEED = 125
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    # enable TensorFloat32 tensor cores for float32 matrix multiplication if available
    torch.set_float32_matmul_precision("high")

    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    PASSAGES_PATH = "../../data/fandoms_qa/harry_potter/all/passages.tsv" # "../../data/ms_marco/ms_marco_v1_1/val/passages.tsv"
    INDEX_PATH = "../../data/fandoms_qa/harry_potter/all/passages.indices.pt" # "../../data/ms_marco/ms_marco_v1_1/val/passages.indices.pt"
    BACKBONE = "bert-base-uncased" # "../../../data/colbertv2.0/" or "bert-base-uncased" or "roberta-base"
    CHECKPOINT_PATH = "../../saves/colbert_ms_marco_v1_1/checkpoints/epoch3_2_loss1.7869_mrr0.5846_acc41.473/" # "../../data/colbertv2.0/"
    

    # config = BaseConfig(
    #     tok_name_or_path=BACKBONE,
    #     backbone_name_or_path=BACKBONE,
    #     similarity="cosine",
    #     dim = 128,
    #     batch_size = 16,
    #     accum_steps = 1,
    #     doc_maxlen=512,
    #     checkpoint=CHECKPOINT_PATH
    # )


    # colbert, tokenizer = load_colbert_and_tokenizer(CHECKPOINT_PATH, device=DEVICE, config=config)
    colbert, tokenizer = load_colbert_and_tokenizer(CHECKPOINT_PATH, device=DEVICE)
    print(colbert.config)
    inference = ColBERTInference(colbert, tokenizer, device=DEVICE)
    indexer = ColBERTIndexer(inference, device=DEVICE, dtype=torch.float16)

    passages = Passages(PASSAGES_PATH)
    data = passages.values().tolist()
    pids = passages.keys().tolist()
    
    # test indexing of already seen data
    indexer.index(data[2:3], pids[2:3], bsize=8)
    indexer.index(data[1:2], pids[1:2], bsize=8)
    indexer.index(data[:3], pids[:3], bsize=8)
    print(indexer.embeddings.shape)
    print(indexer.iid2pid)
    print(indexer.pid2iid)
    print(indexer.iid2pid.shape, indexer.pid2iid.shape, indexer.offset)
    # print()
    

    # test some other methods
    test_iids = torch.arange(0, 10).reshape(5, 2).T[:, None]
    test_pids = indexer.iids_to_pids(test_iids)
    test_embs = indexer.get_pid_embedding(test_pids)


    # index the entire data
    # indexer.index(data, pids, bsize=8)
    # indexer.save(INDEX_PATH)
    indexer.load(INDEX_PATH)
    print(indexer.embeddings.shape)
    print(indexer.iid2pid.shape)
    print(indexer.pid2iid.shape)
    print((indexer.pid2iid.sum(dim=-1) == -32).sum())

    # test retrieval
    queries = ["Who is the author of 'The Witcher'?", "How does an NPC behave when it starts raining?", "Who the hell is Cynthia?"]

    Qs = indexer.inference.query_from_text(queries)
    if Qs.dim() == 2:
        Qs = Qs[None]
    # print(Qs.shape)
    
    batch_sim, batch_iids = indexer.search(Qs, k=25)
    # print(batch_sim.shape, batch_iids.shape)
    batch_pids = indexer.iids_to_pids(batch_iids)
    # print(batch_pids)
    batch_embs, batch_masks = indexer.get_pid_embedding(batch_pids)
    # print(batch_embs, batch_masks)
    
    for i, (Q, pids, topk_embs, mask) in enumerate(zip(Qs, batch_pids, batch_embs, batch_masks)):
        # print(Q.shape, pids.shape, topk_embs.shape, mask.shape)

        # topk_embs @ Q.mT instead of Q @ topk_embs.mT because of the masking later on
        sim = topk_embs @ Q.to(dtype=topk_embs.dtype).mT # (N_doc, L_d, L_q)

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
