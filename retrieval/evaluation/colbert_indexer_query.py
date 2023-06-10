#!/usr/bin/env python3
import time
from tqdm import tqdm
import torch
import torch.nn.functional as F
from collections import defaultdict
from typing import Union, List, Tuple
import logging

from retrieval.configs import BaseConfig
from retrieval.data import Queries
from retrieval.models import ColBERTTokenizer, ColBERTInference, get_colbert_and_tokenizer, load_colbert_and_tokenizer
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
        self.iid2qid = torch.empty(0, device=self.device, dtype=torch.int32)
        self.qid2iid = torch.empty((0, 0), device=self.device, dtype=torch.int32)
        self.next_iid = 0
            
    def index(self, queries: List[str], qids: List[str], bsize: int = 16) -> None:
        batch_queries, batch_qids = self._new_queries(queries, qids)
        # if there are no queries which haven't been indexed yet, return
        if len(queries) == 0:
            return

        with torch.inference_mode():
            with torch.autocast(self.device.type):
                queries_embedded = self.inference.query_from_text(batch_queries, bsize=bsize, show_progress=True)
            assert len(queries_embedded) == len(batch_qids)

            # calculate the new width of the qid2iid matrix
            # (aka the maximal number of IIDs which are assigned to a single QID)
            max_iids_per_qid = max(max(emb.shape[0] for emb in queries_embedded), self.qid2iid.shape[-1])

            # calculate the new height of the qid2iid matrix (aka the new maximal QIDs)
            max_qid = max(max(qids) + 1, self.qid2iid.shape[0])

            # extends the qid2iid matrix (padded with -1)
            qid2iid = torch.full((max_qid, max_iids_per_qid), -1, dtype=torch.int32, device=self.device)
            qid2iid[:self.qid2iid.shape[0], :self.qid2iid.shape[1]] = self.qid2iid

            # extends the iid2qid vector
            num_new_iids = sum(emb.shape[0] for emb in queries_embedded)
            iid2qid = torch.empty((self.iid2qid.shape[0] + num_new_iids,), dtype=torch.int32, device=self.device)
            iid2qid[:self.iid2qid.shape[0]] = self.iid2qid
            
            # update the qid2iid matrix and iid2qid vector for the new embeddings
            next_iid_ = self.iid2qid.shape[0]
            for qid, emb in zip(batch_qids, queries_embedded):
                # adds new iids in the range [next_iid_, next_iid_ + n_embeddings)
                new_iids = torch.arange(next_iid_, next_iid_ + emb.shape[0], device=self.device)
                qid2iid[qid, :emb.shape[0]] = new_iids
                iid2qid[new_iids] = qid
                next_iid_ += emb.shape[0]
            
            # Update the mappings
            self.qid2iid = qid2iid
            self.iid2qid = iid2qid
            
            # Concatenate the new embeddings onto the previous embedding matrix
            # this is done on the cpu to save GPU-RAM
            cat = torch.cat([self.embeddings.cpu(), *map(lambda x: x.cpu(), queries_embedded)], dim=0)
            del queries_embedded
            self.embeddings = cat.to(device=self.device, dtype=self.dtype)
            # self.embeddings = torch.cat([self.embeddings, *queries_embedded], dim=0)
    
    def search(self, passage: torch.Tensor, k: int) -> Tuple[torch.IntTensor, torch.IntTensor]:
        # add batch dimension
        if passage.dim() == 2:
            passage = passage[None]
        passage = passage.to(dtype=self.dtype)
        # passage shape: (B, L_p, D)

        # TODO: use similarity from model config
        if self.similarity == "l2":
            sim = -1.0 * (passage.unsqueeze(-2) - self.embeddings.unsqueeze(-3)).pow(2).sum(dim=-1) # shape: (B * L_p, N_embs)
            # sim = -1.0 * torch.norm(passage - self.embeddings, ord=2, dim=-1) # shape: (B * L_p, N_embs)

        elif self.similarity == "cosine":
            sim = passage @ self.embeddings.mT # shape: (B, L_p, N_embs)
        else:
            raise ValueError()
        
        topk_sim, topk_iids = sim.topk(k, dim=-1) # both shapes: (B, L_p, k)
        return topk_sim, topk_iids
            
    def iids_to_qids(self, batch_iids: torch.IntTensor) -> List[torch.IntTensor]:
        # add batch dimension
        if batch_iids.dim() == 0:
            batch_iids = batch_iids[None, None]
        elif batch_iids.dim() == 1:
            batch_iids = batch_iids[None]

        batch_qids = []
        for iids in self.iid2qid[batch_iids]:
            # TODO torch.unique is major bottleneck of the retriever
            batch_qids.append(torch.unique(iids))

        return batch_qids
    
    def get_qid_embedding(self, batch_qids: List[torch.IntTensor]) -> Tuple[List[torch.Tensor], List[torch.BoolTensor]]:
        batch_embs, batch_masks = [], []
        for qids in batch_qids:
            iids = self.qid2iid[qids]
            embs = self.embeddings[iids]
            mask = iids != -1
            embs[~mask] = 0

            batch_embs.append(embs)
            batch_masks.append(mask)

        return batch_embs, batch_masks

    def save(self, path: str):
        parameters = {
            "iid2qid": self.iid2qid.cpu(),
            "qid2iid": self.qid2iid.cpu(),
            "embeddings": self.embeddings.cpu(),
        }
        torch.save(parameters, path)
    
    def load(self, path: str):
        parameters = torch.load(path)
        self.iid2qid = parameters["iid2qid"].to(self.device)
        self.qid2iid = parameters["qid2iid"].to(self.device)
        self.embeddings = parameters["embeddings"].to(self.device)
        self.dtype = self.embeddings.dtype
        logging.info(f"Successfully loaded the precomputed indices. Changed dtype to {self.dtype}!")
        
    def _new_queries(self, queries: List[str], qids: List[str]) -> Tuple[List[str], List[str]]:
        queries_, qids_ = [], []

        # if the indexer is uninitialized use all passages
        if self.qid2iid.numel() == 0:
            queries_, qids_ = queries, qids

        # else use only passages from unseen qids
        else:
            for qid, query in zip(qids, queries):
                if qid > self.qid2iid.shape[0] - 1 or self.qid2iid[qid][0] == -1:
                    queries_.append(query)
                    qids_.append(qid)

        return queries_, qids_
    
    def to(self, device: Union[str, torch.device]) -> None:
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device

        self.inference.to(self.device)
        self.embeddings = self.embeddings.to(self.device)
        self.iid2qid = self.iid2qid.to(self.device)
        self.qid2iid = self.qid2iid.to(self.device)

    


if __name__ == "__main__":
    import random
    import numpy as np

    SEED = 125
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    # enable TensorFloat32 tensor cores for float32 matrix multiplication if available
    torch.set_float32_matmul_precision("high")

    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    QUERIES_PATH = "../../data/fandoms_qa/harry_potter/val/queries.tsv"
    INDEX_PATH = "../../data/fandoms_qa/harry_potter/val/queries.indices.pt"
    BACKBONE = "bert-base-uncased" # "../../../data/colbertv2.0/" or "bert-base-uncased" or "roberta-base"
    CHECKPOINT_PATH = "../../data/colbertv2.0/"
    

    config = BaseConfig(
        tok_name_or_path=BACKBONE,
        backbone_name_or_path=BACKBONE,
        similarity="cosine",
        dim = 128,
        batch_size = 16,
        accum_steps = 1,
        doc_maxlen=512,
        checkpoint=CHECKPOINT_PATH
    )
    colbert, tokenizer = load_colbert_and_tokenizer(CHECKPOINT_PATH, device=DEVICE, config=config)
    print(config)
    inference = ColBERTInference(colbert, tokenizer, device=DEVICE)
    indexer = ColBERTIndexer(inference, device=DEVICE, dtype=torch.float16)

    queries = Queries(QUERIES_PATH)
    data = queries.values().tolist()
    qids = queries.keys().tolist()
    
    # test indexing of already seen data
    indexer.index(data[:1], qids[:1], bsize=8)
    indexer.index(data[1:2], qids[1:2], bsize=8)
    indexer.index(data[:3], qids[:3], bsize=8)
    print(indexer.embeddings.shape)
    print(indexer.iid2qid)
    print(indexer.qid2iid)

    # test some other methods
    test_iids = torch.arange(0, 10).reshape(5, 2).T[:, None]
    test_qids = indexer.iids_to_qids(test_iids)
    test_embs = indexer.get_qid_embedding(test_qids)

    # index the entire data
    indexer.index(data, qids, bsize=8)
    indexer.save(INDEX_PATH)
    indexer.load(INDEX_PATH)
    print(indexer.embeddings.shape)


    # test retrieval
    passages = ["[Biography] In the forests of Angren on July 25, , Letho was near death after being struck \
                by a slyzard's tail but Geralt, a witcher from the School of the Wolf, found and saved him \
                while chasing after the Wild Hunt to rescue Yennefer. As thanks for saving his life, Letho \
                told Geralt where to find the Wild Hunt and, alongside his own companions, travelled with \
                him to find the spectral riders. Eventually, the group caught up with the Wild Hunt on the \
                Winter Solstice, by the Hanged Man's Tree in Nilfgaard. Despite their skill, the witchers \
                couldn't defeat all the warriors and a stalemate ensued before Geralt offered himself in \
                exchange for Yennefer.", 
                "[Killing kings] In an attempt to catch Letho, Geralt, who now knew that Letho had attempted \
                to get rid of Iorveth, the elf to the kingslayer as a ruse to expose the latter's treachery. \
                However, Iorveth and his Scoia'tael were attacked by the Blue Stripes, leaving Geralt and \
                Letho to battle each other. Letho prevailed in the battle with Geralt, but once the Wolf was \
                down, he spared him. Letho revealed that Geralt once saved his life and so they were now \
                'even', before running off to kidnap Triss and force her to teleport them to Aedirn. He \
                succeeded in capturing the sorceress, but fought and killed Cedric in the process as he \
                tried to defend her.", 
                "[Walkthrough] After Trial by Fire, Geralt, Roche and Foltest finally reach La Valette \
                Castle's temple. Within are numerous common folk seeking refuge and praying at the central \
                altar, as well as several priests and a disgraced nobleman. Foltest reprimands the latter, \
                Arthur Tailles, a count he had condemned who was then 'pardoned' and protected by Baroness \
                Mary Louisa. Geralt questions the Archpriest for the location of the two children; he is \
                given no choice but to use the Axii sign, to which Tailles strenuously objects. (Whether \
                or not Axii actually succeeds, the Archpriest tells the truth: the children are in the \
                solar, in the next tower along the castle wall.)"]

    Ps = indexer.inference.doc_from_text(passages)
    if Ps.dim() == 2:
        Ps = Ps[None]
    # print(Ps.shape)
    
    batch_sim, batch_iids = indexer.search(Ps, k=25)
    # print(batch_sim.shape, batch_iids.shape)
    batch_qids = indexer.iids_to_qids(batch_iids)
    # print(batch_qids)
    batch_embs, batch_masks = indexer.get_qid_embedding(batch_qids)
    # print(batch_embs, batch_masks)
    
    for i, (P, qids, topk_embs, mask) in enumerate(zip(Ps, batch_qids, batch_embs, batch_masks)):
        # print(P.shape, qids.shape, topk_embs.shape, mask.shape)

        # topk_embs @ P.mT instead of P @ topk_embs.mT because of the masking later on
        sim = topk_embs @ P.to(dtype=topk_embs.dtype).mT 

        # replace the similarity results for padding vectors
        sim[~mask] = -torch.inf

        # calculate the sum of max similarities
        sms = sim.max(dim=1).values.sum(dim=-1)
        # print(sim.shape, sms.shape)

        values, indices = torch.sort(sms, descending=True)
        sorted_qids = qids[indices]
        # print(qids, values, indices)
        # print(values, sorted_qids)

        print("=" * 150)
        print(f"Passage: {passages[i]}")
        print("=" * 150)
        for sim, qid in zip(values, sorted_qids[:10]):
            print(round(sim.item(), 3), qid.item(),  queries[qid.item()])
        print(end="\n\n\n")

