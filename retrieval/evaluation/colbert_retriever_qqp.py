#!/usr/bin/env python3
import math
import torch
from typing import List, Union

from retrieval.configs import BaseConfig
from retrieval.data import Passages, Queries, TripleDataset, BucketIterator
from retrieval.models import ColBERTTokenizer, ColBERTInference, get_colbert_and_tokenizer, load_colbert_and_tokenizer
from retrieval.indexing.colbert_indexer import ColBERTIndexer
from retrieval.models.basemodels.tf_idf import TfIdf



class ColBERTRetriever:
    def __init__(self, inference: ColBERTInference, device: Union[str, torch.device] = "cpu", passages=None):
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device

        self.inference = inference
        self.inference.to(device)
        self.indexer = ColBERTIndexer(inference, device=device)
        # TODO: precompute TF-IDF passages!!!!!!!
        self.tfidf = TfIdf(passages = passages)
        # print("tf idf fertig")

    def tf_idf_rank(self, query: List[str], k: int):
        batch_sims, batch_pids = self.tfidf.batchBestKPIDs(k, query) # shape: (B, k)
        batch_pids = torch.tensor(batch_pids, dtype=torch.int32)
        batch_sims = torch.tensor(batch_sims)
        return zip(batch_sims, batch_pids)

    def rerank(self, query: List[str], k: int):
        # embed the query
        Qs = self.inference.query_from_text(query) # (B, L_q, D)
        if Qs.dim() == 2:
            Qs = Qs[None]
        
        # for each query search for the best k PIDs using TF-IDF
        _, batch_pids = self.tfidf.batchBestKPIDs(k, query) # shape: (B, k)

        # since self.indexer.get_pid_embedding expects a torch.Tensor, we
        # need to convert batch_pids to a torch Tensor of shape (B, k)
        batch_pids = torch.tensor(batch_pids, dtype=torch.int32, device=self.device)

        # get the pre-computed embeddings for the PIDs
        batch_embs, batch_masks = self.indexer.get_pid_embedding(batch_pids)
        batch_embs = torch.stack(batch_embs, dim=0,)
        batch_masks = torch.stack(batch_masks, dim=0)
        # batch_embs: Tensor(B, k, L_d, D)
        #   contains for each query the embeddings of the topk documents/passages
        # 
        # batch_masks: Tensor(B, k, L_d)
        #   boolean mask, which is needed since the embedding tensors are padded
        #   (because the number of embedding vectors for each PID is variable),
        #   so we can later ignore the similarity scores for those padding vectors

        sms = self.inference.colbert.similarity(Qs.unsqueeze(1), batch_embs, batch_masks)
        # sms shape: (B, k)

        # select the top-k PIDs and their similarity score wrt. query
        topk_sims, topk_indices = torch.sort(sms, descending=True)
        topk_pids = batch_pids.gather(dim=1, index=topk_indices)

        reranked_pids = zip(topk_sims, topk_pids)
        
        return reranked_pids
    

    def full_retrieval(self, query: List[str], k: int):
        # embed the query
        Qs = self.inference.query_from_text(query) # (B, L_q, D)
        if Qs.dim() == 2:
            Qs = Qs[None]

        # for each query embedding vector, search for the best k_hat index vectors in the passages embedding matrix   
        k_hat = math.ceil(k/2)
        batch_sim, batch_iids = self.indexer.search(Qs, k=k_hat)  # both: (B, L_q, k_hat)

        # for each query get the PIDs containing the best index vectors
        B, L_q = batch_iids.shape[:2]
        batch_pids = self.indexer.iids_to_pids(batch_iids.reshape(B, L_q * k_hat))

        # get the pre-computed embeddings for the PIDs
        batch_embs, batch_masks = self.indexer.get_pid_embedding(batch_pids)

        # batch_embs: List[Tensor(N_pids, L_d, D)]
        #   contains for each query the embeddings for all passages which were in the top-k_hat
        #   for at least one query embedding
        # 
        # batch_masks: List[Tensor(N_pids, L_d)]
        #   boolean mask, which is needed since the embedding tensors are padded
        #   (because the number of embedding vectors for each PID is variable),
        #   so we can later ignore the similarity scores for those padding vectors

        reranked_pids = []
        for Q, pids, embs, mask in zip(Qs, batch_pids, batch_embs, batch_masks):
            sms = self.inference.colbert.similarity(Q[None], embs, mask)
            # sms shape: (k,)

            # select the top-k PIDs and their similarity score wrt. query
            k_ = min(sms.shape[0], k)
            topk_sims, topk_indices = torch.topk(sms, k=k_)
            topk_pids = pids[topk_indices]
            reranked_pids.append([topk_sims, topk_pids])
        
        return reranked_pids
    
    def to(self, device: Union[str, torch.device]) -> None:
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device
        self.inference.to(self.device)
        self.indexer.to(self.device)



if __name__ == "__main__":
    from tqdm import tqdm
    import cProfile
    import pandas as pd
    import numpy as np

    # enable TensorFloat32 tensor cores for float32 matrix multiplication if available
    torch.set_float32_matmul_precision("high")


    BACKBONE = "bert-base-uncased" # "../../../data/colbertv2.0/" or "bert-base-uncased" or "roberta-base"
    INDEX_PATH = "../../data/ms_marco/ms_marco_v1_1/val/passages.indices.pt"
    CHECKPOINT_PATH = "../../data/colbertv2.0/" # or "../../checkpoints/harry_potter_bert_2023-06-03T08:58:15/epoch3_2_loss0.1289_mrr0.9767_acc95.339/"

    config = BaseConfig(
        tok_name_or_path=BACKBONE,
        backbone_name_or_path=BACKBONE,
        similarity="cosine",
        doc_maxlen=512,
        passages_per_query=10,
        checkpoint=CHECKPOINT_PATH
    )

    triples_path = "../../data/fandoms_qa/harry_potter/val/triples.tsv"
    queries_path = "../../data/fandoms_qa/harry_potter/val/queries.tsv"
    passages_path = "../../data/fandoms_qa/harry_potter/val/passages.tsv"
    dataset = TripleDataset(config, triples_path, queries_path, passages_path, mode="QQP")

    # get passage list
    passage_list = [p[1] for p in dataset.passages_items()]
    
    colbert, tokenizer = load_colbert_and_tokenizer(CHECKPOINT_PATH, device="cuda:0", config=config)
    inference = ColBERTInference(colbert, tokenizer)
    retriever = ColBERTRetriever(inference, device="cuda:0", passages=passage_list)
    retriever.indexer.load(INDEX_PATH)


    BSIZE = 8 #8 #16 #8 #16
    K = 100
    top1, top3, top5, top10, top25, top100 = 0, 0, 0, 0, 0, 0
    mrr_10, mrr_100, recall_50 = 0, 0, 0

    df = pd.read_csv(dataset.triples.path, sep='\t')
    df.drop(df.columns[1:2], axis=1, inplace=True)
    prels = df.groupby([df.columns[1]], as_index=False).agg(lambda x: x)
    
    with cProfile.Profile() as pr:
        pids_batch = []
        passage_batch = []
        target_batch = []
        pids_visit = np.zeros(len(dataset), dtype=bool)
        
        for i, triple in enumerate(tqdm(dataset)):

            # for QPP datasets:
            # qid, pid_pos, *pid_neg = triple
            # query, psg_pos, *psg_neg = dataset.id2string(triple)
            # qids_batch.append(qid)
            # query_batch.append(query)
            # target_batch.append(pid_pos)

            # for QQP datasets:
            qid_pos, qid_neg, pid = triple
            query_pos, query_neg, passage = dataset.id2string(triple)
                
            pids_batch.append(pid)
            passage_batch.append(passage)
            target_batch.append(qid_pos)

            if len(query_batch) == BSIZE or i + 1 == len(dataset):
                with torch.autocast(retriever.device.type):
                    qids = retriever.tf_idf_rank(passage_batch, K)
                    # qids = retriever.rerank(passage_batch, K)
                    # qids = retriever.full_retrieval(passage_batch, K)
                
                # print(pids_batch)

                for j, ((sims, pred_qids), pid, target_qid) in enumerate(zip(qids, pids_batch, target_batch)):
                    idx = torch.where(pred_qids == target_qid)[0]
                    # idx = torch.tensor([idx for idx, pred_qid in enumerate(pred_qids) if pred_qid == target_qid])
                    # print(idx, torch.where(pred_qids == target_qid))
                    if idx.numel() == 0:
                        continue
                    # print(target_qid, pred_qids[:10])
                    if idx < 100:
                        top100 += 1
                        mrr_100 += 1 / (idx + 1)
                        if idx < 50:
                            # print(set([prels.iloc[list(prels.iloc[:,0]).index(pid)][1]]))
                            prel = prels.iloc[list(prels.iloc[:,0]).index(pid)][1]
                            if isinstance(prel, np.int64):
                                # print(pid, target_qid, prel, set([prel]), pred_qids[:50])
                                common = set([prel]) & set(pred_qids[:50].cpu().numpy())
                                recall_50 += (len(common) / max(1.0, len(set([prel]))))
                            if isinstance(prel, np.ndarray) and pids_visit[pid]==False:
                                # print(pid, target_qid, prel, set([prel]), pred_qids[:50])
                                common = set(prel) & set(pred_qids[:50].cpu().numpy())
                                recall_50 += (len(common) / max(1.0, len(set(prel))))
                                pids_visit[pid] = True
                            if idx < 25:
                                top25 += 1
                                if idx < 10:
                                    top10 += 1
                                    mrr_10 += 1 / (idx + 1)
                                    if idx < 5:
                                        top5 += 1
                                        if idx < 3:
                                            top3 += 1
                                            if idx < 1:
                                                top1 += 1

                
                pids_batch = []
                passage_batch = []
                target_batch = []
            

        # pr.print_stats()


    print("Top-1-Acc:", round((100 * top1) / len(dataset), 3))
    print("Top-3-Acc:", round((100 * top3) / len(dataset), 3))
    print("Top-5-Acc:", round((100 * top5) / len(dataset), 3))
    print("Top-10-Acc:", round((100 * top10) / len(dataset), 3))
    print("Top-25-Acc:", round((100 * top25) / len(dataset), 3))
    print("Top-100-Acc:", round((100 * top100) / len(dataset), 3))

    print("MRR@10:", round((100 * mrr_10.item()) / len(dataset), 3))
    print("MRR@100:", round((100 * mrr_100.item()) / len(dataset), 3))
    print("Recall@50:", round((100 * recall_50) / len(dataset), 3))
