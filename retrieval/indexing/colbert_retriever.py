import math
import torch
from typing import List

from retrieval.configs import BaseConfig
from retrieval.data import Passages, Queries, TripleDataset, BucketIterator
from retrieval.models import ColBERTTokenizer, ColBERTInference, get_colbert_and_tokenizer
from retrieval.indexing.colbert_indexer import ColBERTIndexer



class ColBERTRetriever:
    def __init__(self, inference, device):
        self.device = device

        self.inference = inference
        self.indexer = ColBERTIndexer(inference, device=device)


    def full_retrieval(self, query: List[str], k: int):
        # embed the query, performed on the GPU
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
            # `embs @ Q.mT` instead of `Q @ topk_embs.mT` because of the masking later on
            sim = embs @ Q.mT # (N_doc, L_d, L_q)

            # replace the similarity results for padding vectors
            sim[~mask] = -torch.inf

            # calculate the sum of max similarities
            sms = sim.max(dim=1).values.sum(dim=-1)

            # select the top-k PIDs and their similarity score wrt. query
            k_ = min(sms.shape[0], k)
            topk_sims, topk_indices = torch.topk(sms, k=k_)
            topk_pids = pids[topk_indices]
            reranked_pids.append([topk_sims.tolist(), topk_pids.tolist()])
        
        return reranked_pids



if __name__ == "__main__":
    import random
    import numpy as np
    from tqdm import tqdm
    import cProfile

    SEED = 125
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    
    MODEL_PATH = "../../data/colbertv2.0/" # "../../../data/colbertv2.0/" or "bert-base-uncased" or "roberta-base"
    INDEX_PATH = "../../data/fandom-qa/witcher_qa/passages.train.indices.pt"

    config = BaseConfig(
        tok_name_or_path=MODEL_PATH,
        backbone_name_or_path=MODEL_PATH,
        similarity="cosine",
        dim = 128,
        batch_size = 32,
        accum_steps = 1,
    )

    colbert, tokenizer = get_colbert_and_tokenizer(config)
    inference = ColBERTInference(colbert, tokenizer)
    retriever = ColBERTRetriever(inference, device="cuda:0")
    retriever.indexer.load(INDEX_PATH)

    dataset = TripleDataset(config,
        triples_path="../../data/fandom-qa/witcher_qa/triples.train.tsv",
        queries_path="../../data/fandom-qa/witcher_qa/queries.train.tsv",
        passages_path="../../data/fandom-qa/witcher_qa/passages.train.tsv",
        mode="qpp")
    dataset.shuffle()

    BSIZE = 16
    K = 100
    
    top1, top3, top5, top10, top25, top100 = 0, 0, 0, 0, 0, 0

    with cProfile.Profile() as pr:
        query_batch = []
        target_batch = []
        for i, triple in enumerate(tqdm(dataset)):
            qid, pid_pos, *pid_neg = triple
            query, psg_pos, *psg_neg = dataset.id2string(triple)
            query_batch.append(query)
            target_batch.append(pid_pos)

            if len(query_batch) == BSIZE or i + 1 == len(dataset):
                pids = retriever.full_retrieval(query_batch, K)
                
                for i, ((sims, pred_pids), target_pit) in enumerate(zip(pids, target_batch)):
                    if target_pit in pred_pids[:100]:
                        top100 += 1
                        if target_pit in pred_pids[:25]:
                            top25 += 1
                            if target_pit in pred_pids[:10]:
                                top10 += 1
                                if target_pit in pred_pids[:5]:
                                    top5 += 1
                                    if target_pit in pred_pids[:3]:
                                        top3 += 1
                                        if target_pit == pred_pids[0]:
                                            top1 += 1

                    
                
                query_batch = []
                target_batch = []

        pr.print_stats()


    print("Top-1-Acc:", round((100 * top1) / len(dataset), 3))
    print("Top-3-Acc:", round((100 * top3) / len(dataset), 3))
    print("Top-5-Acc:", round((100 * top5) / len(dataset), 3))
    print("Top-10-Acc:", round((100 * top10) / len(dataset), 3))
    print("Top-25-Acc:", round((100 * top25) / len(dataset), 3))
    print("Top-100-Acc:", round((100 * top100) / len(dataset), 3))
