import math
import torch
import time

from retrieval.configs import BaseConfig
from retrieval.data import Passages, Queries, TripleDataset, BucketIterator
from retrieval.models import ColBERTTokenizer, ColBERTInference
from retrieval.indexing.colbert_indexer import ColBERTIndexer


query_from_text_time = 0
search_time = 0
iids_to_pids_time = 0
get_pid_embedding_time = 0
reranked_pids_time = 0


class ColBERTRetriever:
    def __init__(self, config, device):
        self.config = config
        self.device = device

        self.tokenizer = ColBERTTokenizer(self.config)
        self.inference = ColBERTInference(self.config, self.tokenizer, device=device)
        self.indexer = ColBERTIndexer(config, device=device)
        INDEX_PATH = "../../data/fandom-qa/witcher_qa/passages.train.indices.pt"
        self.indexer.load(INDEX_PATH)


    def full_retrieval(self, query, k: int):
        global query_from_text_time, search_time, iids_to_pids_time, get_pid_embedding_time, reranked_pids_time

        
        start = time.time()
        # embed the query, performed on the GPU
        Qs = self.inference.query_from_text(query) # (B, L_q, D)
        if Qs.dim() == 2:
            Qs = Qs[None]
        query_from_text_time += time.time() - start


        start = time.time()
        # for each query embedding vector, search for the best k_hat index vectors in the passages embedding matrix
        # performed on the GPU      
        k_hat = math.ceil(k/10)
        sim, iids = self.indexer.search(Qs, k=k_hat) # (B, L_q, k_hat)
        search_time += time.time() - start

        start = time.time()
        # for each query get the PIDs containing the best index vectors
        # performed in the cpu
        pids = self.indexer.iids_to_pids(iids.cpu())  # List[List[int]]
        # outer list is of length B
        # inner list is of variable length N_pids, since there are a different number of 
        # pids for each query
        iids_to_pids_time += time.time() - start

        start = time.time()
        # get the pre-computed embeddings for the PIDs
        Ps = [self.indexer.get_pid_embedding(query_best_pids, pad=True) for query_best_pids in pids]
        # List[Tensor(N_pids, L_d, D), Tensor(N_pids, L_d)]
        # first Tensor contains all the embeddings of matching passages for
        # the given query; the second tensor is a boolean mask, which is needed 
        # since this Tensor is padded (because the number of embedding vectors
        # for each PID is variable), so we can later ignore the similarity scores
        # for those padding vectors
        get_pid_embedding_time += time.time() - start


        start = time.time()
        reranked_pids = []
        for Q, topk_pids, (topk_embs, mask) in zip(Qs, pids, Ps):
            # topk_embs @ Q.mT instead of Q @ topk_embs.mT because of the masking later on
            sim = topk_embs @ Q.mT # (N_doc, L_d, L_q)

            # replace the similarity results for padding vectors
            sim[~mask] = -torch.inf

            # calculate the sum of max similarities
            sms = sim.max(dim=1).values.sum(dim=-1)
            # print(sim.shape, max_sim.shape)


            k_ = min(sms.shape[0], k)
            values, indices = torch.topk(sms, k=k_)
            sorted_pids = torch.tensor(topk_pids, device=indices.device)[indices]
            reranked_pids.append([values.tolist(), sorted_pids.tolist()])

        reranked_pids_time += time.time() - start


        # reranked_pids = []
        # for topk_pids, batch_P in zip(pids, Ps):
        #     sims = []
        #     for P in batch_P:
        #         # print(Q.shape, pid_emb.shape)
        #         out = Q @ P.T
        #         sim = out.max(dim=-1).values.sum()
        #         sims.append(sim)
        #     k_ = min(len(sims), k)
        #     values, indices = torch.topk(torch.tensor(sims), k=k_)
        #     # print(values, imd)
        #     sorted_pids = torch.tensor(topk_pids)[indices]
        #     reranked_pids.append([values.tolist(), sorted_pids.tolist()])
        
        return reranked_pids



if __name__ == "__main__":
    import random
    import numpy as np
    from tqdm import tqdm

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
        batch_size = 32,
        accum_steps = 1,
    )


    retriever = ColBERTRetriever(config, device="cuda:0")
    # query = "Who is the author of 'The Witcher'?"
    # k =  20

    # pids = retriever.full_retrieval(query, k)


    # PATH = "../../data/fandom-qa/witcher_qa/passages.train.tsv"
    # passages = Passages(PATH)
    # for batch in pids:
    #     for sim, pid in batch:
    #         print(round(sim.item(), 3), pid.item(),  passages[pid.item()])

    dataset = TripleDataset(config,
        triples_path="../../data/fandom-qa/witcher_qa/triples.train.tsv",
        queries_path="../../data/fandom-qa/witcher_qa/queries.train.tsv",
        passages_path="../../data/fandom-qa/witcher_qa/passages.train.tsv",
        mode="qpp")
    
    import cProfile
    
    LEN = len(dataset) #len(dataset) #len(dataset) #20_000
    dataset.shuffle()
    sub_dataset = dataset[:LEN]
    BSIZE = 16
    
    top1, top5, top10, top25, top100 = 0, 0, 0, 0, 0
    k = 100
    with cProfile.Profile() as pr:
        query_batch = []
        target_batch = []
        for i, triple in enumerate(tqdm(sub_dataset)):
            qid, pid_pos, *pid_neg = triple
            query, psg_pos, *psg_neg = dataset.id2string(triple)
            query_batch.append(query)
            target_batch.append(pid_pos)

            if len(query_batch) == BSIZE or i + 1 == len(sub_dataset):
                pids = retriever.full_retrieval(query_batch, k)
                
                for i, ((sims, pred_pids), target_pit) in enumerate(zip(pids, target_batch)):
                    if target_pit in pred_pids[:100]:
                        top100 += 1
                        if target_pit in pred_pids[:25]:
                            top25 += 1
                            if target_pit in pred_pids[:10]:
                                top10 += 1
                                if target_pit in pred_pids[:5]:
                                    top5 += 1
                                    if target_pit == pred_pids[0]:
                                        top1 += 1

                    
                
                query_batch = []
                target_batch = []

        pr.print_stats()


    print(round((100 * top1) / len(sub_dataset), 3))
    print(round((100 * top5) / len(sub_dataset), 3))
    print(round((100 * top10) / len(sub_dataset), 3))
    print(round((100 * top25) / len(sub_dataset), 3))
    print(round((100 * top100) / len(sub_dataset), 3))

    print(f"query_from_text_time = {query_from_text_time}")
    print(f"search_time = {search_time}")
    print(f"iids_to_pids_time = {iids_to_pids_time}")
    print(f"get_pid_embedding_time = {get_pid_embedding_time}")
    print(f"reranked_pids_time = {reranked_pids_time}")
