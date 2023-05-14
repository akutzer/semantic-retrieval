import math
import torch

from retrieval.configs import BaseConfig
from retrieval.data import Passages, Queries, TripleDataset
from retrieval.models import ColBERTTokenizer, ColBERTInference
from retrieval.indexing.colbert_indexer import ColBERTIndexer



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
        # embed the query
        Qs = self.inference.query_from_text(query)
        if Qs.dim() == 2:
            Qs = Qs[None]

        # B, L_q, D = Q.shape
        # Q = Q.reshape(B*L_q, -1)

        # search for the best PIDs
        k_hat = math.ceil(k / 2)
        sim, iids = self.indexer.search(Qs, k=k_hat)
        pids = self.indexer.iids_to_pids(iids)

        # get the pre-computed embeddings for the PIDs
        Ps = [self.indexer.get_pid_embedding(query_best_pids, pad=True) for query_best_pids in pids]

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
        batch_size = 16,
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
    
    LEN = 20_000 #len(dataset) #10_000
    dataset.shuffle()
    
    top1, top5, top10, top25, top100 = 0, 0, 0, 0, 0
    k = 100
    for i, triple in enumerate(tqdm(dataset[:LEN])):
        qid, pid_pos, *pid_neg = triple
        query, psg_pos, *psg_neg = dataset.id2string(triple)

        pids = retriever.full_retrieval(query, k)[0][1]

        if pid_pos in pids[:100]:
            top100 += 1
            if pid_pos in pids[:25]:
                top25 += 1
                if pid_pos in pids[:10]:
                    top10 += 1
                    if pid_pos in pids[:5]:
                        top5 += 1
                        if pid_pos == pids[0]:
                            top1 += 1
    
    print(round((100 * top1) / LEN, 3))
    print(round((100 * top5) / LEN, 3))
    print(round((100 * top10) / LEN, 3))
    print(round((100 * top25) / LEN, 3))
    print(round((100 * top100) / LEN, 3))