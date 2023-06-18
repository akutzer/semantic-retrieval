import torch
from typing import Union, List, Tuple
import math

from retrieval.indexing.colbert_retriever import ColBERTRetriever
from retrieval.models.colbert.inference import ColBERTInference

def search(
        query: torch.Tensor, passages: torch.Tensor, k: int, similarity="l2"
    ) -> Tuple[torch.IntTensor, torch.IntTensor]:
    # add batch dimension
    if query.dim() == 2:
        query = query[None]
    query = query.to(dtype=torch.float32)
    # query shape: (B, L_q, D)

    # TODO: use similarity from model config
    if similarity == "l2":
        sim = -1.0 * (query.unsqueeze(-2) - passages.unsqueeze(-3)).pow(2).sum(dim=-1)
        # sim = -1.0 * torch.norm(query - self.embeddings, ord=2, dim=-1) # shape: (B * L_q, N_embs)
        # sim shape: (B * L_q, N_embs)

    elif similarity == "cosine":
        sim = query @ passages.mT  # shape: (B, L_q, N_embs)
    else:
        raise ValueError()

    topk_sim, topk_iids = sim.topk(k, dim=-1)  # both shapes: (B, L_q, k)
    return topk_sim, topk_iids

def full_retrieval_embedding_layer_only(query: List[str], passages: List[str], k: int):
    inference = ColBERTInference.from_pretrained("../../data/colbertv2.0/")
    # embed the query
    query, mask_q = inference.tokenizer.tensorize(query, mode="doc")
    passages, mask_p = inference.tokenizer.tensorize(passages, mode="doc")
    word_emb_q = inference.colbert.backbone.embeddings.word_embeddings(query)
    word_emb_q = word_emb[mask_q.bool()]
    word_emb_p = inference.colbert.backbone.embeddings.word_embeddings(passages)
    word_emb_p = word_emb_p[mask_p.bool()]

    if word_emb_q.dim() == 2:
        Qs = word_emb_q[None]

    # for each query embedding vector, search for the best k_hat index vectors in the passages embedding matrix
    k_hat = math.ceil(k / 2)  # math.ceil(k/10)
    batch_sim, batch_iids = search(word_emb_q, k=k_hat)  # both: (B, L_q, k_hat)

    # for each query get the PIDs containing the best index vectors
    B, L_q = batch_iids.shape[:2]
    batch_pids = indexer.iids_to_pids(batch_iids.reshape(B, L_q * k_hat))

    # get the pre-computed embeddings for the PIDs
    batch_embs, batch_masks = indexer.get_pid_embedding(batch_pids)

    # batch_embs: List[Tensor(N_pids, L_d, D)]
    #   contains for each query the embeddings for all passages which were in the top-k_hat
    #   for at least one query embedding
    #
    # batch_masks: List[Tensor(N_pids, L_d)]
    #   boolean mask, which is needed since the embedding tensors are padded
    #   (because the number of embedding vectors for each PID is variable),
    #   so we can later ignore the similarity scores for those padding vectors

    reranked_pids = []
    for Q, pids, embs, mask in zip(word_emb_q, batch_pids, batch_embs, batch_masks):
        sms = inference.colbert.similarity(Q[None], embs, mask)
        # print(sms.shape)
        # sms shape: (k,)

        # select the top-k PIDs and their similarity score wrt. query
        k_ = min(sms.shape[0], k)
        topk_sims, topk_indices = torch.topk(sms, k=k_)
        topk_pids = pids[topk_indices]
        reranked_pids.append([topk_sims, topk_pids])

    return reranked_pids

colbert = ColBERTInference.from_pretrained("../../data/colbertv2.0/")
query, mask = colbert.tokenizer.tensorize(["How are", "passage2", "passage3", "I am fine", "passage5", "passage6"], mode="doc")
passages, mask_p = colbert.tokenizer.tensorize("I am fine", mode="doc")

# mask = mask.bool()
print("query.shape, mask.shape", query.shape, mask.shape)
print("colbert.colbert.backbone.embeddings", colbert.colbert.backbone.embeddings)
# print(help(colbert.colbert.backbone.embeddings))

# Token embedding, positional embedding, and normalization
normal_emb = colbert.colbert.backbone.embeddings(query)
print("normal_emb.shape", normal_emb.shape)

# maybe it is better to remove the padding tokens (when using batches of strings as input)
normal_emb = normal_emb[mask.bool()]
print("normal_emb, normal_emb.shape", normal_emb, normal_emb.shape)

# The raw token embedding:
word_emb = colbert.colbert.backbone.embeddings.word_embeddings(query)
word_emb = word_emb[mask.bool()]
word_emb_p = colbert.colbert.backbone.embeddings.word_embeddings(passages)
word_emb_p = word_emb_p[mask_p.bool()]
print("word_emb, word_emb.shape", word_emb, word_emb.shape)
print("word_emb_p, word_emb_p.shape", word_emb_p, word_emb_p.shape)

#word_emb = torch.linalg.norm(word_emb, ord=2, dim=-1)
#print("word_emb, word_emb.shape", word_emb, word_emb.shape)

print("search", search(word_emb, word_emb_p, k=4, similarity="cosine"))

print(full_retrieval_embedding_layer_only(["How are you"],["How are", "passage2", "passage3", "I am fine", "passage5", "passage6"], 3))



#print([sum(x) for x in word_emb])