import torch
import numpy as np
from sklearn.cluster import KMeans

from retrieval.configs import BaseConfig
from retrieval.models import ColBERTInference, get_colbert_and_tokenizer


MODEL_PATH = "../../data/colbertv2.0/" # "../../../data/colbertv2.0/" or "bert-base-uncased" or "roberta-base"
config = BaseConfig(
    tok_name_or_path=MODEL_PATH,
    backbone_name_or_path=MODEL_PATH,
    similarity="cosine",
    dim = 128,
)

# load the pretrained ColBERTv2 weights
colbert, tokenizer = get_colbert_and_tokenizer(config)
inference = ColBERTInference(colbert, tokenizer)



query_embedding = inference.query_from_text(["How did Moody feel about the insanity of Alice and Frank Longbottom?"])
# query_embedding shape: (B_q, L_q, D)
print(f"Query embedding shape: {query_embedding.shape} -> (batch_size, fixed_query_len=32, embedding_dim=config.dim={config.dim}))", end="\n\n")

passage1 = "[Personality and traits] At the same time, he was far from being devoid of attachment towards his allies and comrades: He was visibly saddened by the insanity of Alice and Frank Longbottom, noting how being dead would have be better than having to live the rest of their lives insane, and openly acknowledged he was never able to find it easy to get over the loss of a comrade and only by turning the sadness he felt into motivation to get justice was he able to move on, as seen by his expressing sympathy towards Jacob's sibling after they lost Rowan Khanna and even acknowledging he should have trained them well enough."
passage2 = "And the second document/passage!"
passage_embeddings = inference.doc_from_text([passage1, passage2])
# passage_embeddings is a list of tensors each one of shape: (L_d, D)
# since the length of each document/passage (L_d) is variable we can't directly
# concatenate them to a 3d-Tensor, however one could pad them to length max(L_d)
# and then represent them as a 3d-Tensor of shape (B_d, L_d_max, D)
print(f"Passages embedding shapes: {'; '.join(map(lambda x: str(x.shape), passage_embeddings))}", end="\n\n")


# proof that they are already normed:
print(torch.linalg.norm(query_embedding, dim=-1))
print(torch.linalg.norm(passage_embeddings[0], dim=-1))
print(torch.linalg.norm(passage_embeddings[1], dim=-1), end="\n\n")

# (B_q, L_q, D) @ (L_d, D).T = (B_q, L_q, L_d)
cossim_first_passage = query_embedding @ passage_embeddings[0].T
# query_embedding shape: (B_q, L_q, L_d)
print("Cosine similarity of the each query vector with each passage vector:", cossim_first_passage, cossim_first_passage.shape, sep="\n", end="\n\n")

max_cossim = cossim_first_passage.max(dim=-1)
# max_cossim shape: (B_q, L_q)
print("Maximal cosine similarity for each query vector and the index of the corresponding passage vector:", max_cossim.values, max_cossim.indices, sep="\n", end="\n\n")


max_cossim_indices_list = max_cossim.indices.tolist()
freq = {}
#print(max_cossim_indices_list)
for item in max_cossim_indices_list[0]:
    if (item in freq):
        freq[item] += 1
    else:
        freq[item] = 1

max_cossim_indices_list_sorted = list(freq.keys())
max_cossim_indices_list_sorted.sort()

print("max_cossim_indices_list_sorted", max_cossim_indices_list_sorted)

tokens = np.array(tokenizer.tokenize(passage1, "doc"))
best_tokens = tokens[max_cossim_indices_list_sorted]
print("Best Tokens of passage in their original order (according to consine similarity):", best_tokens)

#cluster indices
kmeans = KMeans(n_clusters=3, random_state=0, n_init="auto").fit([[x] for x in max_cossim_indices_list_sorted])
print(kmeans.labels_)

kmeans2 = KMeans(n_clusters=3, random_state=0, n_init="auto").fit([[x] for x in max_cossim_indices_list[0]])
print(kmeans2.labels_)
