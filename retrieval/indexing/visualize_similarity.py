import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import heapq


from scipy.stats import gaussian_kde
from sklearn.cluster import KMeans

from retrieval.configs import BaseConfig
from retrieval.models import ColBERTInference, get_colbert_and_tokenizer

MODEL_PATH = "../../data/colbertv2.0/"  # "../../../data/colbertv2.0/" or "bert-base-uncased" or "roberta-base"


def most_frequent(List):
    return max(set(List), key = List.count)
def rgbk_to_hex(r, g, b,k):
    return '#{:02x}{:02x}{:02x}{:02x}'.format(int(r*255), int(g*255), int(b*255), int(k*255))

def highlighter(word, token_index, density, cmap, norm):
    color = cmap(1-norm(density(token_index)))[0]
    color_hex = rgbk_to_hex(color[0],color[1],color[2],color[3])
    word = '<span style="background-color:' +color_hex+ '">' +word+ '</span>'

    return word

def html_heatmap(tokens, max_cossim_indices_list , plot = True):
    '''Args:    tokens:                     all tokens of the passage in their order
                max_cossim_indices_list     32 (64) indices of the tokens with the highest cosine similarity with the
                                            32 (64) query vectors (duplicates necessary)
    '''
    num_best_tokens = len(max_cossim_indices_list)
    # plot density function
    density = gaussian_kde(max_cossim_indices_list)


    xgrid = np.linspace(min(max_cossim_indices_list), max(max_cossim_indices_list), num_best_tokens)
    plt.plot(xgrid, density(xgrid))
    xmin, xmax, ymin, ymax = plt.axis()

    if plot:
        plt.show()
    # get index of best token
    #best_token_ind = most_frequent(max_cossim_indices_list)

    cmap = matplotlib.colormaps.get_cmap('Spectral')
    norm = matplotlib.colors.Normalize(vmin=ymin, vmax=ymax)

    html_heatmap = ' '.join([highlighter(tokens[i], i, density, cmap, norm) for i in range(0, len(tokens))])
    #create html file with heatmapped text
    f = open('heatmap.html', 'w')
    f.write(html_heatmap)
    f.close()

def get_max_cossim_indices(inference, query, passage, n_largest=1):
    # query_embedding shape: (B_q, L_q, L_d)
    query_embedding = inference.query_from_text(query)
    passage_embeddings = inference.doc_from_text([passage])
    # (B_q, L_q, D) @ (L_d, D).T = (B_q, L_q, L_d)
    cossim_passage = query_embedding @ passage_embeddings[0].T
    # max_cossim shape: (B_q, L_q)
    #print(heapq.nlargest(n_largest, cossim_passage))
    max_cossim = cossim_passage.max(dim=-1)
    max_cossim_indices_list = max_cossim.indices.tolist()
    return max_cossim_indices_list

if __name__ == "__main__":
    #load the pretrained ColBERTv2 weights
    # config = BaseConfig(
    #     tok_name_or_path=MODEL_PATH,
    #     backbone_name_or_path=MODEL_PATH,
    #     similarity="cosine",
    #     dim=128,
    # )
    # colbert, tokenizer = get_colbert_and_tokenizer(config)
    # inference = ColBERTInference(colbert, tokenizer)
    #
    # query_embedding = inference.query_from_text(
    #     ["How did Moody feel about the insanity of Alice and Frank Longbottom?"])
    # # query_embedding shape: (B_q, L_q, D)
    # print(
    #     f"Query embedding shape: {query_embedding.shape} -> (batch_size, fixed_query_len=32, embedding_dim=config.dim={config.dim}))",
    #     end="\n\n")
    #
    # passage1 = "[Personality and traits] At the same time, he was far from being devoid of attachment towards his allies and comrades: He was visibly saddened by the insanity of Alice and Frank Longbottom, noting how being dead would have be better than having to live the rest of their lives insane, and openly acknowledged he was never able to find it easy to get over the loss of a comrade and only by turning the sadness he felt into motivation to get justice was he able to move on, as seen by his expressing sympathy towards Jacob's sibling after they lost Rowan Khanna and even acknowledging he should have trained them well enough."
    # passage2 = "And the second document/passage!"
    # passage_embeddings = inference.doc_from_text([passage1, passage2])
    # # passage_embeddings is a list of tensors each one of shape: (L_d, D)
    # # since the length of each document/passage (L_d) is variable we can't directly
    # # concatenate them to a 3d-Tensor, however one could pad them to length max(L_d)
    # # and then represent them as a 3d-Tensor of shape (B_d, L_d_max, D)
    # print(f"Passages embedding shapes: {'; '.join(map(lambda x: str(x.shape), passage_embeddings))}", end="\n\n")
    #
    # # proof that they are already normed:
    # print(torch.linalg.norm(query_embedding, dim=-1))
    # print(torch.linalg.norm(passage_embeddings[0], dim=-1))
    # print(torch.linalg.norm(passage_embeddings[1], dim=-1), end="\n\n")
    #
    # # (B_q, L_q, D) @ (L_d, D).T = (B_q, L_q, L_d)
    # cossim_first_passage = query_embedding @ passage_embeddings[0].T
    # # query_embedding shape: (B_q, L_q, L_d)
    # print("Cosine similarity of the each query vector with each passage vector:", cossim_first_passage,
    #       cossim_first_passage.shape, sep="\n", end="\n\n")
    #
    # max_cossim = cossim_first_passage.max(dim=-1)
    # # max_cossim shape: (B_q, L_q)
    # print("Maximal cosine similarity for each query vector and the index of the corresponding passage vector:",
    #       max_cossim.values, max_cossim.indices, sep="\n", end="\n\n")
    #
    # max_cossim_indices_list = max_cossim.indices.tolist()
    # freq = {}
    # # print(max_cossim_indices_list)
    # for item in max_cossim_indices_list[0]:
    #     if (item in freq):
    #         freq[item] += 1
    #     else:
    #         freq[item] = 1
    #
    # max_cossim_indices_list_sorted = list(freq.keys())
    # max_cossim_indices_list_sorted.sort()
    #
    # print("max_cossim_indices_list_sorted", max_cossim_indices_list_sorted)
    #
    # tokens = np.array(tokenizer.tokenize(passage1, "doc"))
    # best_tokens = tokens[max_cossim_indices_list_sorted]
    # print("Best Tokens of passage in their original order (according to consine similarity):", best_tokens)

    # cluster indices with KMeans
    # kmeans = KMeans(n_clusters=3, random_state=0, n_init="auto").fit([[x] for x in max_cossim_indices_list_sorted])
    # print(kmeans.labels_)
    # kmeans2 = KMeans(n_clusters=3, random_state=0, n_init="auto").fit([[x] for x in max_cossim_indices_list[0]])
    # print(kmeans2.labels_)



    # load the pretrained ColBERTv2 weights
    config = BaseConfig(
        tok_name_or_path=MODEL_PATH,
        backbone_name_or_path=MODEL_PATH,
        similarity="cosine",
        dim=128,
    )
    colbert, tokenizer = get_colbert_and_tokenizer(config)
    inference = ColBERTInference(colbert, tokenizer)
    query = "How did Moody feel about the insanity of Alice and Frank Longbottom?"
    passage = "[Personality and traits] At the same time, he was far from being devoid of attachment towards his allies and comrades: He was visibly saddened by the insanity of Alice and Frank Longbottom, noting how being dead would have be better than having to live the rest of their lives insane, and openly acknowledged he was never able to find it easy to get over the loss of a comrade and only by turning the sadness he felt into motivation to get justice was he able to move on, as seen by his expressing sympathy towards Jacob's sibling after they lost Rowan Khanna and even acknowledging he should have trained them well enough."

    max_cossim_indices_list = get_max_cossim_indices(inference, query, passage)
    tokens = np.array(tokenizer.tokenize(passage, "doc"))
    html_heatmap(tokens, max_cossim_indices_list, True)
    #TODO: top n extrahieren statt nur top 1 -> 64 statt 32
    #Todo: max_cossim_indices_list may not only contain one type of number [4,4,4,4] throw error