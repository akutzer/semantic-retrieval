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


def highlighter(word, normed_value, cmap):
    color = cmap(normed_value)
    color_hex = rgbk_to_hex(color[0],color[1],color[2],color[3])
    word = '<span style="background-color:' +color_hex+ '">' +word+ '</span>'
    return word


def html_heatmap(tokens, flat_topk_cossim_indices_list, flat_topk_cossim_values_list, plot = True, symbol='#'):
    '''Args:    tokens:                     all tokens of the passage in their order
                max_cossim_indices_list     32 (64) indices of the tokens with the highest cosine similarity with the
                                            32 (64) query vectors (duplicates necessary)
    '''
    num_best_tokens = len(flat_topk_cossim_indices_list)

    #remove first two and last token for kde
    kde_flat_topk_cossim_indices_list = [i for i in flat_topk_cossim_indices_list if i != 0 and i!=1 and i!=(len(tokens)-1)]
    # plot density function
    density = gaussian_kde(kde_flat_topk_cossim_indices_list)

    #remove leading #-symbols
    stripped_tokens = []
    for token in tokens:
        stripped_token = token.lstrip(symbol)
        if len(stripped_token) == 0:
            stripped_token = token
        stripped_tokens.append(stripped_token)
    #print(stripped_tokens)

    xgrid = np.linspace(min(flat_topk_cossim_indices_list), max(flat_topk_cossim_indices_list), num_best_tokens)
    plt.plot(xgrid, density(xgrid))
    xmin, xmax, ymin, ymax = plt.axis()

    if plot:
        plt.show()
    # get index of best token
    #best_token_ind = most_frequent(max_cossim_indices_list)

    cmap = matplotlib.colormaps.get_cmap('Blues_r')

    #kde heatmap
    norm_kde = matplotlib.colors.Normalize(vmin=ymin, vmax=ymax)
    html_heatmap_kde = ' '.join([highlighter(stripped_tokens[i], 1-norm_kde(density(i))[0], cmap) for i in range(0, len(stripped_tokens))])

    #absolute heatmap
    count_ind = [flat_topk_cossim_indices_list.count(i) for i in range(0, len(stripped_tokens))]
    norm_absolute = matplotlib.colors.Normalize(vmin=0, vmax=max(count_ind))
    html_heatmap_absolute = ' '.join([highlighter(stripped_tokens[i], 1-norm_absolute(count_ind[i]), cmap)
                                      for i in range(0, len(stripped_tokens))])

    #added values heatmap
    sum_values = [0]*len(stripped_tokens)
    for i in range(0, len(flat_topk_cossim_indices_list)):
        sum_values[flat_topk_cossim_indices_list[i]] += flat_topk_cossim_values_list[i]
    norm_added_values = matplotlib.colors.Normalize(vmin=0, vmax=max(sum_values))
    html_added_values = ' '.join([highlighter(stripped_tokens[i], 1 - norm_added_values(sum_values[i]), cmap)
                                      for i in range(0, len(tokens))])

    #create html file with heatmapped text
    f = open('heatmap_kde.html', 'w')
    f.write(html_heatmap_kde)
    f.close()
    f = open('heatmap_absolute.html', 'w')
    f.write(html_heatmap_absolute)
    f.close()
    f = open('heatmap_added_values.html', 'w')
    f.write(html_added_values)
    f.close()
    return html_heatmap_kde, html_heatmap_absolute, html_added_values
def tokens_to_indices(tokens, string, symbol = '#'):
    '''maps tokens to their begin and end idex in the string
    currently not used'''
    lower_string = string.lower()
    start = 0
    indices = []
    for token in tokens:
        stripped_token = token.replace(symbol, "")
        if len(stripped_token) == 0:
            stripped_token = token
        stripped_token = stripped_token.lower()
        index = lower_string.find(stripped_token, start)
        start = index + len(stripped_token)
        indices.append((index, index+len(stripped_token)))

    return indices


def get_topk_cossim_indices_and_values(k, inference, query, passage, n_largest=1):
    # query_embedding shape: (B_q, L_q, L_d)
    query_embedding = inference.query_from_text(query)
    passage_embeddings = inference.doc_from_text([passage])
    # (B_q, L_q, D) @ (L_d, D).T = (B_q, L_q, L_d)
    cossim_passage = query_embedding @ passage_embeddings[0].T
    # max_cossim shape: (B_q, L_q)
    topk_cossim = cossim_passage.topk(k=k, dim=-1)
    topk_cossim_indices_list = topk_cossim.indices.tolist()
    topk_cossim_values_list = topk_cossim.values.tolist()
    return topk_cossim_indices_list, topk_cossim_values_list

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
        doc_maxlen=512
    )
    colbert, tokenizer = get_colbert_and_tokenizer(config)
    inference = ColBERTInference(colbert, tokenizer)
    queries = ["How did Moody feel about the insanity of Alice and Frank Longbottom?",
              'Who is the author of ''The Witcher''?',
              'How does an NPC react if it starts raining?',
              'How does an NPC behave if it starts raining?',
              'What is the name of the red car?',
               "Who did Sigismund recruit as his informant during the interwar period?",
               "What did Yennefer do to protect Dandelion when he was captured by Rience?",
               "Why did Temerian agents pursue Geralt and Ciri?",
               "What is the end of the end?"
                ]
    passages = ["[Personality and traits] At the same time, he was far from being devoid of attachment towards his allies and comrades: He was visibly saddened by the insanity of Alice and Frank Longbottom, noting how being dead would have be better than having to live the rest of their lives insane, and openly acknowledged he was never able to find it easy to get over the loss of a comrade and only by turning the sadness he felt into motivation to get justice was he able to move on, as seen by his expressing sympathy towards Jacob's sibling after they lost Rowan Khanna and even acknowledging he should have trained them well enough.",
                """"The Witcher" (Polish: "Wiedźmin") is a short story written by Andrzej Sapkowski, having first been published in the "Fantastyka" magazine and later in the now obsolete book, "Wiedźmin" before being re-published in . It introduces the witcher Geralt and his famous fight with a striga. 21.335 2492 The Witcher (Polish: "Cykl wiedźmiński") by Andrzej Sapkowski is a series of fantasy short stories (collected in two books, except for two stories) and five novels about the witcher Geralt of Rivia. The books have been adapted into a movie and two television series ("The Hexer" and ), a video game series (), a comic book and others. The novel series (excluding the short stories) is also called the Witcher Saga (Polish: "saga o wiedźminie") or the Blood of the Elves saga.""",
                 """The Witcher" follows the story of Geralt of Rivia, a witcher: a traveling monster hunter for hire, gifted with unnatural powers. Taking place in a fictional medieval world, the game implements detailed visuals. The natural light during various phases of the day is realistically altered, and the day and night transitions serve to enrich the game's ambiance. The weather can dynamically change from a light drizzle to a dark, stormy downpour accompanied by thunder and lightning, and the NPCs react to the rain by hiding under roofs trying to get out of the rain.""",
                 """The Witcher" follows the story of Geralt of Rivia, a witcher: a traveling monster hunter for hire, gifted with unnatural powers. Taking place in a fictional medieval world, the game implements detailed visuals. The natural light during various phases of the day is realistically altered, and the day and night transitions serve to enrich the game's ambiance. The weather can dynamically change from a light drizzle to a dark, stormy downpour accompanied by thunder and lightning, and the NPCs react to the rain by hiding under roofs trying to get out of the rain.""",
                "The name of the red car is Gerald and it is very fast.",
                "[Interwar Period] In , he recruited Dandelion as his informant and asked Yennefer to protect the bard when he was captured by Rience in a town near Bleobheris. The sorceress saved Dandelion and ordered him to hide under Dijkstra's wing. Later, Sigismund managed to calm the tension when Geralt and Olsen killed Temerian agents who, while they were legit agents, had at the time been acting on their own in pursuit of Geralt and Ciri to try and claim Rience's reward for the pair. Together with Philippa, the spymaster asked Dandelion about Geralt's current whereabouts; Dijkstra was surprised when Philippa mentioned Ciri as well, perceiving it as a hasty move.",
                "[Interwar Period] In , he recruited Dandelion as his informant and asked Yennefer to protect the bard when he was captured by Rience in a town near Bleobheris. The sorceress saved Dandelion and ordered him to hide under Dijkstra's wing. Later, Sigismund managed to calm the tension when Geralt and Olsen killed Temerian agents who, while they were legit agents, had at the time been acting on their own in pursuit of Geralt and Ciri to try and claim Rience's reward for the pair. Together with Philippa, the spymaster asked Dandelion about Geralt's current whereabouts; Dijkstra was surprised when Philippa mentioned Ciri as well, perceiving it as a hasty move.",
                "[Interwar Period] In , he recruited Dandelion as his informant and asked Yennefer to protect the bard when he was captured by Rience in a town near Bleobheris. The sorceress saved Dandelion and ordered him to hide under Dijkstra's wing. Later, Sigismund managed to calm the tension when Geralt and Olsen killed Temerian agents who, while they were legit agents, had at the time been acting on their own in pursuit of Geralt and Ciri to try and claim Rience's reward for the pair. Together with Philippa, the spymaster asked Dandelion about Geralt's current whereabouts; Dijkstra was surprised when Philippa mentioned Ciri as well, perceiving it as a hasty move.",
                "start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start end"

                ]

    #get best k tokens per query vectory (32)

    k = 2
    query_indices = [0,1,2,3,4,5,6,7,8]
    html_heatmap_kde_list, html_heatmap_absolute_list, html_added_values_list = [], [], []
    for query_index in query_indices:
        print(query_index)
        topk_cossim_indices_list, topk_cossim_values_list = get_topk_cossim_indices_and_values(k, inference,
                                                                            queries[query_index], passages[query_index])

        #flatten lists:
        flat_topk_cossim_indices_list = [item for sublist in topk_cossim_indices_list for item in sublist]
        flat_topk_cossim_values_list = [item for sublist in topk_cossim_values_list for item in sublist]

        #get tokens
        tokens = np.array(tokenizer.tokenize(passages[query_index], "doc", add_special_tokens=True))
        print(tokens.shape, len(topk_cossim_indices_list))
        #print(passage_embeddings.shape)

        #check whether there are atleast two different tokens in the topk_indices
        if all(x == flat_topk_cossim_indices_list[0] for x in flat_topk_cossim_indices_list):
            print("All elements in list are equal.")
            print("The only relevant Token is:", tokens[flat_topk_cossim_indices_list[0]])
        else:
            print(k,"* 32 = ", len(flat_topk_cossim_indices_list), "datapoints used")
            print("Max topk index:", max(flat_topk_cossim_indices_list))
            print("len(tokens):", len(tokens))
            #create heatmaps
            html_heatmap_kde, html_heatmap_absolute, html_added_values = html_heatmap(tokens, flat_topk_cossim_indices_list
                                                                                      , flat_topk_cossim_values_list, True, '#')
            html_heatmap_kde_list.append(html_heatmap_kde)
            html_heatmap_absolute_list.append(html_heatmap_absolute)
            html_added_values_list.append(html_added_values)


    # combined
    # clear
    f = open('combined_heatmaps.html', 'w')
    f.close()
    # write
    f = open('combined_heatmaps.html', 'a')
    for query_index in range(0, len(queries)):
        f.write('<h2>'+ queries[query_index] +'</h2>')
        f.write('<h3> kde </h3>')
        f.write(html_heatmap_kde_list[query_index])
        f.write('<h3> absolute count </h3>')
        f.write(html_heatmap_absolute_list[query_index])
        f.write('<h3> added values </h3>')
        f.write(html_added_values_list[query_index])
    f.close()
            
