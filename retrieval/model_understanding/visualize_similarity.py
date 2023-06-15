import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.stats import gaussian_kde

from retrieval.models import ColBERTInference, load_colbert_and_tokenizer



MODEL_PATH = "../../data/colbertv2.0/"  # "../../../data/colbertv2.0/" or "bert-base-uncased" or "roberta-base"


def rgbk_to_hex(r, g, b,k):
    return '#{:02x}{:02x}{:02x}{:02x}'.format(int(r*255), int(g*255), int(b*255), int(k*255))


def highlighter(word, normed_value, cmap):
    color = cmap(normed_value)
    color_hex = rgbk_to_hex(color[0],color[1],color[2],color[3])
    word = '<span style="background-color:' +color_hex+ '">' +word+ '</span>'
    return word


def html_heatmap(tokens, topk_indices, topk_values, prefix='##', plot=False, store=False):
    '''Args:    tokens:                     all tokens of the passage in their order
                max_cossim_indices_list     32 (64) indices of the tokens with the highest cosine similarity with the
                                            32 (64) query vectors (duplicates necessary)
    '''
    cmap = matplotlib.colormaps["Blues"]
    topk_indices, topk_values = topk_indices.tolist(), topk_values.tolist()

    #remove ##-prefix
    stripped_tokens = []
    for token in tokens:
        if token != prefix:
            if token.startswith(prefix):
                token = token[len(prefix):]
            else:
                token = " " + token
        stripped_tokens.append(token)

    # ignore the first two and last token for kde
    kde_topk_indices = [i for i in topk_indices if i > 1 and i < (len(tokens)-1)]

    # calculate kernel density
    kernel = gaussian_kde(kde_topk_indices, bw_method="scott")
    xgrid = np.arange(len(tokens))
    density = kernel(xgrid)
    ymin, ymax = min(density), max(density)

    # kde heatmap
    norm_kde = Normalize(vmin=ymin, vmax=1.66 * ymax)
    html_heatmap_kde = ''.join([highlighter(stripped_tokens[i], norm_kde(kernel(i))[0], cmap) for i in range(0, len(stripped_tokens))]).strip()

    # count heatmap
    count_ind = [topk_indices.count(i) for i in range(0, len(tokens))]
    norm_count = Normalize(vmin=0, vmax=1.66 * max(count_ind))
    html_heatmap_count = ''.join([highlighter(stripped_tokens[i], norm_count(count_ind[i]), cmap)
                                      for i in range(0, len(stripped_tokens))]).strip()

    # cummulated similarities heatmap
    sum_values = np.zeros(len(stripped_tokens))
    for idx, value in zip(topk_indices, topk_values):
        sum_values[idx] += value
    norm_cum = Normalize(vmin=0, vmax=1.66 * max(sum_values))
    html_heatmap_sum = ''.join([highlighter(stripped_tokens[i], norm_cum(sum_values[i]), cmap)
                                      for i in range(0, len(tokens))]).strip()

    if plot:
        kernel_scott = gaussian_kde(kde_topk_indices, bw_method="scott") 
        kernel_silverman = gaussian_kde(kde_topk_indices, bw_method="silverman")
        plt.plot(xgrid, kernel_scott(xgrid), label="scott")
        plt.plot(xgrid, kernel_silverman(xgrid), label="silverman")
        plt.legend()
        plt.show()
    
    if store:
        #create html file with heatmapped text
        with open('heatmap_kde.html', 'w') as f:
            f.write(html_heatmap_kde)
        with open('heatmap_count.html', 'w') as f:
            f.write(html_heatmap_count)
        with open('heatmap_sum.html', 'w') as f:
            f.write(html_heatmap_sum)

    return html_heatmap_kde, html_heatmap_count, html_heatmap_sum


def get_topk_token(colbert_inf: ColBERTInference, query: str, passage: str, k=1, similarity="cosine"):
    Q = inference.query_from_text(query) # shape: (L_q, D)
    P = inference.doc_from_text(passage) # shape: (L_d, D)
    if similarity == "cosine":
        sim = Q @ P.T
    elif similarity == "l2":
        sim = (Q.unsqueeze(-2) - P.unsqueeze(-3)).pow(2).sum(dim=-1).sqrt()
    
    topk_sim = sim.topk(k=k, dim=-1)
    values, indicies = topk_sim
    values, indicies  = values.cpu().numpy(), indicies.cpu().numpy()

    return values, indicies


def visualize(colbert_inf: ColBERTInference, query: str, passage: str, k=1, similarity="cosine"):
    topk_token_sim, topk_token_idx = get_topk_token(inference, query, passage, k)
    topk_token_idx, topk_token_sim = topk_token_idx.flatten(), topk_token_sim.flatten()

    # get tokens
    tokens = np.array(inference.tokenizer.tokenize(passage, "doc", add_special_tokens=True))
    print(tokens.shape, topk_token_idx.shape, k)

    #check whether there are atleast two different tokens in the topk_indices
    if np.all(topk_token_idx == topk_token_idx.flat[0]):
        print("All elements in list are equal.")
        print("The only relevant Token is:", tokens[topk_token_idx.flat[0]])
        return

    print(k,"* 32 = ", topk_token_idx.size, "datapoints used")
    print("Max topk index:", topk_token_idx.max())
    print("len(tokens):", topk_token_idx.shape)

    #create heatmaps
    kde_heatmap, count_heatmap, sum_heatmap = html_heatmap(tokens, topk_token_idx, topk_token_sim, plot=False, store=True)
    
    return kde_heatmap, count_heatmap, sum_heatmap


if __name__ == "__main__":

    queries = [
        "How did Moody feel about the insanity of Alice and Frank Longbottom?",
        'Who is the author of ''The Witcher''?',
        'How does an NPC react if it starts raining?',
        'How does an NPC behave if it starts raining?',
        'What is the name of the red car?',
        "Who did Sigismund recruit as his informant during the interwar period?",
        "What did Yennefer do to protect Dandelion when he was captured by Rience?",
        "Why did Temerian agents pursue Geralt and Ciri?",
        "What is the end of the end?",
        "Who won the football championship in 2006?"
    ]


    passages = ["[Personality and traits] At the same time, he was far from being devoid of attachment towards his allies and comrades: He was visibly saddened by the insanity of Alice and Frank Longbottom, noting how being dead would have be better than having to live the rest of their lives insane, and openly acknowledged he was never able to find it easy to get over the loss of a comrade and only by turning the sadness he felt into motivation to get justice was he able to move on, as seen by his expressing sympathy towards Jacob's sibling after they lost Rowan Khanna and even acknowledging he should have trained them well enough.",
                """"The Witcher" (Polish: "Wiedźmin") is a short story written by Andrzej Sapkowski, having first been published in the "Fantastyka" magazine and later in the now obsolete book, "Wiedźmin" before being re-published in . It introduces the witcher Geralt and his famous fight with a striga. 21.335 2492 The Witcher (Polish: "Cykl wiedźmiński") by Andrzej Sapkowski is a series of fantasy short stories (collected in two books, except for two stories) and five novels about the witcher Geralt of Rivia. The books have been adapted into a movie and two television series ("The Hexer" and ), a video game series (), a comic book and others. The novel series (excluding the short stories) is also called the Witcher Saga (Polish: "saga o wiedźminie") or the Blood of the Elves saga.""",
                 """The Witcher" follows the story of Geralt of Rivia, a witcher: a traveling monster hunter for hire, gifted with unnatural powers. Taking place in a fictional medieval world, the game implements detailed visuals. The natural light during various phases of the day is realistically altered, and the day and night transitions serve to enrich the game's ambiance. The weather can dynamically change from a light drizzle to a dark, stormy downpour accompanied by thunder and lightning, and the NPCs react to the rain by hiding under roofs trying to get out of the rain.""",
                 """The Witcher" follows the story of Geralt of Rivia, a witcher: a traveling monster hunter for hire, gifted with unnatural powers. Taking place in a fictional medieval world, the game implements detailed visuals. The natural light during various phases of the day is realistically altered, and the day and night transitions serve to enrich the game's ambiance. The weather can dynamically change from a light drizzle to a dark, stormy downpour accompanied by thunder and lightning, and the NPCs react to the rain by hiding under roofs trying to get out of the rain.""",
                "The name of the red car is Gerald and it is very fast.",
                "[Interwar Period] In , he recruited Dandelion as his informant and asked Yennefer to protect the bard when he was captured by Rience in a town near Bleobheris. The sorceress saved Dandelion and ordered him to hide under Dijkstra's wing. Later, Sigismund managed to calm the tension when Geralt and Olsen killed Temerian agents who, while they were legit agents, had at the time been acting on their own in pursuit of Geralt and Ciri to try and claim Rience's reward for the pair. Together with Philippa, the spymaster asked Dandelion about Geralt's current whereabouts; Dijkstra was surprised when Philippa mentioned Ciri as well, perceiving it as a hasty move.",
                "[Interwar Period] In , he recruited Dandelion as his informant and asked Yennefer to protect the bard when he was captured by Rience in a town near Bleobheris. The sorceress saved Dandelion and ordered him to hide under Dijkstra's wing. Later, Sigismund managed to calm the tension when Geralt and Olsen killed Temerian agents who, while they were legit agents, had at the time been acting on their own in pursuit of Geralt and Ciri to try and claim Rience's reward for the pair. Together with Philippa, the spymaster asked Dandelion about Geralt's current whereabouts; Dijkstra was surprised when Philippa mentioned Ciri as well, perceiving it as a hasty move.",
                "[Interwar Period] In , he recruited Dandelion as his informant and asked Yennefer to protect the bard when he was captured by Rience in a town near Bleobheris. The sorceress saved Dandelion and ordered him to hide under Dijkstra's wing. Later, Sigismund managed to calm the tension when Geralt and Olsen killed Temerian agents who, while they were legit agents, had at the time been acting on their own in pursuit of Geralt and Ciri to try and claim Rience's reward for the pair. Together with Philippa, the spymaster asked Dandelion about Geralt's current whereabouts; Dijkstra was surprised when Philippa mentioned Ciri as well, perceiving it as a hasty move.",
                "start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start start end",
                "The football championship in the year 2006 was a great sports event that was won by Italy."
                ]

    

    colbert, tokenizer = load_colbert_and_tokenizer(MODEL_PATH)
    inference = ColBERTInference(colbert, tokenizer)

    K = 2
    kde_heatmaps, count_heatmaps, sum_heatmaps = [], [], []

    for query, passage in zip(queries, passages):
        kde_heatmap, count_heatmap, sum_heatmap = visualize(inference, query, passage, k=K, similarity="cosine")

        kde_heatmaps.append(kde_heatmap)
        count_heatmaps.append(count_heatmap)
        sum_heatmaps.append(sum_heatmap)

    with open('combined_heatmaps.html', 'w') as f:
        for i in range(len(queries)):
            f.write('<h2>'+ queries[i] +'</h2>')
            f.write('<h3> kde </h3>')
            f.write(kde_heatmaps[i])
            f.write('<h3> absolute count </h3>')
            f.write(count_heatmaps[i])
            f.write('<h3> added values </h3>')
            f.write(sum_heatmaps[i])          
