from retrieval.models import ColBERTInference, load_colbert_and_tokenizer
from retrieval.model_understanding.visualize_similarity import visualize
from retrieval.model_understanding.tf_idf_vs_colbert import colbert_vs_tf_idf
from operator import itemgetter
from retrieval.configs import BaseConfig
from retrieval.indexing import ColBERTIndexer, ColBERTRetriever
from retrieval.data import TripleDataset, Passages
from retrieval.models import TfIdf, load_colbert_and_tokenizer
from retrieval.model_understanding.visualize_similarity import *
from retrieval.model_understanding.tf_idf_vs_colbert import colbert_vs_tf_idf2

import torch

CHECKPOINT_PATH = "../../data/colbertv2.0/"  # "../../../data/colbertv2.0/" or "bert-base-uncased" or "roberta-base"
INDEX_PATH = "../../data/fandoms_qa/fandoms_all/human_verified/final/witc/all/passages.idx"

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
OUTPUT_COUNT = 20
K = 2
MODE = "ONEWORD"
FILENAME = 'combined_heatmaps_first_word.html'
WORD_INDEX = 0

if __name__ == "__main__":
    CHECKPOINT_PATH = "../../data/colbertv2.0/"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    colbert, tokenizer = load_colbert_and_tokenizer(CHECKPOINT_PATH)
    tokenizer.doc_maxlen = 512
    colbert.config.skip_punctuation = False
    colbert.skiplist = None
    inference = ColBERTInference(colbert, tokenizer, device=DEVICE)
    #retriever.to(device="cpu", dtype=torch.float32)

    dataset = TripleDataset(BaseConfig(passages_per_query=10),
                            triples_path="../../data/fandoms_qa/fandoms_all/human_verified/final/witc/all/triples.tsv",
                            queries_path="../../data/fandoms_qa/fandoms_all/human_verified/final/witc/all/queries.tsv",
                            passages_path="../../data/fandoms_qa/fandoms_all/human_verified/final/witc/all/passages.tsv",
                            mode="QQP")

    retriever = ColBERTRetriever(inference, device=DEVICE, passages=dataset.passages)
    # precompute indicies
    #retriever.indexer.dtype = torch.float32

    print(retriever.indexer.dtype)
    # data = dataset.passages.values().tolist()
    # pids = dataset.passages.keys().tolist()
    # retriever.indexer.index(data, pids, bsize=8)
    # retriever.indexer.save(INDEX_PATH)
    retriever.indexer.load(INDEX_PATH)
    retriever.to(device="cpu", dtype=torch.float32)
    tf_idf = TfIdf(
        passages=dataset.passages.values(),
        mapping_rowInd_pid=dict(enumerate(dataset.passages.keys())),
    )

    cb_tf_idf_list = colbert_vs_tf_idf2(dataset, retriever, tf_idf, testing_max_count=100_000_000, K=5000)
    # print(cb_tf_idf_list[:10])
    # print(len(cb_tf_idf_list))
    #print(cb_tf_idf_list)
    #print(len(cb_tf_idf_list))

    with open(FILENAME, 'w', encoding="utf-8") as f:
        cb_sorted = sorted(cb_tf_idf_list, reverse=False, key=itemgetter(-3))
        f.write('<h1> cb_good </h1>')
        cb_good = cb_sorted[:OUTPUT_COUNT]
        for tuple in cb_good:
            query = tuple[0]
            passage = dataset.passages[tuple[1]]
            if not MODE == "ONEWORD":
                kde_heatmap, count_heatmap, sum_heatmap = visualize(inference, query, passage, k=K, similarity="cosine")
            else:
                kde_heatmap, count_heatmap, sum_heatmap = visualize_for_specific_word_in_question(inference, query, passage,
                                                                                              word_index=WORD_INDEX)
            f.write('<h2>' + str(query) + " cb: " + str(tuple[4]) + " tf_idf: " + str(tuple[5]) + '</h2>')
            f.write('<h2> Correct Passage </h2>')
            f.write('<h3> kde </h3>')
            f.write(kde_heatmap)
            f.write('<h3> absolute count </h3>')
            f.write(count_heatmap)
            f.write('<h3> added values </h3>')
            f.write(sum_heatmap)


            query = tuple[0]
            passage = dataset.passages[tuple[2]]
            f.write('<h2> Predicted Passage COLBERT </h2>')
            if not MODE == "ONEWORD":
                kde_heatmap, count_heatmap, sum_heatmap = visualize(inference, query, passage, k=K, similarity="cosine")
            else:
                kde_heatmap, count_heatmap, sum_heatmap = visualize_for_specific_word_in_question(inference, query,
                                                                                                  passage,
                                                                                                  word_index=0)
            f.write('<h3> added values </h3>')
            f.write(sum_heatmap)

            query = tuple[0]
            passage = dataset.passages[tuple[3]]
            f.write('<h2> Predicted Passage TF IDF </h2>')
            if not MODE == "ONEWORD":
                kde_heatmap, count_heatmap, sum_heatmap = visualize(inference, query, passage, k=K, similarity="cosine")
            else:
                kde_heatmap, count_heatmap, sum_heatmap = visualize_for_specific_word_in_question(inference, query,
                                                                                                  passage,
                                                                                                  word_index=0)
            f.write('<h3> added values </h3>')
            f.write(sum_heatmap)

        f.write('<h1> cb_bad </h1>')
        cb_bad = cb_sorted[-OUTPUT_COUNT:]
        for tuple in reversed(cb_bad):
            query = tuple[0]
            passage = dataset.passages[tuple[1]]
            if not MODE == "ONEWORD":
                kde_heatmap, count_heatmap, sum_heatmap = visualize(inference, query, passage, k=K, similarity="cosine")
            else:
                kde_heatmap, count_heatmap, sum_heatmap = visualize_for_specific_word_in_question(inference, query,
                                                                                                  passage,
                                                                                                  word_index=0)
            f.write('<h2>' + str(query) + " cb: " + str(tuple[4]) + " tf_idf: " + str(tuple[5]) + '</h2>')
            f.write('<h3> kde </h3>')
            f.write(kde_heatmap)
            f.write('<h3> absolute count </h3>')
            f.write(count_heatmap)
            f.write('<h3> added values </h3>')
            f.write(sum_heatmap)


            query = tuple[0]
            passage = dataset.passages[tuple[2]]
            f.write('<h2> Predicted Passage COLBERT </h2>')
            if not MODE == "ONEWORD":
                kde_heatmap, count_heatmap, sum_heatmap = visualize(inference, query, passage, k=K, similarity="cosine")
            else:
                kde_heatmap, count_heatmap, sum_heatmap = visualize_for_specific_word_in_question(inference, query,
                                                                                                  passage,
                                                                                                  word_index=0)
            f.write('<h3> added values </h3>')
            f.write(sum_heatmap)

            query = tuple[0]
            passage = dataset.passages[tuple[3]]
            f.write('<h2> Predicted Passage TF IDF </h2>')
            if not MODE == "ONEWORD":
                kde_heatmap, count_heatmap, sum_heatmap = visualize(inference, query, passage, k=K, similarity="cosine")
            else:
                kde_heatmap, count_heatmap, sum_heatmap = visualize_for_specific_word_in_question(inference, query,
                                                                                                  passage,
                                                                                                  word_index=0)
            f.write('<h3> added values </h3>')
            f.write(sum_heatmap)

        tf_idf_sorted = sorted(cb_tf_idf_list, reverse=False, key=itemgetter(-2))
        f.write('<h1> tf_idf_good </h1>')
        cb_good = tf_idf_sorted[:OUTPUT_COUNT]
        for tuple in cb_good:
            query = tuple[0]
            passage = dataset.passages[tuple[1]]
            if not MODE == "ONEWORD":
                kde_heatmap, count_heatmap, sum_heatmap = visualize(inference, query, passage, k=K, similarity="cosine")
            else:
                kde_heatmap, count_heatmap, sum_heatmap = visualize_for_specific_word_in_question(inference, query,
                                                                                                  passage,
                                                                                                  word_index=0)
            f.write('<h2>' + str(query) + " cb: " + str(tuple[4]) + " tf_idf: " + str(tuple[5]) + '</h2>')
            f.write('<h3> kde </h3>')
            f.write(kde_heatmap)
            f.write('<h3> absolute count </h3>')
            f.write(count_heatmap)
            f.write('<h3> added values </h3>')
            f.write(sum_heatmap)


            query = tuple[0]
            passage = dataset.passages[tuple[2]]
            f.write('<h2> Predicted Passage COLBERT </h2>')
            if not MODE == "ONEWORD":
                kde_heatmap, count_heatmap, sum_heatmap = visualize(inference, query, passage, k=K, similarity="cosine")
            else:
                kde_heatmap, count_heatmap, sum_heatmap = visualize_for_specific_word_in_question(inference, query,
                                                                                                  passage,
                                                                                                  word_index=0)
            f.write('<h3> added values </h3>')
            f.write(sum_heatmap)

            query = tuple[0]
            passage = dataset.passages[tuple[3]]
            f.write('<h2> Predicted Passage TF IDF </h2>')
            if not MODE == "ONEWORD":
                kde_heatmap, count_heatmap, sum_heatmap = visualize(inference, query, passage, k=K, similarity="cosine")
            else:
                kde_heatmap, count_heatmap, sum_heatmap = visualize_for_specific_word_in_question(inference, query,
                                                                                                  passage,
                                                                                                  word_index=0)
            f.write('<h3> added values </h3>')
            f.write(sum_heatmap)

        f.write('<h1> tf_idf_bad </h1>')
        tf_idf_bad = tf_idf_sorted[-OUTPUT_COUNT:]
        for tuple in reversed(tf_idf_bad):
            query = tuple[0]
            passage = dataset.passages[tuple[1]]
            if not MODE == "ONEWORD":
                kde_heatmap, count_heatmap, sum_heatmap = visualize(inference, query, passage, k=K, similarity="cosine")
            else:
                kde_heatmap, count_heatmap, sum_heatmap = visualize_for_specific_word_in_question(inference, query,
                                                                                                  passage,
                                                                                                  word_index=0)
            f.write('<h2>' + str(query) + " cb: " + str(tuple[4]) + " tf_idf: " + str(tuple[5]) + '</h2>')
            f.write('<h3> kde </h3>')
            f.write(kde_heatmap)
            f.write('<h3> absolute count </h3>')
            f.write(count_heatmap)
            f.write('<h3> added values </h3>')
            f.write(sum_heatmap)


            query = tuple[0]
            passage = dataset.passages[tuple[2]]
            f.write('<h2> Predicted Passage COLBERT </h2>')
            if not MODE == "ONEWORD":
                kde_heatmap, count_heatmap, sum_heatmap = visualize(inference, query, passage, k=K, similarity="cosine")
            else:
                kde_heatmap, count_heatmap, sum_heatmap = visualize_for_specific_word_in_question(inference, query,
                                                                                                  passage,
                                                                                                  word_index=0)
            f.write('<h3> added values </h3>')
            f.write(sum_heatmap)

            query = tuple[0]
            passage = dataset.passages[tuple[3]]
            f.write('<h2> Predicted Passage TF IDF </h2>')
            if not MODE == "ONEWORD":
                kde_heatmap, count_heatmap, sum_heatmap = visualize(inference, query, passage, k=K, similarity="cosine")
            else:
                kde_heatmap, count_heatmap, sum_heatmap = visualize_for_specific_word_in_question(inference, query,
                                                                                                  passage,
                                                                                                  word_index=0)
            f.write('<h3> added values </h3>')
            f.write(sum_heatmap)

        difference_sorted = sorted(cb_tf_idf_list, reverse=False, key=itemgetter(-1))
        f.write('<h1> cb_good_tf_idf_bad </h1>')
        cb_good_tf_idf_bad = difference_sorted[-OUTPUT_COUNT:]
        for tuple in reversed(cb_good_tf_idf_bad):
            query = tuple[0]
            passage = dataset.passages[tuple[1]]
            if not MODE == "ONEWORD":
                kde_heatmap, count_heatmap, sum_heatmap = visualize(inference, query, passage, k=K, similarity="cosine")
            else:
                kde_heatmap, count_heatmap, sum_heatmap = visualize_for_specific_word_in_question(inference, query,
                                                                                                  passage,
                                                                                                  word_index=0)
            f.write('<h2>' + str(query) + " cb: " + str(tuple[4]) + " tf_idf: " + str(tuple[5]) + '</h2>')
            f.write('<h3> kde </h3>')
            f.write(kde_heatmap)
            f.write('<h3> absolute count </h3>')
            f.write(count_heatmap)
            f.write('<h3> added values </h3>')
            f.write(sum_heatmap)


            query = tuple[0]
            passage = dataset.passages[tuple[2]]
            f.write('<h2> Predicted Passage COLBERT </h2>')
            if not MODE == "ONEWORD":
                kde_heatmap, count_heatmap, sum_heatmap = visualize(inference, query, passage, k=K, similarity="cosine")
            else:
                kde_heatmap, count_heatmap, sum_heatmap = visualize_for_specific_word_in_question(inference, query,
                                                                                                  passage,
                                                                                                  word_index=0)
            f.write('<h3> added values </h3>')
            f.write(sum_heatmap)

            query = tuple[0]
            passage = dataset.passages[tuple[3]]
            f.write('<h2> Predicted Passage TF IDF </h2>')
            if not MODE == "ONEWORD":
                kde_heatmap, count_heatmap, sum_heatmap = visualize(inference, query, passage, k=K, similarity="cosine")
            else:
                kde_heatmap, count_heatmap, sum_heatmap = visualize_for_specific_word_in_question(inference, query,
                                                                                                  passage,
                                                                                                  word_index=0)
            f.write('<h3> added values </h3>')
            f.write(sum_heatmap)

        f.write('<h1> cb_bad_tf_idf_good </h1>')
        cb_bad_tf_idf_good = difference_sorted[:OUTPUT_COUNT]
        for tuple in cb_bad_tf_idf_good:
            query = tuple[0]
            passage = dataset.passages[tuple[1]]
            if not MODE == "ONEWORD":
                kde_heatmap, count_heatmap, sum_heatmap = visualize(inference, query, passage, k=K, similarity="cosine")
            else:
                kde_heatmap, count_heatmap, sum_heatmap = visualize_for_specific_word_in_question(inference, query,
                                                                                                  passage,
                                                                                                  word_index=0)
            f.write('<h2>' + str(query) + " cb: " + str(tuple[4]) + " tf_idf: " + str(tuple[5]) + '</h2>')
            f.write('<h3> kde </h3>')
            f.write(kde_heatmap)
            f.write('<h3> absolute count </h3>')
            f.write(count_heatmap)
            f.write('<h3> added values </h3>')
            f.write(sum_heatmap)


            query = tuple[0]
            passage = dataset.passages[tuple[2]]
            f.write('<h2> Predicted Passage COLBERT </h2>')
            if not MODE == "ONEWORD":
                kde_heatmap, count_heatmap, sum_heatmap = visualize(inference, query, passage, k=K, similarity="cosine")
            else:
                kde_heatmap, count_heatmap, sum_heatmap = visualize_for_specific_word_in_question(inference, query,
                                                                                                  passage,
                                                                                                  word_index=0)
            f.write('<h3> added values </h3>')
            f.write(sum_heatmap)

            query = tuple[0]
            passage = dataset.passages[tuple[3]]
            f.write('<h2> Predicted Passage TF IDF </h2>')
            if not MODE == "ONEWORD":
                kde_heatmap, count_heatmap, sum_heatmap = visualize(inference, query, passage, k=K, similarity="cosine")
            else:
                kde_heatmap, count_heatmap, sum_heatmap = visualize_for_specific_word_in_question(inference, query,
                                                                                                  passage,
                                                                                                  word_index=0)
            f.write('<h3> added values </h3>')
            f.write(sum_heatmap)
