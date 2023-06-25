import random
import torch
from operator import itemgetter

from tqdm import tqdm
from retrieval.configs import BaseConfig
from retrieval.indexing import ColBERTIndexer, ColBERTRetriever
from retrieval.data import TripleDataset, Passages
from retrieval.models import TfIdf, load_colbert_and_tokenizer
from retrieval.model_understanding.visualize_similarity import *


BSIZE=1
SEED = 125
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

CHECKPOINT_PATH = "../../data/colbertv2.0/"  # "../../../data/colbertv2.0/" or "bert-base-uncased" or "roberta-base"
INDEX_PATH = "../../data/fandoms_qa/fandoms_all/human_verified/final/el/all/passages.index.pt"

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"



def colbert_vs_tf_idf(testing_max_count = 100, size_datasets_good = 100, size_datasets_bad = 100, K_good=1000, K_bad = 1000, return_size=20):
    '''Args:
    testing_max_count: how many
    '''

    dataset = TripleDataset(BaseConfig(passages_per_query=10),
                            triples_path="../../data/fandoms_qa/fandoms_all/human_verified/final/el/all/triples.tsv",
                            queries_path="../../data/fandoms_qa/fandoms_all/human_verified/final/el/all/queries.tsv",
                            passages_path="../../data/fandoms_qa/fandoms_all/human_verified/final/el/all/passages.tsv",
                            mode="QQP")


    colbert, tokenizer = load_colbert_and_tokenizer(CHECKPOINT_PATH)
    inference = ColBERTInference(colbert, tokenizer)
    retriever = ColBERTRetriever(inference, device=DEVICE, passages=dataset.passages)

    #precompute indicies
    retriever.indexer.dtype = torch.float32
    # data = dataset.passages.values().tolist()
    # pids = dataset.passages.keys().tolist()
    # retriever.indexer.index(data, pids, bsize=8)
    # retriever.indexer.save(INDEX_PATH)
    retriever.indexer.load(INDEX_PATH)

    #print([x for x in dataset.passages_items()])
    tf_idf = TfIdf(
            passages=dataset.passages.values(),
            mapping_rowInd_pid=dict(enumerate(dataset.passages.keys())),
    )
    print(tf_idf.best_and_worst_pids(["Who is gerald of riva?"], 5, 10))
    good_pairs_cb = {}
    good_pairs_tf_idf = {}
    bad_pairs_cb = {}
    bad_pairs_tf_idf = {}
    query_batch = []
    target_batch = []
    testing_count = 0
    num_passages = len(dataset.passages)
    print(num_passages)
    for i, triple in enumerate(tqdm(dataset)):
        testing_count += 1
        print(testing_count)
        if testing_count >= testing_max_count:
            break

        # for QPP datasets:
        # qid, pid_pos, *pid_neg = triple
        # query, psg_pos, *psg_neg = dataset.id2string(triple)
        # query_batch.append(query)
        # target_batch.append(pid_pos)

        # for QQP datasets:
        qid_pos, qid_neg, pid_pos = triple
        query_pos, query_neg, passage = dataset.id2string(triple)
        query_batch.append(query_pos)
        target_batch.append(pid_pos)

        if len(query_batch) == BSIZE or i + 1 == len(dataset):
            #pids, worst_pids = retriever.best_and_worst_pid_retrieval(query_batch, K_good)
            ##print(query_batch)
            pids = [x for (y,x) in retriever.full_retrieval(query_batch, K_good)]
            #pids_tf_idf, worst_pids_tft_idf = list(tf_idf.best_and_worst_pids(query_batch, K_good, K_bad))
            pids_tf_idf = list(tf_idf.answerQuestion(query_batch, K_good))

            # COLBERT
            for i, (pred_pids, target_pit, query) in enumerate(zip(pids, target_batch, query_batch)):
                pred_pids = pred_pids.tolist()
                if target_pit in pred_pids:
                    if len(good_pairs_cb.keys()) < size_datasets_good:
                        t = (query, target_pit)
                        # print(target_pit, pred_pids)
                        good_pairs_cb[t] = pred_pids.index(target_pit)
                    elif pred_pids.index(target_pit) < max(set(good_pairs_cb.values()), default=K_good):
                        t = (query, target_pit)
                        good_pairs_cb[t] = pred_pids.index(target_pit)
                    elif pred_pids.index(target_pit) > min(set(bad_pairs_cb.values()), default=0):
                        t = (query, target_pit)
                        good_pairs_cb[t] = pred_pids.index(target_pit)

                    if len(good_pairs_cb.keys()) > size_datasets_good:
                        good_pairs_cb.pop(max(good_pairs_cb, key=good_pairs_cb.get))
                    if len(bad_pairs_cb.keys()) > size_datasets_bad:
                        bad_pairs_cb.pop(min(bad_pairs_cb, key=bad_pairs_cb.get))
                else:
                    bad_pairs_cb[(query, target_pit)] = K_good
                    if len(bad_pairs_cb.keys()) > size_datasets_bad:
                        bad_pairs_cb.pop(min(bad_pairs_cb, key=bad_pairs_cb.get))

            #TF IDF
            for i, (pred_pids, target_pit, query) in enumerate(zip(pids_tf_idf, target_batch, query_batch)):
                pred_pids = pred_pids.tolist()
                if target_pit in pred_pids:
                    if len(good_pairs_tf_idf.keys()) < size_datasets_good:
                        t = (query, target_pit)
                        # print(target_pit, pred_pids)
                        good_pairs_tf_idf[t] = pred_pids.index(target_pit)
                    elif pred_pids.index(target_pit) < max(set(good_pairs_tf_idf.values()), default=K_good):
                        t = (query, target_pit)
                        good_pairs_tf_idf[t] = pred_pids.index(target_pit)
                    elif pred_pids.index(target_pit) > min(set(bad_pairs_tf_idf.values()), default=0):
                        t = (query, target_pit)
                        good_pairs_tf_idf[t] = pred_pids.index(target_pit)

                    if len(good_pairs_tf_idf.keys()) > size_datasets_good:
                        good_pairs_tf_idf.pop(max(good_pairs_tf_idf, key=good_pairs_tf_idf.get))
                    if len(bad_pairs_tf_idf.keys()) > size_datasets_bad:
                        bad_pairs_tf_idf.pop(min(bad_pairs_tf_idf, key=bad_pairs_tf_idf.get))
                else:
                    bad_pairs_tf_idf[(query, target_pit)] = K_good
                    if len(bad_pairs_tf_idf.keys()) > size_datasets_bad:
                        bad_pairs_tf_idf.pop(min(bad_pairs_tf_idf, key=bad_pairs_tf_idf.get))

            query_batch = []
            target_batch = []

    keys_tf_good_cb_good = set(list(set(good_pairs_cb.keys()).intersection(good_pairs_tf_idf.keys()))[:return_size])
    keys_tf_good_cb_bad = set(list(set(bad_pairs_cb.keys()).intersection(good_pairs_tf_idf.keys()))[:return_size])
    keys_tf_bad_cb_good = set(list(set(good_pairs_cb.keys()).intersection(bad_pairs_tf_idf.keys()))[:return_size])
    keys_tf_bad_cb_bad = set(list(set(bad_pairs_cb.keys()).intersection(bad_pairs_tf_idf.keys()))[:return_size])
    print("tf idf", set(good_pairs_tf_idf.keys()).intersection(bad_pairs_tf_idf.keys()))
    print("cb", set(good_pairs_cb.keys()).intersection(bad_pairs_cb.keys()))

    dict_tf_good_cb_good = dict(((k[0], dataset.passages[k[1]]), (good_pairs_cb[k], good_pairs_tf_idf[k])) for k in keys_tf_good_cb_good)
    dict_tf_good_cb_bad = dict(((k[0], dataset.passages[k[1]]), (bad_pairs_cb[k], good_pairs_tf_idf[k])) for k in keys_tf_good_cb_bad)
    dict_tf_bad_cb_good = dict(((k[0], dataset.passages[k[1]]), (good_pairs_cb[k], bad_pairs_tf_idf[k])) for k in keys_tf_bad_cb_good)
    dict_tf_bad_cb_bad = dict(((k[0], dataset.passages[k[1]]), (bad_pairs_cb[k], bad_pairs_tf_idf[k])) for k in keys_tf_bad_cb_bad)
    dict_cb_bad = bad_pairs_cb
    return dict_tf_good_cb_good, \
        dict_tf_good_cb_bad, \
        dict_tf_bad_cb_good, \
        dict_tf_bad_cb_bad, \
        dict_cb_bad


def colbert_vs_tf_idf2(dataset, cb_retriever, tf_idf, testing_max_count, K):

    testing_count = 0

    cb_tf_idf_list = []

    for i, triple in enumerate(tqdm(dataset)):
        testing_count += 1
        if testing_count >= testing_max_count:
            break

        # for QQP datasets:
        qid_pos, qid_neg, pid_pos = triple
        query_pos, query_neg, passage = dataset.id2string(triple)

        pred_pids = [x for (y,x) in cb_retriever.full_retrieval([query_pos], K)][0].tolist()
        pred_pids_tf_idf = list(tf_idf.answerQuestion([query_pos], K))[0].tolist()

        if pid_pos in pred_pids:
            index_cb = pred_pids.index(pid_pos)
        else:
            index_cb = K

        if pid_pos in pred_pids_tf_idf:
            index_tf_idf = pred_pids_tf_idf.index(pid_pos)
        else:
            index_tf_idf = K

        difference = index_tf_idf - index_cb
        entry = (query_pos, pid_pos, pred_pids[0], pred_pids_tf_idf[0], index_cb, index_tf_idf, difference)
        cb_tf_idf_list.append(entry)


    return cb_tf_idf_list


if __name__ == "__main__":
    # sets = colbert_vs_tf_idf(size_datasets_good = 100, size_datasets_bad = 100, testing_max_count=100_000, K_good=500, return_size=10)
    # print("tf_good_cb_good, size:", len(sets[0].keys()), sets[0])
    # print("tf_good_cb_bad, size:", len(sets[1].keys()), sets[1])
    # print("tf_bad_cb_good, size:", len(sets[2].keys()), sets[2])
    # print("tf_bad_cb_bad, size:", len(sets[3].keys()), sets[3])
    # print("cb_bad", sets[4])
    dataset = TripleDataset(BaseConfig(passages_per_query=10),
                            triples_path="../../data/fandoms_qa/fandoms_all/human_verified/final/el/all/triples.tsv",
                            queries_path="../../data/fandoms_qa/fandoms_all/human_verified/final/el/all/queries.tsv",
                            passages_path="../../data/fandoms_qa/fandoms_all/human_verified/final/el/all/passages.tsv",
                            mode="QQP")

    colbert, tokenizer = load_colbert_and_tokenizer(CHECKPOINT_PATH)
    inference = ColBERTInference(colbert, tokenizer)
    retriever = ColBERTRetriever(inference, device=DEVICE, passages=dataset.passages)

    # precompute indicies
    retriever.indexer.dtype = torch.float32
    # data = dataset.passages.values().tolist()
    # pids = dataset.passages.keys().tolist()
    # retriever.indexer.index(data, pids, bsize=8)
    # retriever.indexer.save(INDEX_PATH)
    retriever.indexer.load(INDEX_PATH)

    tf_idf = TfIdf(
        passages=dataset.passages.values(),
        mapping_rowInd_pid=dict(enumerate(dataset.passages.keys())),
    )

    cb_tf_idf_list = colbert_vs_tf_idf2(dataset, retriever, tf_idf, testing_max_count=100_000, K=1000)
    print(cb_tf_idf_list[:10])
    print(len(cb_tf_idf_list))

    #cb_good = sorted(cb_tf_idf_list, reverse=False, key=itemgetter(-3))
    cb_bad = sorted(cb_tf_idf_list, reverse=True, key=itemgetter(-3))
    # tf_idf_good = sorted(cb_tf_idf_list, reverse=False, key=itemgetter(-2))
    # tf_idf_bad = sorted(cb_tf_idf_list, reverse=True, key=itemgetter(-2))
    # cb_good_tf_idf_bad = sorted(cb_good, reverse=True, key=itemgetter(-1))[:5]
    print(cb_bad[:2])


