
import torch
import random
import numpy as np
from tqdm import tqdm

from operator import itemgetter
from retrieval.indexing.colbert_retriever import ColBERTRetriever
from retrieval.configs import BaseConfig
from retrieval.data import Passages, Queries, TripleDataset, BucketIterator
from retrieval.models import ColBERTTokenizer, ColBERTInference, get_colbert_and_tokenizer
from retrieval.models.basemodels.tf_idf import TfIdf
from retrieval.indexing.visualize_similarity import *


BSIZE=1
SEED = 125
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

MODEL_PATH = "../../data/colbertv2.0/"  # "../../../data/colbertv2.0/" or "bert-base-uncased" or "roberta-base"
INDEX_PATH = "../../data/fandom-qa/witcher_qa/passages.train.indices.pt"

config = BaseConfig(
    tok_name_or_path=MODEL_PATH,
    backbone_name_or_path=MODEL_PATH,
    similarity="cosine",
    dim=128,
    batch_size=32,
    accum_steps=1,
)

def colbert_vs_tf_idf(testing_max_count = 100, size_datasets_good = 100, size_datasets_bad = 100, K_good=1000, K_bad = 1000, return_size=20):
    '''Args:
    testing_max_count: how many
    '''
    colbert, tokenizer = get_colbert_and_tokenizer(config)
    inference = ColBERTInference(colbert, tokenizer)
    retriever = ColBERTRetriever(inference, device="cuda:0")
    retriever.indexer.load(INDEX_PATH)

    dataset = TripleDataset(config,
                            triples_path="../../data/fandom-qa/witcher_qa/triples.train.tsv",
                            queries_path="../../data/fandom-qa/witcher_qa/queries.train.tsv",
                            passages_path="../../data/fandom-qa/witcher_qa/passages.train.tsv",
                            mode="qpp")

    tf_idf = TfIdf(folders=['../../data/fandom-qa/witcher_qa'])

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
        if testing_count >= testing_max_count:
            break

        qid, pid_pos, *pid_neg = triple
        query, psg_pos, *psg_neg = dataset.id2string(triple)
        query_batch.append(query)
        target_batch.append(pid_pos)

        if len(query_batch) == BSIZE or i + 1 == len(dataset):
            pids, worst_pids = retriever.best_and_worst_pid_retrieval(query_batch, K_good)
            pids_tf_idf, worst_pids_tft_idf = list(tf_idf.batchBestKPIDs(K_good, query_batch, best_and_worst = True))
            print(worst_pids_tft_idf)
            #worst_pids_tft_idf = list(tf_idf.batchBestKPIDs(K_bad, query_batch, best = False)[1])
            # print(worst_pids[1])
            # print(len(worst_pids[1]))
            # print(worst_pids_tft_idf)
            # print(len(worst_pids_tft_idf))
            #COLBERT
            for i, (pred_pids, target_pit, query) in enumerate(zip(pids, target_batch, query_batch)):
                if target_pit in pred_pids:
                    if len(good_pairs_cb.keys()) < size_datasets_good:
                        t = (query, target_pit)
                        good_pairs_cb[t] = pred_pids.index(target_pit)
                    else:
                        if pred_pids.index(target_pit) < max(set(good_pairs_cb.values())):
                            t = (query, target_pit)
                            good_pairs_cb[t] = pred_pids.index(target_pit)
                        if len(good_pairs_cb.keys()) > size_datasets_good:
                            good_pairs_cb.pop(max(good_pairs_cb, key=good_pairs_cb.get))

            for i, (pred_pids, target_pit, query) in enumerate(zip(worst_pids, target_batch, query_batch)):
                if target_pit in pred_pids:
                    if len(bad_pairs_cb.keys()) < size_datasets_bad:
                        t = (query, target_pit)
                        bad_pairs_cb[t] = pred_pids.index(target_pit) - K_bad + num_passages
                    else:
                        # if len(good_pairs_cb.keys()) > 0:
                        if pred_pids.index(target_pit) > min(set(bad_pairs_cb.values())):
                            t = (query, target_pit)
                            bad_pairs_cb[t] = pred_pids.index(target_pit) - K_bad + num_passages
                        if len(bad_pairs_cb.keys()) > size_datasets_bad:
                            bad_pairs_cb.pop(min(bad_pairs_cb, key=bad_pairs_cb.get))
                #else:
                #    bad_pairs_cb[(query, target_pit)] = K

            #TF IDF
            for i, (pred_pids, target_pit, query) in enumerate(zip(pids_tf_idf, target_batch, query_batch)):
                if target_pit in pred_pids:
                    if len(good_pairs_tf_idf.keys()) < size_datasets_good:
                        t = (query, target_pit)
                        good_pairs_tf_idf[t] = list(pred_pids).index(target_pit)
                    else:
                        if list(pred_pids).index(target_pit) < max(set(good_pairs_tf_idf.values())):
                            t = (query, target_pit)
                            good_pairs_tf_idf[t] = list(pred_pids).index(target_pit)
                        if len(good_pairs_tf_idf.keys()) > size_datasets_good:
                            good_pairs_tf_idf.pop(max(good_pairs_tf_idf, key=good_pairs_tf_idf.get))
                #else:
                #    bad_pairs_tf_idf[(query, target_pit)] = K

            for i, (pred_pids, target_pit, query) in enumerate(zip(worst_pids_tft_idf, target_batch, query_batch)):

                if target_pit in pred_pids:
                    if len(bad_pairs_tf_idf.keys()) < size_datasets_good:
                        t = (query, target_pit)
                        bad_pairs_tf_idf[t] = list(pred_pids).index(target_pit) - K_bad + num_passages
                    else:
                        if list(pred_pids).index(target_pit) < max(set(bad_pairs_tf_idf.values())):
                            t = (query, target_pit)
                            bad_pairs_tf_idf[t] = list(pred_pids).index(target_pit) - K_bad + num_passages
                        if len(bad_pairs_tf_idf.keys()) > size_datasets_good:
                            bad_pairs_tf_idf.pop(max(bad_pairs_tf_idf, key=bad_pairs_tf_idf.get))

            query_batch = []
            target_batch = []

    print(bad_pairs_cb.keys())
    print(bad_pairs_tf_idf.keys())
    keys_tf_good_cb_good = set(list(set(good_pairs_cb.keys()).intersection(good_pairs_tf_idf.keys()))[:return_size])
    keys_tf_good_cb_bad = set(list(set(bad_pairs_cb.keys()).intersection(good_pairs_tf_idf.keys()))[:return_size])
    keys_tf_bad_cb_good = set(list(set(good_pairs_cb.keys()).intersection(bad_pairs_tf_idf.keys()))[:return_size])
    keys_tf_bad_cb_bad = set(list(set(bad_pairs_cb.keys()).intersection(bad_pairs_tf_idf.keys()))[:return_size])

    return dict(((k[0], dataset.passages[k[1]]), (good_pairs_cb[k], good_pairs_tf_idf[k])) for k in keys_tf_good_cb_good), \
        dict(((k[0], dataset.passages[k[1]]), (bad_pairs_cb[k], good_pairs_tf_idf[k])) for k in keys_tf_good_cb_bad),\
        dict(((k[0], dataset.passages[k[1]]), (good_pairs_cb[k], bad_pairs_tf_idf[k])) for k in keys_tf_bad_cb_good), \
        dict(((k[0], dataset.passages[k[1]]), (bad_pairs_cb[k], bad_pairs_tf_idf[k])) for k in keys_tf_bad_cb_bad)


if __name__ == "__main__":
    sets = colbert_vs_tf_idf(size_datasets_good = 200, size_datasets_bad = 200, testing_max_count=100, K_good=1000, K_bad=1000, return_size=5)
    print("tf_good_cb_good, size:", len(sets[0].keys()), sets[0])
    print("tf_good_cb_bad, size:", len(sets[1].keys()), sets[1])
    print("tf_bad_cb_good, size:", len(sets[2].keys()), sets[2])
    print("tf_bad_cb_bad, size:", len(sets[3].keys()), sets[3])
