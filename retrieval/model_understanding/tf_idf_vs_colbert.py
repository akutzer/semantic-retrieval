import random
from tqdm import tqdm
import torch

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
INDEX_PATH = "../../data/fandoms_qa/witcher/all/passages.indices.pt"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"



def colbert_vs_tf_idf(testing_max_count = 100, size_datasets_good = 100, size_datasets_bad = 100, K_good=1000, K_bad = 1000, return_size=20):
    '''Args:
    testing_max_count: how many
    '''

    dataset = TripleDataset(BaseConfig(passages_per_query=10),
                            triples_path="../../data/fandoms_qa/witcher/all/triples.tsv",
                            queries_path="../../data/fandoms_qa/witcher/all/queries.tsv",
                            passages_path="../../data/fandoms_qa/witcher/all/passages.tsv",
                            mode="QQP")


    colbert, tokenizer = load_colbert_and_tokenizer(CHECKPOINT_PATH)
    inference = ColBERTInference(colbert, tokenizer)
    retriever = ColBERTRetriever(inference, device=DEVICE, passages=dataset.passages)

    # precompute indicies
    # retriever.indexer.dtype = torch.float16
    data = dataset.passages.values().tolist()
    pids = dataset.passages.keys().tolist()
    retriever.indexer.index(data, pids, bsize=8)
    retriever.indexer.save(INDEX_PATH)
    # retriever.indexer.load(INDEX_PATH)

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
                    else:
                        if pred_pids.index(target_pit) < max(set(good_pairs_cb.values())):
                            t = (query, target_pit)
                            good_pairs_cb[t] = pred_pids.index(target_pit)
                        if len(good_pairs_cb.keys()) > size_datasets_good:
                            good_pairs_cb.pop(max(good_pairs_cb, key=good_pairs_cb.get))
                else:
                    bad_pairs_cb[(query, target_pit)] = K_good

            # for i, (pred_pids, target_pit, query) in enumerate(zip(worst_pids, target_batch, query_batch)):
            #     if target_pit in pred_pids:
            #         if len(bad_pairs_cb.keys()) < size_datasets_bad:
            #             t = (query, target_pit)
            #             bad_pairs_cb[t] = pred_pids.index(target_pit) - K_bad + num_passages
            #         else:
            #             # if len(good_pairs_cb.keys()) > 0:
            #             if pred_pids.index(target_pit) > min(set(bad_pairs_cb.values())):
            #                 t = (query, target_pit)
            #                 bad_pairs_cb[t] = pred_pids.index(target_pit) - K_bad + num_passages
            #             if len(bad_pairs_cb.keys()) > size_datasets_bad:
            #                 bad_pairs_cb.pop(min(bad_pairs_cb, key=bad_pairs_cb.get))


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
                else:
                    bad_pairs_tf_idf[(query, target_pit)] = K_good

            # for i, (pred_pids, target_pit, query) in enumerate(zip(worst_pids_tft_idf, target_batch, query_batch)):
            #
            #     if target_pit in pred_pids:
            #         if len(bad_pairs_tf_idf.keys()) < size_datasets_good:
            #             t = (query, target_pit)
            #             bad_pairs_tf_idf[t] = list(pred_pids).index(target_pit) - K_bad + num_passages
            #         else:
            #             if list(pred_pids).index(target_pit) < max(set(bad_pairs_tf_idf.values())):
            #                 t = (query, target_pit)
            #                 bad_pairs_tf_idf[t] = list(pred_pids).index(target_pit) - K_bad + num_passages
            #             if len(bad_pairs_tf_idf.keys()) > size_datasets_good:
            #                 bad_pairs_tf_idf.pop(max(bad_pairs_tf_idf, key=bad_pairs_tf_idf.get))

            query_batch = []
            target_batch = []

    print(bad_pairs_cb.keys())
    print(good_pairs_tf_idf.keys())
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
    sets = colbert_vs_tf_idf(size_datasets_good = 200, size_datasets_bad = 200, testing_max_count=100_000_000, K_good=50_000, return_size=10)
    print("tf_good_cb_good, size:", len(sets[0].keys()), sets[0])
    print("tf_good_cb_bad, size:", len(sets[1].keys()), sets[1])
    print("tf_bad_cb_good, size:", len(sets[2].keys()), sets[2])
    print("tf_bad_cb_bad, size:", len(sets[3].keys()), sets[3])
