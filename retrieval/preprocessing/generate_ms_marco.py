import os
import csv
import json
import random
random.seed(125)

from tqdm import tqdm
from datasets import load_dataset



DATASET = "ms_marco"
VERSION = "v2.1" # "v1.1" or "v2.1"
MODE = "tsv"
NUM_NEG_PIDS = 9

DATASET_DIR = f"../../data/{DATASET}_{VERSION}"
os.makedirs(DATASET_DIR, exist_ok=True)


datasets = load_dataset(DATASET, VERSION, split=None)
print(datasets)

for dataset_type, dataset  in datasets.items():
    # skip the test dataset since they have no labels and we are not interested
    # in uploading our results to the ms marco website
    if dataset_type == "test":
        continue

    n_pos = 0
    n_total = 0

    SUB_DATASET_DIR = os.path.join(DATASET_DIR, dataset_type)
    os.makedirs(SUB_DATASET_DIR, exist_ok=True)

    unique_queries = set()
    unique_passages = set()

    qid2query = {}
    query2qid = {}
    pid2passage = {}
    passage2pid = {}
    triples = []

    for data in tqdm(dataset):
        query, passages, is_selected = data["query"], data["passages"]["passage_text"], data["passages"]["is_selected"]

        # register query and get its QID
        if query in unique_queries:
            qid = query2qid[query]
        else:
            qid = len(unique_queries)
            unique_queries.add(query)
            query2qid[query] = qid
            qid2query[qid] = query
        
        pids = []
        # register passages and get their PIDs
        for passage in passages:
            if passage in unique_passages:
                pid = passage2pid[passage]
            else:     
                pid = len(unique_passages)
                unique_passages.add(passage)
                passage2pid[passage] = pid
                pid2passage[pid] = passage
            
            pids.append(pid)
        
        # skip triples where there is no answer
        if sum(is_selected) == 0:
            continue

        pos_pids, neg_pids = [], []
        for selected, pid in zip(is_selected, pids):
            if selected:
                pos_pids.append(pid)
            else:
                neg_pids.append(pid)
        
        if len(neg_pids) == 0:
            continue
        
        # if there a too few PID⁻, we sample some PID⁻ multiple times
        missing_neg = NUM_NEG_PIDS - len(neg_pids)
        if missing_neg != 0:
            neg_pids.extend(random.choices(neg_pids, k=missing_neg))
        
        # for each PID⁺ we add a new "triple"
        for pos_pid in pos_pids:
            triples.append([qid] + [pos_pid] + neg_pids)
        
        if len(pos_pids) > 1:
            n_pos += 1
        n_total += 1
    print(n_pos, n_total, round((100 * n_pos) / n_total, 3))

    triples_path = os.path.join(SUB_DATASET_DIR, f"triples.{dataset_type}.{MODE.lower()}")
    queries_path = os.path.join(SUB_DATASET_DIR, f"queries.{dataset_type}.{MODE.lower()}")
    passages_path = os.path.join(SUB_DATASET_DIR, f"passages.{dataset_type}.{MODE.lower()}")

    if MODE.lower() == "tsv":
        with open(triples_path, mode="w", encoding="utf-8", newline="") as trip_f:
            writer = csv.writer(trip_f, delimiter="\t", lineterminator="\n")
            writer.writerows(triples)
        
        with open(queries_path, mode="w", encoding="utf-8", newline="") as q_f:
            writer = csv.writer(q_f, delimiter="\t", lineterminator="\n")
            writer.writerows(qid2query.items())
        
        with open(passages_path, mode="w", encoding="utf-8", newline="") as p_f:
            writer = csv.writer(p_f, delimiter="\t", lineterminator="\n")
            writer.writerows(pid2passage.items())

    elif MODE.lower() == "json":
        with open(triples_path, mode="w", encoding="utf-8") as trip_f:
            json.dump(triples, trip_f)
        
        with open(queries_path, mode="w", encoding="utf-8") as q_f:
            json.dump(qid2query, q_f, indent=0)
        
        with open(passages_path, mode="w", encoding="utf-8") as p_f:
            json.dump(pid2passage, p_f, indent=0)
