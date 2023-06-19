import torch
from tqdm import tqdm
from retrieval.configs import BaseConfig
from retrieval.data import Passages, Queries, TripleDataset, BucketIterator
from retrieval.models import ColBERTTokenizer, ColBERTInference, get_colbert_and_tokenizer, load_colbert_and_tokenizer
from retrieval.indexing.colbert_indexer import ColBERTIndexer
from retrieval.indexing.colbert_retriever import ColBERTRetriever

if __name__ == "__main__":
    import cProfile
    import pandas as pd
    import numpy as np
    import argparse

    # enable TensorFloat32 tensor cores for float32 matrix multiplication if available
    torch.set_float32_matmul_precision("high")

    parser = argparse.ArgumentParser(description="ColBERT Retrieving")
    # Dataset arguments
    dataset_args = parser.add_argument_group("Dataset Arguments")
    dataset_args.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset")
    dataset_args.add_argument("--dataset_mode", type=str, required=True, choices=["QQP", "QPP"], help="Mode of the dataset")
    dataset_args.add_argument("--passages_path_val", type=str, help="Path to the validation passages.tsv file")
    dataset_args.add_argument("--queries_path_val", type=str, help="Path to the validation queries.tsv file")
    dataset_args.add_argument("--triples_path_val", type=str, help="Path to the validation triples.tsv file")

    # Model arguments
    model_args = parser.add_argument_group("Model Arguments")
    model_args.add_argument("--backbone", type=str, help="Name of the backbone model")
    model_args.add_argument("--indexer", type=str, help="Path of the indexer which should be loaded")
    model_args.add_argument("--checkpoint", type=str, help="Path of the checkpoint which should be loaded")

    args = parser.parse_args()

    dataset = TripleDataset(
        BaseConfig(passages_per_query=10),
        args.triples_path_val,
        args.queries_path_val,
        args.passages_path_val,
        args.dataset_mode,
    )

    # colbert, tokenizer = load_colbert_and_tokenizer(args.checkpoint, device="cuda:0", config=config)
    colbert, tokenizer = load_colbert_and_tokenizer(args.checkpoint, device="cuda:0")
    inference = ColBERTInference(colbert, tokenizer)
    retriever = ColBERTRetriever(inference, device="cuda:0", passages=dataset.passages)
    retriever.indexer.load(args.indexer)

    BSIZE = 8  # 8 #16 #8 #16
    K = 1000
    recall_1, recall_3, recall_5, recall_10, recall_25, recall_50, recall_100, recall_200, recall_1000 = 0, 0, 0, 0, 0, 0, 0, 0, 0
    mrr_5, mrr_10, mrr_100 = 0, 0, 0

    # df = dataset.triples.data
    df = pd.read_csv(dataset.triples.path, sep='\t', index_col=False)

    if args.dataset_mode=="QPP":
        df.drop(df.columns[2:], axis=1, inplace=True)
        qrels = df.groupby('QID', as_index=False).agg(lambda x: set(x))
        qid_0 = df['QID'][0]

    if args.dataset_mode=="QQP":
        df.drop(df.columns[1:2], axis=1, inplace=True)
        qrels = df.groupby('QID+', as_index=False).agg(lambda x: set(x))
        qid_0 = df['QID+'][0]

    with cProfile.Profile() as pr:
        qids_batch = []
        query_batch = []
        target_batch = []
        qids_visit = np.zeros(2*len(dataset), dtype=bool)
        
        for i, triple in enumerate(tqdm(dataset)):

            if args.dataset_mode=="QPP":
                qid, pid_pos, *pid_neg = triple
                query, psg_pos, *psg_neg = dataset.id2string(triple)
                qids_batch.append(qid)
                query_batch.append(query)
                target_batch.append(pid_pos)

            if args.dataset_mode=="QQP":
                qid_pos, qid_neg, pid_pos = triple
                query_pos, query_neg, passage = dataset.id2string(triple)
                qids_batch.append(qid_pos)
                query_batch.append(query_pos)
                target_batch.append(pid_pos)

            if len(query_batch) == BSIZE or i + 1 == len(dataset):
                with torch.autocast(retriever.device.type):
                    # pids = retriever.tf_idf_rank(query_batch, K)
                    # pids = retriever.rerank(query_batch, K)
                    pids = retriever.full_retrieval(query_batch, K)

                for j, ((sims, pred_pids), qid, target_pid) in enumerate(zip(pids, qids_batch, target_batch)):
                    qrel = qrels.iloc[list(qrels.iloc[:,0]).index(qid)][1]
                    # idxs = torch.where(pred_pids == torch.tensor(list(qrel)).to("cuda:0"))[0]
                    idxs = torch.tensor([idx for idx, pred_pid in enumerate(pred_pids) if pred_pid in list(qrel)])
                    # print(qid, target_pid, qrel, pred_pids[:10], idxs)
                    if idxs.numel() == 0:
                        continue
                    if idxs.numel() > 1:
                        idxs, indices = torch.sort(idxs, dim=0)

                    if qids_visit[qid-qid_0]==False:
                        if idxs[0] < 1000:
                            common = qrel & set(pred_pids[:1000].cpu().numpy())
                            recall_1000 += (len(common) / max(1.0, len(qrel)))
                
                        if idxs[0] < 200:
                            common = qrel & set(pred_pids[:200].cpu().numpy())
                            recall_200 += (len(common) / max(1.0, len(qrel)))

                            if idxs[0] < 100:
                                common = qrel & set(pred_pids[:100].cpu().numpy())
                                recall_100 += (len(common) / max(1.0, len(qrel)))
                                for idx in idxs:
                                    if idx < 100:
                                        mrr_100 += 1 / (idx + 1)

                                if idxs[0] < 50:
                                    common = qrel & set(pred_pids[:50].cpu().numpy())
                                    recall_50 += (len(common) / max(1.0, len(qrel)))

                                    if idxs[0] < 25:
                                        common = qrel & set(pred_pids[:25].cpu().numpy())
                                        recall_25 += (len(common) / max(1.0, len(qrel)))

                                        if idxs[0] < 10:
                                            common = qrel & set(pred_pids[:10].cpu().numpy())
                                            recall_10 += (len(common) / max(1.0, len(qrel)))
                                            for idx in idxs:
                                                if idx < 10:
                                                    mrr_10 += 1 / (idx + 1)

                                            if idxs[0] < 5:
                                                common = qrel & set(pred_pids[:5].cpu().numpy())
                                                recall_5 += (len(common) / max(1.0, len(qrel)))
                                                for idx in idxs:
                                                    if idx < 5:
                                                        mrr_5 += 1 / (idx + 1)

                                                if idxs[0] < 3:
                                                    common = qrel & set(pred_pids[:3].cpu().numpy())
                                                    recall_3 += (len(common) / max(1.0, len(qrel)))

                                                    if idxs[0] < 1:
                                                        common = qrel & set(pred_pids[:1].cpu().numpy())
                                                        recall_1 += (len(common) / max(1.0, len(qrel)))
                    qids_visit[qid-qid_0] = True                    

                qids_batch = []
                query_batch = []
                target_batch = []

        # pr.print_stats()

    print("Recall@1:", round((100 * recall_1) / len(dataset), 3))
    print("Recall@3:", round((100 * recall_3) / len(dataset), 3))
    print("Recall@5:", round((100 * recall_5) / len(dataset), 3))
    print("Recall@10:", round((100 * recall_10) / len(dataset), 3))
    print("Recall@25:", round((100 * recall_25) / len(dataset), 3))
    print("Recall@50:", round((100 * recall_50) / len(dataset), 3))
    print("Recall@100:", round((100 * recall_100) / len(dataset), 3))
    print("Recall@200:", round((100 * recall_200) / len(dataset), 3))
    print("Recall@1000:", round((100 * recall_1000) / len(dataset), 3))

    print("MRR@5:", round((100 * mrr_5.item()) / len(dataset), 3))
    print("MRR@10:", round((100 * mrr_10.item()) / len(dataset), 3))
    print("MRR@100:", round((100 * mrr_100.item()) / len(dataset), 3))

