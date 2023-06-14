import torch
from retrieval.configs import BaseConfig
from retrieval.data import Passages, Queries, TripleDataset, BucketIterator
from retrieval.models import ColBERTTokenizer, ColBERTInference, get_colbert_and_tokenizer, load_colbert_and_tokenizer
from retrieval.indexing.colbert_indexer import ColBERTIndexer
from retrieval.indexing.colbert_retriever import ColBERTRetriever

if __name__ == "__main__":
    from tqdm import tqdm
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

    config = BaseConfig(
        tok_name_or_path=args.backbone,
        backbone_name_or_path=args.backbone,
        similarity="cosine",
        doc_maxlen=512,
        passages_per_query=10,
        checkpoint=args.checkpoint
    )

    dataset = TripleDataset(config, args.triples_path_val, args.queries_path_val, args.passages_path_val, mode=args.dataset_mode)

    # get passage list
    passage_list = [p[1] for p in dataset.passages_items()]

    # colbert, tokenizer = load_colbert_and_tokenizer(CHECKPOINT, device="cuda:0", config=config)
    colbert, tokenizer = load_colbert_and_tokenizer(args.checkpoint, device="cuda:0")
    inference = ColBERTInference(colbert, tokenizer)
    retriever = ColBERTRetriever(inference, device="cuda:0", passages=passage_list)
    retriever.indexer.load(args.indexer)

    BSIZE = 8 #8 #16 #8 #16
    K = 100
    top1, top3, top5, top10, top25, top100 = 0, 0, 0, 0, 0, 0
    mrr_10, mrr_100, recall_50 = 0, 0, 0

    # df = dataset.triples.data
    df = pd.read_csv(dataset.triples.path, sep='\t', index_col=False)

    if args.dataset_mode=="QPP":
        df.drop(df.columns[2:], axis=1, inplace=True)
        qrels = df.groupby('QID', as_index=False).agg(lambda x: set(x))

    if args.dataset_mode=="QQP":
        df.drop(df.columns[1:2], axis=1, inplace=True)
        qrels = df.groupby('QID+', as_index=False).agg(lambda x: set(x))

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
                    idx = torch.where(pred_pids == target_pid)[0]
                    # idx = torch.tensor([idx for idx, pred_pid in enumerate(pred_pids) if pred_pid == target_pid])
                    # print(idx, torch.where(pred_pids == target_pid))
                    if idx.numel() == 0:
                        continue
                    # print(target_pid, pred_pids[:10])
                    if idx < 100:
                        top100 += 1
                        mrr_100 += 1 / (idx + 1)
                        if idx < 50 and qids_visit[qid]==False:
                            qrel = qrels.iloc[list(qrels.iloc[:,0]).index(qid)][1]
                            # print(qid, target_pid, qrel, pred_pids[:50])
                            common = qrel & set(pred_pids[:50].cpu().numpy())
                            recall_50 += (len(common) / max(1.0, len(qrel)))
                            qids_visit[qid] = True
                            if idx < 25:
                                top25 += 1
                                if idx < 10:
                                    top10 += 1
                                    mrr_10 += 1 / (idx + 1)
                                    if idx < 5:
                                        top5 += 1
                                        if idx < 3:
                                            top3 += 1
                                            if idx < 1:
                                                top1 += 1

                qids_batch = []
                query_batch = []
                target_batch = []

        # pr.print_stats()


    print("Top-1-Acc:", round((100 * top1) / len(dataset), 3))
    print("Top-3-Acc:", round((100 * top3) / len(dataset), 3))
    print("Top-5-Acc:", round((100 * top5) / len(dataset), 3))
    print("Top-10-Acc:", round((100 * top10) / len(dataset), 3))
    print("Top-25-Acc:", round((100 * top25) / len(dataset), 3))
    print("Top-100-Acc:", round((100 * top100) / len(dataset), 3))

    print("MRR@10:", round((100 * mrr_10.item()) / len(dataset), 3))
    print("MRR@100:", round((100 * mrr_100.item()) / len(dataset), 3))
    print("Recall@50:", round((100 * recall_50) / len(dataset), 3))

