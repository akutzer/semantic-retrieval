
from dataclasses import dataclass

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from retrieval.configs import BaseConfig
from retrieval.data import TripleDataset
from retrieval.models import ColBERTInference, load_colbert_and_tokenizer, inference_to_embedding
from retrieval.indexing import ColBERTRetriever, index



@dataclass
class RetrievalConfig:
    dataset_mode: str
    passages_path: str
    queries_path: str
    triples_path: str

    checkpoint_path: str
    index_path: str

    use_gpu: bool
    device: str
    dtype: torch.dtype = torch.float16
    batch_size: int = 8
    k: int = 1000


def argparser2retrieval_config(args):
    device = "cuda" if args.use_gpu and torch.cuda.is_available() else "cpu"

    if args.dtype.upper() == "FP16":
        dtype = torch.float16
    elif args.dtype.upper() == "FP32":
        dtype = torch.float32
    else:
        dtype = torch.float64

    config = RetrievalConfig(
        dataset_mode = args.dataset_mode,
        passages_path = args.passages_path,
        queries_path = args.queries_path,
        triples_path = args.triples_path,
        checkpoint_path = args.checkpoint_path,
        index_path = args.index_path,
        use_gpu = args.use_gpu,
        dtype = dtype,
        device = device,
        batch_size = args.batch_size,
        k = args.k
    )   

    return config

def evaluate(pids, qids_visit, qids_batch, qrels, 
             recall_1, recall_3, recall_5, recall_10, recall_25, recall_50, 
             recall_100, recall_200, recall_1000, mrr_5, mrr_10, mrr_100):
    for ((_, pred_pids), qid) in zip(pids, qids_batch):
        pred_pids = pred_pids.cpu().numpy()

        if config.dataset_mode=="QPP":
            qrel = qrels[qrels['QID'] == qid]['PID+'].values[0]
        if config.dataset_mode=="QQP":
            qrel = qrels[qrels['QID+'] == qid]['PID'].values[0]
            
        idxs = np.where(np.isin(pred_pids, list(qrel)))[0]
            
        if idxs.size < 1 or len(qrel) < 1:
            continue

        if not qids_visit[qid]:
            qids_visit[qid] = True
            if idxs[0] < 1000:
                common = qrel & set(pred_pids[:1000])
                recall_1000 += len(common) / len(qrel)
    
                if idxs[0] < 200:
                    common = qrel & set(pred_pids[:200])
                    recall_200 += len(common) / len(qrel)

                    if idxs[0] < 100:
                        common = qrel & set(pred_pids[:100])
                        recall_100 += len(common) / len(qrel)
                        mrr_100 += 1 / (idxs[0] + 1)

                        if idxs[0] < 50:
                            common = qrel & set(pred_pids[:50])
                            recall_50 += len(common) / len(qrel)

                            if idxs[0] < 25:
                                common = qrel & set(pred_pids[:25])
                                recall_25 += len(common) / len(qrel)

                                if idxs[0] < 10:
                                    common = qrel & set(pred_pids[:10])
                                    recall_10 += len(common) / len(qrel)
                                    mrr_10 += 1 / (idxs[0] + 1)

                                    if idxs[0] < 5:
                                        common = qrel & set(pred_pids[:5])
                                        recall_5 += len(common) / len(qrel)
                                        mrr_5 += 1 / (idxs[0] + 1)

                                        if idxs[0] < 3:
                                            common = qrel & set(pred_pids[:3])
                                            recall_3 += len(common) / len(qrel)

                                            if idxs[0] < 1:
                                                common = qrel & set(pred_pids[:1])
                                                recall_1 += len(common) / len(qrel)
            
        
    return qids_visit, recall_1, recall_3, recall_5, recall_10, recall_25, recall_50, recall_100, recall_200, recall_1000, mrr_5, mrr_10, mrr_100

# TODO: clean code
# TODO: evaluate tf_idf, rerank and full_retrieval at the same time!!!
def evaluate_colbert(retriever: ColBERTRetriever, dataset: TripleDataset, config: RetrievalConfig):
    
    # tf_idf_recall_1, tf_idf_recall_3, tf_idf_recall_5, tf_idf_recall_10 = 0, 0, 0, 0
    # tf_idf_recall_25, tf_idf_recall_50, tf_idf_recall_100, tf_idf_recall_200 = 0, 0, 0, 0
    # tf_idf_recall_1000, tf_idf_mrr_5, tf_idf_mrr_10, tf_idf_mrr_100 = 0, 0, 0, 0

    rerank_recall_1, rerank_recall_3, rerank_recall_5, rerank_recall_10 = 0, 0, 0, 0
    rerank_recall_25, rerank_recall_50, rerank_recall_100, rerank_recall_200 = 0, 0, 0, 0
    rerank_recall_1000, rerank_mrr_5, rerank_mrr_10, rerank_mrr_100 = 0, 0, 0, 0

    full_recall_1, full_recall_3, full_recall_5, full_recall_10 = 0, 0, 0, 0
    full_recall_25, full_recall_50, full_recall_100, full_recall_200 = 0, 0, 0, 0
    full_recall_1000, full_mrr_5, full_mrr_10, full_mrr_100 = 0, 0, 0, 0

    # df = dataset.triples.data
    df = pd.read_csv(dataset.triples.path, sep='\t', index_col=False)

    if config.dataset_mode=="QPP":
        df.drop(df.columns[2:], axis=1, inplace=True)
        qrels = df.groupby('QID', as_index=False).agg(lambda x: set(x))
        datalen = df['QID'].max() + 1

    if config.dataset_mode=="QQP":
        df.drop(df.columns[1:2], axis=1, inplace=True)
        qrels = df.groupby('QID+', as_index=False).agg(lambda x: set(x))
        datalen = df['QID+'].max() + 1

    qids_batch = []
    query_batch = []
    # tf_idf_qids_visit = np.zeros(datalen, dtype=bool)
    rerank_qids_visit = np.zeros(datalen, dtype=bool)
    full_qids_visit = np.zeros(datalen, dtype=bool)
    
    for i, triple in enumerate(tqdm(dataset)):
        if config.dataset_mode=="QPP":
            qid, pid_pos, *pid_neg = triple
            query, psg_pos, *psg_neg = dataset.id2string(triple)
            qids_batch.append(qid)
            query_batch.append(query)

        if config.dataset_mode=="QQP":
            qid_pos, qid_neg, pid_pos = triple
            query_pos, query_neg, passage = dataset.id2string(triple)
            qids_batch.append(qid_pos)
            query_batch.append(query_pos)

        if len(query_batch) == config.batch_size or i + 1 == len(dataset):
            with torch.autocast(retriever.device.type):
                # tf_idf_pids = retriever.tf_idf_rank(query_batch, config.k)
                rerank_pids = retriever.rerank(query_batch, config.k)
                full_pids = retriever.full_retrieval(query_batch, config.k)

            # tf_idf_qids_visit, tf_idf_recall_1, tf_idf_recall_3, tf_idf_recall_5, tf_idf_recall_10, tf_idf_recall_25, tf_idf_recall_50, tf_idf_recall_100, tf_idf_recall_200, tf_idf_recall_1000, tf_idf_mrr_5, tf_idf_mrr_10, tf_idf_mrr_100 = evaluate(tf_idf_pids, tf_idf_qids_visit, qids_batch, qrels, 
            #          tf_idf_recall_1, tf_idf_recall_3, tf_idf_recall_5, tf_idf_recall_10,
            #          tf_idf_recall_25, tf_idf_recall_50, tf_idf_recall_100, tf_idf_recall_200, 
            #          tf_idf_recall_1000, tf_idf_mrr_5, tf_idf_mrr_10, tf_idf_mrr_100)
            
            rerank_qids_visit, rerank_recall_1, rerank_recall_3, rerank_recall_5, rerank_recall_10, rerank_recall_25, rerank_recall_50, rerank_recall_100, rerank_recall_200, rerank_recall_1000, rerank_mrr_5, rerank_mrr_10, rerank_mrr_100 = evaluate(rerank_pids, rerank_qids_visit, qids_batch, qrels, 
                     rerank_recall_1, rerank_recall_3, rerank_recall_5, rerank_recall_10,
                     rerank_recall_25, rerank_recall_50, rerank_recall_100, rerank_recall_200, 
                     rerank_recall_1000, rerank_mrr_5, rerank_mrr_10, rerank_mrr_100)    
                    
            full_qids_visit, full_recall_1, full_recall_3, full_recall_5, full_recall_10, full_recall_25, full_recall_50, full_recall_100, full_recall_200, full_recall_1000, full_mrr_5, full_mrr_10, full_mrr_100 = evaluate(full_pids, full_qids_visit, qids_batch, qrels,
                     full_recall_1, full_recall_3, full_recall_5, full_recall_10, 
                     full_recall_25, full_recall_50, full_recall_100, full_recall_200, 
                     full_recall_1000, full_mrr_5, full_mrr_10, full_mrr_100)
            
            qids_batch = []
            query_batch = []
            
    # print("tf_if_retrieval:")
    # print("Recall@1:", round((100 * tf_idf_recall_1) / len(dataset), 3))
    # print("Recall@3:", round((100 * tf_idf_recall_3) / len(dataset), 3))
    # print("Recall@5:", round((100 * tf_idf_recall_5) / len(dataset), 3))
    # print("Recall@10:", round((100 * tf_idf_recall_10) / len(dataset), 3))
    # print("Recall@25:", round((100 * tf_idf_recall_25) / len(dataset), 3))
    # print("Recall@50:", round((100 * tf_idf_recall_50) / len(dataset), 3))
    # print("Recall@100:", round((100 * tf_idf_recall_100) / len(dataset), 3))
    # print("Recall@200:", round((100 * tf_idf_recall_200) / len(dataset), 3))
    # print("Recall@1000:", round((100 * tf_idf_recall_1000) / len(dataset), 3))

    # print("MRR@5:", round((100 * tf_idf_mrr_5.item()) / len(dataset), 3))
    # print("MRR@10:", round((100 * tf_idf_mrr_10.item()) / len(dataset), 3))
    # print("MRR@100:", round((100 * tf_idf_mrr_100.item()) / len(dataset), 3))
    # print("") 

    print("rerank_retrieval:")
    print("Recall@1:", round((100 * rerank_recall_1) / len(qrels), 3))
    print("Recall@3:", round((100 * rerank_recall_3) / len(qrels), 3))
    print("Recall@5:", round((100 * rerank_recall_5) / len(qrels), 3))
    print("Recall@10:", round((100 * rerank_recall_10) / len(qrels), 3))
    print("Recall@25:", round((100 * rerank_recall_25) / len(qrels), 3))
    print("Recall@50:", round((100 * rerank_recall_50) / len(qrels), 3))
    print("Recall@100:", round((100 * rerank_recall_100) / len(qrels), 3))
    print("Recall@200:", round((100 * rerank_recall_200) / len(qrels), 3))
    print("Recall@1000:", round((100 * rerank_recall_1000) / len(qrels), 3))

    print("MRR@5:", round((100 * rerank_mrr_5.item()) / len(qrels), 3))
    print("MRR@10:", round((100 * rerank_mrr_10.item()) / len(qrels), 3))
    print("MRR@100:", round((100 * rerank_mrr_100.item()) / len(qrels), 3))
    print("")

    print("full_retrieval:")
    print("Recall@1:", round((100 * full_recall_1) / len(qrels), 3))
    print("Recall@3:", round((100 * full_recall_3) / len(qrels), 3))
    print("Recall@5:", round((100 * full_recall_5) / len(qrels), 3))
    print("Recall@10:", round((100 * full_recall_10) / len(qrels), 3))
    print("Recall@25:", round((100 * full_recall_25) / len(qrels), 3))
    print("Recall@50:", round((100 * full_recall_50) / len(qrels), 3))
    print("Recall@100:", round((100 * full_recall_100) / len(qrels), 3))
    print("Recall@200:", round((100 * full_recall_200) / len(qrels), 3))
    print("Recall@1000:", round((100 * full_recall_1000) / len(qrels), 3))

    print("MRR@5:", round((100 * full_mrr_5.item()) / len(qrels), 3))
    print("MRR@10:", round((100 * full_mrr_10.item()) / len(qrels), 3))
    print("MRR@100:", round((100 * full_mrr_100.item()) / len(qrels), 3))

if __name__ == "__main__":
    import argparse

    # enable TensorFloat32 tensor cores for float32 matrix multiplication if available
    torch.set_float32_matmul_precision("high")

    parser = argparse.ArgumentParser(description="Evaluate Retrieval")
    parser.add_argument("--dataset-mode", type=str, required=True, choices=["QQP", "QPP"], help="Mode of the dataset")
    parser.add_argument("--passages-path", type=str, required=True, help="Path to the passages.tsv file")
    parser.add_argument("--queries-path", type=str, required=True, help="Path to the queries.tsv file")
    parser.add_argument("--triples-path", type=str, required=True, help="Path to the triples.tsv file")
    parser.add_argument("--checkpoint-path", type=str, required=True, help="Path to ColBERT Checkpoint (should be the same checkpoint which was used for the indexation)")
    parser.add_argument("--index-path", type=str, help="Path of the indexer which should be loaded")
    parser.add_argument("--use-gpu", action="store_true", help="Use GPU for indexing (recommended)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for used during the retrieval process")
    parser.add_argument("--dtype", type=str, default="FP16", choices=["FP16", "FP32", "FP64"], help="Floating-point precision of the indices")
    parser.add_argument("--k", type=int, default=100, help="Number of top-k passages that should be retrieved")
    parser.add_argument("--embedding-only", action="store_true", help="This used only the word embedding layer of the ColBERT model")

    args = parser.parse_args()
    config = argparser2retrieval_config(args)
    print(config)

    dataset = TripleDataset(
        BaseConfig(),
        config.triples_path,
        config.queries_path,
        config.passages_path,
        config.dataset_mode,
    )

    colbert, tokenizer = load_colbert_and_tokenizer(config.checkpoint_path)
    print(colbert.config)
    inference = ColBERTInference(colbert, tokenizer)
    if args.embedding_only:
        inference = inference_to_embedding(inference, just_word_emb=False, layer_norm=True)
        print(inference.colbert)
    retriever = ColBERTRetriever(inference, device=config.device, passages=dataset.passages)

    if config.index_path is not None:
        retriever.indexer.load(config.index_path)
    else:
        # just horrible but it works :)))
        retriever.indexer = index(inference, config)
    
    evaluate_colbert(retriever, dataset, config)



