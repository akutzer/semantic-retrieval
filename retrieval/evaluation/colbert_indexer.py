#!/usr/bin/env python3
import logging
from typing import Union, List, Tuple
import torch

from retrieval.models import ColBERTInference
from retrieval.indexing.colbert_indexer import ColBERTIndexer
from retrieval.indexing.indexer import IndexerInterface


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

if __name__ == "__main__":
    import random
    import numpy as np
    import argparse

    from retrieval.data import Passages
    from retrieval.models import load_colbert_and_tokenizer

    SEED = 125
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    # enable TensorFloat32 tensor cores for float32 matrix multiplication if available
    torch.set_float32_matmul_precision("high")

    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

    parser = argparse.ArgumentParser(description="ColBERT Retrieving")
    # Dataset arguments
    dataset_args = parser.add_argument_group("Dataset Arguments")
    dataset_args.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset")
    dataset_args.add_argument("--passages_path_val", type=str, help="Path to the validation passages.tsv file")

    # Model arguments
    model_args = parser.add_argument_group("Model Arguments")
    model_args.add_argument("--indexer", type=str, help="Path of the indexer which should be saved")
    model_args.add_argument("--checkpoint", type=str, help="Path of the checkpoint which should be loaded")

    args = parser.parse_args()

    colbert, tokenizer = load_colbert_and_tokenizer(args.checkpoint, device=DEVICE)
    inference = ColBERTInference(colbert, tokenizer, device=DEVICE)
    indexer = ColBERTIndexer(inference, device=DEVICE, dtype=torch.float16)
    print(colbert.config)

    passages = Passages(args.passages_path_val)
    data = passages.values().tolist()
    pids = passages.keys().tolist()

    # test indexing of already seen data
    # indexer.index(data[1:2], pids[1:2], bsize=8)
    # indexer.index(data[:2], pids[:2], bsize=8)
    # indexer.index(data[:3], pids[:3], bsize=8)
    # print(indexer.embeddings.shape)
    # print(indexer.iid2pid)
    # print(indexer.pid2iid)
    # print(indexer.iid2pid.shape, indexer.pid2iid.shape, indexer.offset)

    # test some other methods
    # test_iids = torch.arange(0, 10).reshape(5, 2).T[:, None]
    # test_pids = indexer.iids_to_pids(test_iids)
    # test_embs = indexer.get_pid_embedding(test_pids)

    # index the entire data
    indexer.index(data, pids, bsize=8)
    indexer.save(args.indexer)
    indexer.load(args.indexer)
    print(indexer.embeddings.shape)
    print(indexer.iid2pid.shape)
    print(indexer.pid2iid.shape)
    print((indexer.pid2iid.sum(dim=-1) == -32).sum())

    # test retrieval
    # queries = [
    #    "Who is the author of 'The Witcher'?",
    #    "How does an NPC behave when it starts raining?",
    #    "Who the hell is Cynthia?",
    # ]
    queries = [
        "Who was Olympe Maxime, and what were her characteristics?",
        "What recognition did Fleur Delacour receive after the Battle of Hogwarts?",
        "What happened when Angus Buchanan put on the Sorting Hat?",
    ]

    Qs = indexer.inference.query_from_text(queries)
    if Qs.dim() == 2:
        Qs = Qs[None]
    # print(Qs.shape)

    batch_sim, batch_iids = indexer.search(Qs, k=25)
    # print(batch_sim.shape, batch_iids.shape)
    batch_pids = indexer.iids_to_pids(batch_iids)
    # print(batch_pids)
    batch_embs, batch_masks = indexer.get_pid_embedding(batch_pids)
    # print(batch_embs, batch_masks)

    for i, (Q, pids, topk_embs, mask) in enumerate(zip(Qs, batch_pids, batch_embs, batch_masks)):
        # print(Q.shape, pids.shape, topk_embs.shape, mask.shape)

        # topk_embs @ Q.mT instead of Q @ topk_embs.mT because of the masking later on
        sim = topk_embs @ Q.to(dtype=topk_embs.dtype).mT  # (N_doc, L_d, L_q)

        # replace the similarity results for padding vectors
        sim[~mask] = -torch.inf

        # calculate the sum of max similarities
        sms = sim.max(dim=1).values.sum(dim=-1)
        # print(sim.shape, sms.shape)

        values, indices = torch.sort(sms, descending=True)
        sorted_pids = pids[indices]
        # print(pids, values, indices)
        # print(values, sorted_pids)

        print("=" * 150)
        print(f"Query: {queries[i]}")
        print("=" * 150)
        for sim, pid in zip(values, sorted_pids[:10]):
            print(round(sim.item(), 3), pid.item(), passages[pid.item()])
        print(end="\n\n\n")

