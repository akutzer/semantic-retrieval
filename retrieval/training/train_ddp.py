#!/usr/bin/env python3
import os
import random
from tqdm import tqdm
import torch
import numpy as np


from retrieval.configs import BaseConfig
from retrieval.data import TripleDataset, BucketIterator
from retrieval.models import ColBERT, get_colbert_and_tokenizer

# tensorboard --logdir=runs
# http://localhost:6006/
from torch.utils.tensorboard import SummaryWriter

import argparse
import matplotlib.pyplot as plt
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import distributed as dist
from torch.utils.data.distributed import DistributedSampler

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

SEED = 125
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

parser = argparse.ArgumentParser()
parser.add_argument('--local-rank', type=int, help="local gpu id")
args = parser.parse_args()

torch.cuda.set_device(args.local_rank)
dist.init_process_group(backend='gloo') # nccl 
local_rank = dist.get_rank()

MODEL_PATH = "bert-base-uncased" # "bert-base-uncased" or "../../data/colbertv2.0/" or "roberta-base"
DEVICE = torch.device("cuda", args.local_rank)
# DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

config = BaseConfig(
    tok_name_or_path = MODEL_PATH,
    backbone_name_or_path = MODEL_PATH,
    passages_per_query = 1,
    epochs = 1,
    bucket_size = 16*10,
    batch_size = 16,
    accum_steps = 2,    # sub_batch_size = ceil(batch_size / accum_steps)
    similarity = "cosine",
    intra_batch_similarity = True,
    num_hidden_layers = 12,
    num_attention_heads = 12,
    dropout = 0.1,
    dim = 32)

writer = SummaryWriter()

colbert, tokenizer = get_colbert_and_tokenizer(config, device=DEVICE)
colbert = colbert.to(DEVICE)
colbert = DDP(colbert, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

triples_path = "../../data/fandom-qa/witcher_qa/triples.train.tsv"
queries_path = "../../data/fandom-qa/witcher_qa/queries.train.tsv"
passages_path = "../../data/fandom-qa/witcher_qa/passages.train.tsv"
dataset = TripleDataset(config, triples_path, queries_path, passages_path, mode="QPP")
bucket_iter = BucketIterator(config, dataset, tokenizer)
sampler = DistributedSampler(dataset)

optimizer = torch.optim.AdamW(colbert.parameters(), lr=5e-6, eps=1e-8)
criterion = torch.nn.CrossEntropyLoss(reduction="sum")

colbert.train()
for epoch in range(1, config.epochs+1):
    bucket_iter.shuffle()
    # sampler.set_epoch(epoch)
    for i, bucket in enumerate(tqdm(bucket_iter)):
        optimizer.zero_grad()
        losses, accs = 0, 0
        for j, batch in enumerate(bucket):
            Q, P = batch
            (q_tokens, q_masks), (p_tokens, p_masks) = Q, P
            sub_B = q_tokens.shape[0]

            out = colbert(Q, P)
            loss = criterion(out, torch.arange(0, sub_B, device=DEVICE, dtype=torch.long))
            loss *= 1 / config.batch_size

            # calculate & accumulate gradients, the update step is done after the entire batch
            # has been passed through the model
            loss.backward()

            with torch.inference_mode():
                losses += loss.item()
                # calculate the accuracy within a subbatch -> extremly inflated accuracy
                accs += torch.sum(out.detach().max(dim=-1).indices == torch.arange(0, sub_B, device=DEVICE, dtype=torch.long))
            
            # after accum_steps, update the weights + log the metrics
            if (j + 1) % config.accum_steps == 0:
                    # update model parameters
                optimizer.step()
                optimizer.zero_grad()

                # TODO: check if calculation is correct
                writer.add_scalar("Loss/train", losses, (epoch-1)*len(bucket_iter) + i*config.bucket_size/config.batch_size + j/config.accum_steps)
                writer.add_scalar("Accuracy/train", accs / config.batch_size, (epoch-1)*len(bucket_iter) + i*config.bucket_size/config.batch_size + j/config.accum_steps)
                losses, accs = 0, 0

    bucket_iter.reset()
