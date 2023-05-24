#!/usr/bin/env python3
import random
from tqdm import tqdm
import torch
import numpy as np


from retrieval.configs import BaseConfig
from retrieval.data import TripleDataset, BucketIterator
from retrieval.models import get_colbert_and_tokenizer

# tensorboard --logdir=runs
# http://localhost:6006/
from torch.utils.tensorboard import SummaryWriter


SEED = 125
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


MODEL_PATH = "roberta-base" # "bert-base-uncased" or "../../data/colbertv2.0/" or "roberta-base"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

config = BaseConfig(
    tok_name_or_path=MODEL_PATH,
    backbone_name_or_path=MODEL_PATH,
    passages_per_query = 10,
    epochs = 1,
    bucket_size = 2*10,
    batch_size = 2,
    accum_steps = 1,    # sub_batch_size = ceil(batch_size / accum_steps)
    similarity="cosine",
    intra_batch_similarity=False,
    num_hidden_layers = 12,
    num_attention_heads = 12,
    dropout = 0.1,
    dim=32)

writer = SummaryWriter()

colbert, tokenizer = get_colbert_and_tokenizer(config, device=DEVICE)

triples_path = "../../data/ms_marco_v1.1/train/triples.train.tsv"
queries_path = "../../data/ms_marco_v1.1/train/queries.train.tsv"
passages_path = "../../data/ms_marco_v1.1/train/passages.train.tsv"
dataset = TripleDataset(config, triples_path, queries_path, passages_path, mode="QPP")
bucket_iter = BucketIterator(config, dataset, tokenizer)

optimizer = torch.optim.AdamW(colbert.parameters(), lr=5e-6, eps=1e-8)
criterion = torch.nn.CrossEntropyLoss(reduction="sum")

import cProfile

with cProfile.Profile() as pr:
    for epoch in range(1, config.epochs+1):
        bucket_iter.shuffle()
        for i, bucket in enumerate(tqdm(bucket_iter)):
            optimizer.zero_grad()
            losses, accs = 0, 0
            for j, batch in enumerate(bucket):
                Q, P = batch
                (q_tokens, q_masks), (p_tokens, p_masks) = Q, P
                sub_B = q_tokens.shape[0]

                out = colbert(Q, P)
                # print(out.shape)
                # target = torch.arange(0, sub_B, device=DEVICE, dtype=torch.long)
                target = torch.ones(sub_B, device=DEVICE, dtype=torch.long)
                loss = criterion(out, target)
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
    
    pr.print_stats()
