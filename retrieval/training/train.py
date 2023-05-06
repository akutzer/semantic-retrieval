#!/usr/bin/env python3
import random
from tqdm import tqdm
import torch
import numpy as np


from retrieval.configs import BaseConfig
from retrieval.data import TripleDataset, DataIterator
from retrieval.models import ColBERT

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
    passages_per_query = 1,
    epochs = 10,
    batch_size = 32,
    accum_steps = 2,    # sub_batch_size = ceil(batch_size / accum_steps)
    similarity="cosine",
    intra_batch_similarity=True,
    num_hidden_layers = 12,
    num_attention_heads = 12,
    dropout = 0.1,
    dim=32)

writer = SummaryWriter()

triples_path = "../../data/fandom-qa/witcher_qa/triples.train.tsv"
queries_path = "../../data/fandom-qa/witcher_qa/queries.train.tsv"
passages_path = "../../data/fandom-qa/witcher_qa/passages.train.tsv"

dataset = TripleDataset(config, triples_path, queries_path, passages_path, mode="QPP")
data_iter = DataIterator(config, dataset)

colbert = ColBERT(config, tokenizer=data_iter.tokenizer, device=DEVICE)

optimizer = torch.optim.AdamW(colbert.parameters(), lr=5e-6, eps=1e-8)
criterion = torch.nn.CrossEntropyLoss(reduction="sum")

for epoch in range(1, config.epochs+1):
    data_iter.shuffle()
    for i, batch in enumerate(tqdm(data_iter)):
        optimizer.zero_grad()
        losses, accs = 0, 0
        for sub_batch in batch:
            q_tokens, q_masks, p_tokens, p_masks = sub_batch
            Q, P = (q_tokens, q_masks), (p_tokens, p_masks)
            sub_B = q_tokens.shape[0]

            out = colbert(Q, P)

            loss = criterion(out, torch.arange(0, sub_B, device=DEVICE, dtype=torch.long))
            loss *= 1 / config.batch_size

            # calculate the accuracy within a subbatch -> extremly inflated accuracy
            accs += torch.sum(out.detach().max(dim=-1).indices == torch.arange(0, sub_B, device=DEVICE, dtype=torch.long))
            
            # calculate & accumulate gradients, the update step is done after the entire batch
            # has been passed through the model
            loss.backward()

            losses += loss.item()
        
        # update model parameters
        optimizer.step() 

        writer.add_scalar("Loss/train", losses, (epoch-1)*len(data_iter) + i)
        writer.add_scalar("Accuracy/train", accs / config.batch_size, (epoch-1)*len(data_iter) + i)

    data_iter.reset()
