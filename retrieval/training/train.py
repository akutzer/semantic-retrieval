#!/usr/bin/env python3
import random
from tqdm import tqdm
import torch
import numpy as np


from retrieval.configs import BaseConfig
from retrieval.data import DataIterator
from retrieval.models import ColBERT

# tensorboard --logdir=runs
# http://localhost:6006/
from torch.utils.tensorboard import SummaryWriter


random.seed(125)
np.random.seed(125)
torch.manual_seed(125)
torch.cuda.manual_seed_all(125)


MODEL_PATH = "../../data/colbertv2.0/" # "bert-base-uncased" #"bert-base-uncased" # 
DEVICE = "cuda:0"

config = BaseConfig(
    tok_name_or_path=MODEL_PATH,
    backbone_name_or_path=MODEL_PATH,
    epochs = 10,
    batch_size = 32,
    accum_steps = 2,
    similarity="cosine", 
    intra_batch_similarity=True)

writer = SummaryWriter()

triples_path = "/home/aaron/Documents/Studium/Informatik/6_Semester/KP BigData/semantic-retrieval/data/fandom-qa/harry_potter_qa/triples.train.tsv"
queries_path = "/home/aaron/Documents/Studium/Informatik/6_Semester/KP BigData/semantic-retrieval/data/fandom-qa/harry_potter_qa/queries.train.tsv"
passages_path = "/home/aaron/Documents/Studium/Informatik/6_Semester/KP BigData/semantic-retrieval/data/fandom-qa/harry_potter_qa/passages.train.tsv"

data_iter = DataIterator(config, triples_path, queries_path, passages_path)

colbert = ColBERT(config, device=DEVICE)

optimizer = torch.optim.AdamW(colbert.parameters(), lr=5e-6, eps=1e-8)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(1, config.epochs+1):
    data_iter.shuffle()
    for i, batch in enumerate(tqdm(data_iter)):
        #B = sum(map(lambda x: x[0].shape[0], batch))
        optimizer.zero_grad()
        losses = 0
        accs = 0
        for sub_batch in batch:
            q_tokens, q_masks, p_tokens, p_masks = sub_batch
            Q, P = (q_tokens, q_masks), (p_tokens, p_masks)
            sub_B = q_tokens.shape[0]

            out = colbert(Q, P)        
            accs += torch.sum(out.detach().max(dim=-1).indices == torch.arange(0, sub_B, device=DEVICE, dtype=torch.long))        
            loss = criterion(out, torch.arange(0, sub_B, device=DEVICE, dtype=torch.long))
            loss *= 1 / config.batch_size
            loss.backward()

            losses += loss.item()
        
        optimizer.step()   

        writer.add_scalar("Loss/train", losses, epoch*len(data_iter) + i)
        writer.add_scalar("Acc/train", accs / config.batch_size, epoch*len(data_iter) + i)

    data_iter.reset()
