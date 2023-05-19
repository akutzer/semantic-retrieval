#!/usr/bin/env python3
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
from torch.cuda.amp import autocast as autocast
import matplotlib.pyplot as plt

SEED = 125
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


MODEL_PATH = "roberta-base" # "bert-base-uncased" or "../../data/colbertv2.0/" or "roberta-base"
DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"

config = BaseConfig(
    tok_name_or_path=MODEL_PATH,
    backbone_name_or_path=MODEL_PATH,
    passages_per_query = 1,
    epochs = 10,
    lr_warmup_epochs=3,
    lr_warmup_decay=0.3333333333333333,
    bucket_size = 16*10,
    batch_size = 16,
    accum_steps = 2,    # sub_batch_size = ceil(batch_size / accum_steps)
    similarity="cosine",
    intra_batch_similarity=True,
    num_hidden_layers = 12,
    num_attention_heads = 12,
    dropout = 0.1,
    dim=32)

writer = SummaryWriter()

colbert, tokenizer = get_colbert_and_tokenizer(config, device=DEVICE)

triples_path = "../../data/fandom-qa/witcher_qa/triples.train.tsv"
queries_path = "../../data/fandom-qa/witcher_qa/queries.train.tsv"
passages_path = "../../data/fandom-qa/witcher_qa/passages.train.tsv"
dataset = TripleDataset(config, triples_path, queries_path, passages_path, mode="QPP")
bucket_iter = BucketIterator(config, dataset, tokenizer)

# add scheduler with sequential learning rate
optimizer = torch.optim.AdamW(colbert.parameters(), lr=1e-5, eps=1e-8)
main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs-config.lr_warmup_epochs, verbose=True)
warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=config.lr_warmup_decay, total_iters=config.lr_warmup_epochs, verbose=True)
lr_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[config.lr_warmup_epochs])
criterion = torch.nn.CrossEntropyLoss()

# instantiation of GradScaler
scaler = torch.cuda.amp.GradScaler()

lr = []
for epoch in range(1, config.epochs+1):
    bucket_iter.shuffle()
    for i, bucket in enumerate(tqdm(bucket_iter)):
        optimizer.zero_grad()
        losses, accs = 0, 0
        for j, batch in enumerate(bucket):
            Q, P = batch
            (q_tokens, q_masks), (p_tokens, p_masks) = Q, P
            sub_B = q_tokens.shape[0]

            with autocast():
                out = colbert(Q, P)
                loss = criterion(out, torch.arange(0, sub_B, device=DEVICE, dtype=torch.long))
                loss *= 1 / config.batch_size

                # calculate & accumulate gradients, the update step is done after the entire batch
                # has been passed through the model
                # loss.backward()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            with torch.inference_mode():
                losses += loss.item()
                # calculate the accuracy within a subbatch -> extremly inflated accuracy
                accs += torch.sum(out.detach().max(dim=-1).indices == torch.arange(0, sub_B, device=DEVICE, dtype=torch.long))
            
            # after accum_steps, update the weights + log the metrics
            if (j + 1) % config.accum_steps == 0:
                # update model parameters
                optimizer.step()
                optimizer.zero_grad()
                lr.append(lr_scheduler.get_last_lr())

                # TODO: check if calculation is correct
                writer.add_scalar("Loss/train", losses, (epoch-1)*len(bucket_iter) + i*config.bucket_size/config.batch_size + j/config.accum_steps)
                writer.add_scalar("Accuracy/train", accs / config.batch_size, (epoch-1)*len(bucket_iter) + i*config.bucket_size/config.batch_size + j/config.accum_steps)
                losses, accs = 0, 0

    # print("learning rate of %d epoch: %f" % (epoch, optimizer.param_groups[0]['lr']))
    # update learning rate
    lr_scheduler.step() 
    bucket_iter.reset()

plt.figure()
plt.plot(np.arange(len(lr)), lr, label='roberta-base_witcher-qa', color='b')
plt.title('roberta-base_witcher-qa')
plt.xlabel('steps')
plt.ylabel('learning rate')
plt.legend()
plt.show()
plt.savefig('roberta-base_witcher-qa.jpg')
