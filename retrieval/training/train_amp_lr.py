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


SEED = 125
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


MODEL_PATH = "bert-base-uncased" # "bert-base-uncased" or "../../data/colbertv2.0/" or "roberta-base"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

config = BaseConfig(
    tok_name_or_path=MODEL_PATH,
    backbone_name_or_path=MODEL_PATH,
    passages_per_query = 1,
    epochs = 3,
    lr_warmup_epochs = 1,
    lr_warmup_decay = 0.3333333333333333,
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

# sequential lr_scheduler learning rate decay
optimizer = torch.optim.AdamW(colbert.parameters(), lr=5e-6, eps=1e-8)
main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs-config.lr_warmup_epochs, verbose=True)
warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=config.lr_warmup_decay, total_iters=config.lr_warmup_epochs, verbose=True)
lr_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[config.lr_warmup_epochs])
criterion = torch.nn.CrossEntropyLoss()

# instantiation of GradScaler
scaler = torch.cuda.amp.GradScaler()

def check_loss(loss):
    return loss != loss

for epoch in range(1, config.epochs+1):
    bucket_iter.shuffle()
    for i, bucket in enumerate(tqdm(bucket_iter)):
        optimizer.zero_grad()
        losses, accs = 0, 0
        for j, batch in enumerate(bucket):
            Q, P = batch
            (q_tokens, q_masks), (p_tokens, p_masks) = Q, P
            sub_B = q_tokens.shape[0]

            # out = colbert(Q, P)
            # loss = criterion(out, torch.arange(0, sub_B, device=DEVICE, dtype=torch.long))
            # loss *= 1 / config.batch_size

            # # calculate & accumulate gradients, the update step is done after the entire batch
            # # has been passed through the model
            # loss.backward()

            with autocast():
                out = colbert(Q, P)
                loss = criterion(out, torch.arange(0, sub_B, device=DEVICE, dtype=torch.long))
                loss *= 1 / config.batch_size
                
            # Backprop weights / gradients scaling
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

                # TODO: check if calculation is correct
                writer.add_scalar("Loss/train", losses, (epoch-1)*len(bucket_iter) + i*config.bucket_size/config.batch_size + j/config.accum_steps)
                writer.add_scalar("Accuracy/train", accs / config.batch_size, (epoch-1)*len(bucket_iter) + i*config.bucket_size/config.batch_size + j/config.accum_steps)
                losses, accs = 0, 0

    print('Epoch {} | Losses {} | Accs {}'.format(epoch, losses.item(), accs.item()))
    lr_scheduler.step() 
    bucket_iter.reset()

