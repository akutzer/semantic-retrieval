#!/usr/bin/env python3
import os
from tqdm import tqdm
import torch

from retrieval.configs import FANDOM_CONFIG_DEFAULT
from retrieval.data import get_pytorch_dataloader, load_dataset
from retrieval.models import get_colbert_and_tokenizer
from retrieval.training.utils import seed, get_tensorboard_writer




if __name__ == "__main__":

    SEED = 125
    seed(SEED)

    DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    if DEVICE.type == "cuda":
        # enable TensorFloat32 tensor cores for float32 matrix multiplication if available
        torch.set_float32_matmul_precision("high")

    config = FANDOM_CONFIG_DEFAULT
    DATASET = "harry_potter"
    DATASET_PATH = "../../data/fandoms_qa/harry_potter/"

    # instantiation of GradScaler for automatic mixed precission
    if config.use_amp:
        scaler = torch.cuda.amp.GradScaler()
    
    # disable tokenizer parallelism if we are forking otherwise this warning occurs:
    #     huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
    #     To disable this warning, you can either:
    #             - Avoid using `tokenizers` before the fork if possible
    #             - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
    if config.num_workers > 0:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"


    # instantiation of TensorBoard logger
    # run the following command to view the live-metrics in the browser (http://localhost:6006/):
    #     tensorboard --logdir=runs
    writer = get_tensorboard_writer(config, DATASET)
 

    colbert, tokenizer = get_colbert_and_tokenizer(config, device=DEVICE)
    print("Loaded ColBERT!")
    train_dataset = load_dataset(config, os.path.join(DATASET_PATH, "train"), mode="QQP")
    eval_dataset = load_dataset(config, os.path.join(DATASET_PATH, "val"), mode="QQP")
    print("Loaded Dataset!")
    train_dataloader = get_pytorch_dataloader(config, train_dataset, tokenizer)
    eval_dataloader = get_pytorch_dataloader(config, eval_dataset, tokenizer, batch_size=config.batch_size, shuffle=False, drop_last=False, num_workers=0)
    print("Initialized DataLoader!")

    optimizer = torch.optim.AdamW(colbert.parameters(), lr=5e-6, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False, fused=False)
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    print("Starting training!")
    for epoch in range(1, config.epochs+1):
        optimizer.zero_grad()
        losses, accs = 0, 0

        colbert.train()
        for i, (Q, P) in enumerate(tqdm(train_dataloader)):
            (q_tokens, q_masks), (p_tokens, p_masks) = Q, P
            sub_B = min(q_tokens.shape[0], p_tokens.shape[0])

            # forward pass through the model & calculate loss
            with torch.autocast(DEVICE.type, enabled=config.use_amp):
                out = colbert(Q, P)
                # print(out.shape)
                # target is the 0-th aka first question from the list of queries given to a passage
                target = torch.zeros(sub_B, device=out.device, dtype=torch.long)
                loss = criterion(out, target)
                loss *= 1 / config.batch_size

            # calculate & accumulate gradients, the update step is done after the entire batch
            # has been passed through the model
            if config.use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            with torch.inference_mode():
                losses += loss.item()
                # calculate the accuracy within a subbatch -> extremly inflated accuracy
                accs += torch.sum(out.detach().max(dim=-1).indices == torch.zeros(sub_B, device=out.device, dtype=torch.long))
            
            # after accum_steps, update the weights + log the metrics
            if (i + 1) % config.accum_steps == 0:
                # update model parameters
                if config.use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

                # TODO: check if calculation is correct
                time_step = (epoch - 1) * (len(train_dataloader) // config.accum_steps)  + i // config.accum_steps
                writer.add_scalar("Loss/train", losses, time_step)
                writer.add_scalar("Accuracy/train", accs / config.batch_size, time_step)
                losses, accs = 0, 0

            if i > 10:
                break
        
        colbert.save("uwu")
        tokenizer.save("uwu", store_config=False)
        
        # evaluate the model on the entire validation set after each epoch
        colbert.eval()
        with torch.inference_mode():
            loss = 0.0
            acc = 0.0

            for j, (Q, P) in enumerate(tqdm(eval_dataloader)):
                (q_tokens, q_masks), (p_tokens, p_masks) = Q, P
                sub_B = min(q_tokens.shape[0], p_tokens.shape[0])

                # forward pass through the model & calculate loss
                with torch.autocast(DEVICE.type, enabled=config.use_amp):
                    out = colbert(Q, P)
                    # print(out.shape)
                    # target is the 0-th aka first question from the list of queries given to a passage
                    target = torch.zeros(sub_B, device=out.device, dtype=torch.long)
                    loss += criterion(out, target)
                    acc += torch.sum(out.max(dim=-1).indices == torch.zeros(sub_B, device=out.device, dtype=torch.long))
            
            loss /= len(eval_dataset)
            acc /= len(eval_dataset)

            print(loss, acc)




