#!/usr/bin/env python3
import os
from tqdm import tqdm
import torch

from retrieval.configs import FANDOM_CONFIG_DEFAULT
from retrieval.data import get_pytorch_dataloader, load_dataset
from retrieval.models import get_colbert_and_tokenizer
from retrieval.training.utils import seed, get_tensorboard_writer, get_run_name


def validation(model, criterion, dataloader, config):
    model.eval()
    with torch.inference_mode():
        eval_loss = 0.0
        eval_acc = 0.0

        for Q, P in dataloader:
            # forward pass through the model & calculate loss
            with torch.autocast(DEVICE.type, enabled=config.use_amp):
                out = model(Q, P)
                # target is the 0-th aka first question from the list of queries given to a passage
                target = torch.zeros(out.shape[0], device=out.device, dtype=torch.long)
                eval_loss += criterion(out, target)
                eval_acc += torch.sum(out.max(dim=-1).indices == target)
        
        eval_loss /= len(eval_dataset)
        eval_acc /= len(eval_dataset)

    model.train()
    return eval_loss, eval_acc


if __name__ == "__main__":

    SEED = 125
    seed(SEED)

    DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    if "cuda" in DEVICE.type:
        # enable TensorFloat32 tensor cores for float32 matrix multiplication if available
        torch.set_float32_matmul_precision("high")
        print("[LOGGING] Enabled TensorFloat32 calculations!")

    config = FANDOM_CONFIG_DEFAULT
    DATASET_NAME = "harry_potter"
    DATASET_PATH = "../../data/fandoms_qa/harry_potter/"
    RUN_NAME = get_run_name(config, DATASET_NAME)

    N_EVAL_PER_EPOCH = 5

    # instantiation of GradScaler for automatic mixed precission
    if config.use_amp:
        scaler = torch.cuda.amp.GradScaler()
        print("[LOGGING] Enabled AMP!")
    
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
    writer = get_tensorboard_writer(RUN_NAME)
 

    colbert, tokenizer = get_colbert_and_tokenizer(config, device=DEVICE)
    print("[LOGGING] Loaded ColBERT!")
    print(tokenizer)
    print(colbert)
    train_dataset = load_dataset(config, os.path.join(DATASET_PATH, "train"), mode="QQP")
    eval_dataset = load_dataset(config, os.path.join(DATASET_PATH, "val"), mode="QQP")
    print("[LOGGING] Loaded Dataset!")
    train_dataloader = get_pytorch_dataloader(config, train_dataset, tokenizer)
    eval_dataloader = get_pytorch_dataloader(config, eval_dataset, tokenizer, batch_size=config.batch_size, shuffle=False, drop_last=False, num_workers=0)
    print("[LOGGING] Initialized DataLoader!")

    optimizer = torch.optim.AdamW(colbert.parameters(), lr=5e-6, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False, fused=False)
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")


    print("[LOGGING] Starting initial evaluation!")
    # evaluate the model on the entire validation set after each epoch
    eval_loss, eval_acc = validation(colbert, criterion, eval_dataloader, config)
    print(eval_loss.item(), eval_acc.item())
    
    # write the metric to the tensorboard logger
    writer.add_scalar("Loss/eval", eval_loss, 0)
    writer.add_scalar("Accuracy/eval", eval_acc, 0)


    print("[LOGGING] Starting training!")
    for epoch in range(1, config.epochs+1):
        print(f"[LOGGING] Starting epoch: {epoch}")
        optimizer.zero_grad()
        losses, accs = 0, 0

        # calculate after which steps we need to run the evaluation
        # and make sure to run the last evaluation after the last iteration of
        # the training dataloader
        eval_iterations = [i * (len(train_dataloader) // N_EVAL_PER_EPOCH) for i in range(1, N_EVAL_PER_EPOCH + 1)]
        eval_iterations[-1] = len(train_dataloader)

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

                # write the metric to the tensorboard logger
                time_step = (epoch - 1) * (len(train_dataloader) // config.accum_steps)  + i // config.accum_steps
                writer.add_scalar("Loss/train", losses, time_step)
                writer.add_scalar("Accuracy/train", accs / config.batch_size, time_step)
                losses, accs = 0, 0
            

            if (i + 1) in eval_iterations:
                # evaluate the model on the entire validation set after each epoch
                eval_loss, eval_acc = validation(colbert, criterion, eval_dataloader, config)
                print(eval_loss.item(), eval_acc.item())
                
                # write the metric to the tensorboard logger
                time_step = (epoch - 1) * (len(train_dataloader) // config.accum_steps)  + i // config.accum_steps
                writer.add_scalar("Loss/eval", eval_loss, time_step)
                writer.add_scalar("Accuracy/eval", eval_acc, time_step)


        # shitty checkpointing
        checkpoint_path = f"../../checkpoint/{RUN_NAME}/epoch{epoch}_loss%.5f_acc%.3f" % (eval_loss.item(), eval_acc.item() * 100)
        colbert.save(checkpoint_path)
        tokenizer.save(checkpoint_path, store_config=False)
    

    print("[LOGGING] Training finished")
