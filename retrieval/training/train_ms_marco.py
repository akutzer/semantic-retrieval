#!/usr/bin/env python3
import os
from tqdm import tqdm
import torch

from retrieval.configs import MS_MARCO_DEFAULT_CONFIG
from retrieval.data import get_pytorch_dataloader, load_dataset
from retrieval.models import get_colbert_and_tokenizer
from retrieval.training.utils import seed, get_tensorboard_writer, get_run_name


def validation(model, criterion, dataloader, config):
    model.eval()
    with torch.inference_mode():
        loss = 0.0
        acc = 0.0
        mrr = 0.0

        for Q, P in tqdm(dataloader):
            # forward pass through the model & calculate loss
            with torch.autocast(DEVICE.type, enabled=config.use_amp):
                out = model(Q, P)
                # target is the 0-th aka first question from the list of queries given to a passage
                target = torch.zeros(out.shape[0], device=out.device, dtype=torch.long)
                loss += criterion(out, target)
                acc += torch.sum(out.max(dim=-1).indices == target)
                ranks = out.sort(dim=-1, descending=True).indices
                positiv_rank = torch.where(ranks == 0)[-1]
                mrr += torch.sum(1.0 / (positiv_rank + 1).float())
        
        loss /= len(eval_dataset)
        acc /= len(eval_dataset)
        mrr /= len(eval_dataset)

    model.train()
    return loss, acc, mrr


if __name__ == "__main__":

    SEED = 125
    seed(SEED)

    DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    if "cuda" in DEVICE.type:
        # enable TensorFloat32 tensor cores for float32 matrix multiplication if available
        torch.set_float32_matmul_precision("high")
        print("[LOGGING] Enabled TensorFloat32 calculations!")

    config = MS_MARCO_DEFAULT_CONFIG
    config.batch_size = 32

    DATASET_NAME = "ms_marco_v1.1"
    DATASET_PATH = "../../data/ms_marco_v1.1"
    RUN_NAME = get_run_name(config, DATASET_NAME)

    N_EVAL_PER_EPOCH = 6
    N_CHECKPOINTS_PER_EPOCH = 2
    assert N_EVAL_PER_EPOCH % N_CHECKPOINTS_PER_EPOCH == 0

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
    train_dataset = load_dataset(config, os.path.join(DATASET_PATH, "train"), mode="QPP")
    eval_dataset = load_dataset(config, os.path.join(DATASET_PATH, "validation"), mode="QPP")
    print("[LOGGING] Loaded Dataset!")
    train_dataloader = get_pytorch_dataloader(config, train_dataset, tokenizer)
    eval_dataloader = get_pytorch_dataloader(config, eval_dataset, tokenizer, batch_size=config.batch_size, shuffle=False, drop_last=False, num_workers=0)
    print("[LOGGING] Initialized DataLoader!")

    optimizer = torch.optim.AdamW(colbert.parameters(), lr=5e-6, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False, fused=False)
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")


    print("[LOGGING] Starting initial evaluation!")
    # evaluate the model on the entire validation set after each epoch
    eval_loss, eval_acc, eval_mrr = validation(colbert, criterion, eval_dataloader, config)
    print(eval_loss.item(), eval_acc.item(), eval_mrr.item())
    
    # write the metric to the tensorboard logger
    writer.add_scalar("Loss/eval", eval_loss, 0)
    writer.add_scalar("Accuracy/eval", eval_acc, 0)
    writer.add_scalar("MRR/eval", eval_mrr, 0)


    print("[LOGGING] Starting training!")
    for epoch in range(1, config.epochs+1):
        print(f"[LOGGING] Starting epoch: {epoch}")
        optimizer.zero_grad()
        losses, accs, mrr = 0.0, 0.0, 0.0

        # calculate after which steps we need to run the evaluation & save a checkpoint
        # and make sure to run the last evaluation after the last iteration of
        # the training dataloader
        eval_iterations = [i * (len(train_dataloader) // N_EVAL_PER_EPOCH) for i in range(1, N_EVAL_PER_EPOCH + 1)]
        eval_iterations[-1] = len(train_dataloader)

        checkpoint_iterations = [i * (len(train_dataloader) // N_CHECKPOINTS_PER_EPOCH) for i in range(1, N_CHECKPOINTS_PER_EPOCH + 1)]
        checkpoint_iterations[-1] = len(train_dataloader)

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
                ranks = out.sort(dim=-1, descending=True).indices
                positiv_rank = torch.where(ranks == 0)[-1]
                mrr += torch.sum(1.0 / (positiv_rank + 1).float())
            
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
                writer.add_scalar("MRR/train", mrr / config.batch_size, time_step)
                losses, accs, mrr = 0.0, 0.0, 0.0
            

            if (i + 1) in eval_iterations:
                # evaluate the model on the entire validation set after each epoch
                eval_loss, eval_acc, eval_mrr = validation(colbert, criterion, eval_dataloader, config)
                print(eval_loss.item(), eval_acc.item(), eval_mrr.item())
                
                # write the metric to the tensorboard logger
                time_step = (epoch - 1) * (len(train_dataloader) // config.accum_steps)  + i // config.accum_steps
                writer.add_scalar("Loss/eval", eval_loss, time_step)
                writer.add_scalar("Accuracy/eval", eval_acc, time_step)
                writer.add_scalar("MRR/eval", eval_mrr, time_step)
            
            if (i + 1) in checkpoint_iterations:
                # shitty checkpointing
                checkpoint_in_epoch = checkpoint_iterations.index(i + 1) + 1
                checkpoint_path = f"../../checkpoint/{RUN_NAME}/epoch{epoch}_{checkpoint_in_epoch}_loss%.4f_mrr%.4f_acc%.3f" % (eval_loss.item(), eval_mrr.item(), eval_acc.item() * 100)
                colbert.save(checkpoint_path)
                tokenizer.save(checkpoint_path, store_config=False)
    

    print("[LOGGING] Training finished")
