#!/usr/bin/env python3
import os
import torch
import torch.cuda.amp as amp
import torch.nn as nn
from tqdm import tqdm
import logging
import argparse
from pathlib import Path

from retrieval.configs import FANDOM_DEFAULT_CONFIG, BaseConfig
from retrieval.data import get_pytorch_dataloader, load_dataset, TripleDataset
from retrieval.models import get_colbert_and_tokenizer
from retrieval.training.utils import seed, get_tensorboard_writer, get_run_name


def validation(model, criterion, dataloader):
    model.eval()
    with torch.inference_mode():
        loss = 0.0
        acc = 0.0
        mrr = 0.0

        for Q, P in dataloader:
            with torch.autocast(model.device.type, enabled=model.config.use_amp):
                out = model(Q, P)
                target = torch.zeros(out.shape[0], device=out.device, dtype=torch.long)
                loss += criterion(out, target)
                acc += torch.sum(out.max(dim=-1).indices == target)
                ranks = out.sort(dim=-1, descending=True).indices
                positive_rank = torch.where(ranks == 0)[-1]
                mrr += torch.sum(1.0 / (positive_rank + 1).float())
        
        n = len(dataloader.dataset)
        loss /= n
        acc /= n
        mrr /= n

    model.train()
    return loss, acc, mrr


def train(args):
    # INITIALIZATION OF UTILITIES
    run_name = get_run_name(args)

    # Set up logging
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    logging.info(f"Starting run: {run_name}")


    # disable parallelism for huggingface's fast-tokenizers in case we use multiple
    # subprocesses for the dataloading, otherwise we can encounter deadlocks
    if args.train_workers > 0 or args.val_workers > 0:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        logging.info("Disabled tokenizer parallelism!")

    # TODO: seed dataloader too
    seed(args.seed)

    # load config using the given arguments
    config = get_config_from_argparser(args)
    logging.info("Loaded config!")
    logging.info(config)

    use_gpu = config.num_gpus > 0 and torch.cuda.is_available()
    device = torch.device("cuda" if use_gpu else "cpu")
    logging.info(f"Running on {device.type}!")
    if use_gpu:
        torch.set_float32_matmul_precision("high")
        logging.info("Enabled TensorFloat32 calculations!")


    if config.use_amp:
        scaler = amp.GradScaler()
        logging.info("Enabled AMP!")


    writer = get_tensorboard_writer(run_name, path=args.tensorboard_path)
    logging.info("Initialized TensorBoard @ `http://localhost:6006/`!")
 
    colbert, tokenizer = get_colbert_and_tokenizer(config, device)
    logging.info("Loaded ColBERT!")
    logging.info(tokenizer)
    logging.info(colbert)

    # TODO: rewrite get_pytorch_dataloader and load_dataset
    run_eval = args.triples_path_val is not None and args.num_eval_per_epoch > 0
    train_dataset = TripleDataset(config, args.triples_path_train, args.queries_path_train, args.passages_path_train, mode=args.dataset_mode)
    train_dataloader = get_pytorch_dataloader(config, train_dataset, tokenizer, num_workers=args.train_workers)
    logging.info("Loaded Training-Dataset & initialized Training-DataLoader!")
    if run_eval:
        eval_dataset = TripleDataset(config, args.triples_path_val, args.queries_path_val, args.passages_path_val, mode=args.dataset_mode)
        eval_dataloader = get_pytorch_dataloader(config, eval_dataset, tokenizer, batch_size=config.batch_size, shuffle=False, drop_last=False, num_workers=args.val_workers)
        logging.info("Loaded Evaluation-Dataset & initialized Evaluation-DataLoader!")

        eval_iterations = [i * (len(train_dataloader) // args.num_eval_per_epoch) for i in range(1, args.num_eval_per_epoch + 1)]
        eval_iterations[-1] = len(train_dataloader)
    
    if args.checkpoints_per_epoch > 0:
        checkpoint_iterations = [i * (len(train_dataloader) // args.checkpoints_per_epoch) for i in range(1, args.checkpoints_per_epoch + 1)]
        checkpoint_iterations[-1] = len(train_dataloader)
    else:
        checkpoint_iterations = [len(train_dataloader)]


    optimizer = torch.optim.AdamW(colbert.parameters(), lr=5e-6, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
    criterion = nn.CrossEntropyLoss(reduction="sum")

    if run_eval:
        logging.info("Starting initial evaluation!")
        eval_loss, eval_acc, eval_mrr = validation(colbert, criterion, eval_dataloader)
        logging.info(f"Eval Loss: {eval_loss.item()}, Eval Accuracy: {eval_acc.item()}, Eval MRR: {eval_mrr.item()}")
        writer.add_scalar("Loss/eval", eval_loss, 0)
        writer.add_scalar("Accuracy/eval", eval_acc, 0)
        writer.add_scalar("MRR/eval", eval_mrr, 0)       
    

    logging.info("Starting training!")
    for epoch in range(1, config.epochs + 1):
        logging.info(f"Starting epoch: {epoch}")
        optimizer.zero_grad()
        losses, accs, mrr = 0.0, 0.0, 0.0

        colbert.train()
        for i, (Q, P) in enumerate(tqdm(train_dataloader)):
            with torch.autocast(device.type, enabled=config.use_amp):
                out = colbert(Q, P)
                target = torch.zeros(out.shape[0], device=out.device, dtype=torch.long)
                loss = criterion(out, target)
                loss *= 1 / config.batch_size

            if config.use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            with torch.inference_mode():
                losses += loss.item()
                accs += torch.sum(out.detach().max(dim=-1).indices == torch.zeros(out.shape[0], device=out.device, dtype=torch.long))
                ranks = out.sort(dim=-1, descending=True).indices
                positive_rank = torch.where(ranks == 0)[-1]
                mrr += torch.sum(1.0 / (positive_rank + 1).float())
            
            if (i + 1) % config.accum_steps == 0:
                if config.use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

                time_step = (epoch - 1) * (len(train_dataloader) // config.accum_steps)  + i // config.accum_steps
                writer.add_scalar("Loss/train", losses, time_step)
                writer.add_scalar("Accuracy/train", accs / config.batch_size, time_step)
                writer.add_scalar("MRR/train", mrr / config.batch_size, time_step)
                losses, accs, mrr = 0.0, 0.0, 0.0
            

            if run_eval and (i + 1) in eval_iterations:
                eval_loss, eval_acc, eval_mrr = validation(colbert, criterion, eval_dataloader)
                logging.info(f"Eval Loss: {eval_loss.item()}, Eval Accuracy: {eval_acc.item()}, Eval MRR: {eval_mrr.item()}")
                time_step = (epoch - 1) * (len(train_dataloader) // config.accum_steps)  + i // config.accum_steps
                writer.add_scalar("Loss/eval", eval_loss, time_step)
                writer.add_scalar("Accuracy/eval", eval_acc, time_step)
                writer.add_scalar("MRR/eval", eval_mrr, time_step)

            if (i + 1) in checkpoint_iterations:
                checkpoint_in_epoch = checkpoint_iterations.index(i + 1) + 1
                checkpoint_path = f"{args.checkpoints_path}/{run_name}/epoch{epoch}_{checkpoint_in_epoch}"
                if run_eval and (i + 1) in eval_iterations:
                    checkpoint_path += "_loss%.4f_mrr%.4f_acc%.3f" % (eval_loss.item(), eval_mrr.item(), eval_acc.item() * 100)
                colbert.save(checkpoint_path)
                tokenizer.save(checkpoint_path, store_config=False)
                logging.info(f"Saved checkpoint: {checkpoint_path}")

    writer.flush()
    writer.close()


def get_config_from_argparser(args):
    config = BaseConfig(
        # TokenizerSettings
        tok_name_or_path=args.backbone,

        # ModelSettings
        backbone_name_or_path=args.backbone,
        dim=args.dim,
        dropout=args.dropout,
        skip_punctuation=True,
        similarity=args.similarity,
        normalize=args.normalize,

        # DocSettings/QuerySettings
        doc_maxlen=args.doc_maxlen,
        query_maxlen=args.query_maxlen,

        # DataLoaderSettings
        batch_size=args.batch_size,
        accum_steps=args.accum_steps,
        passages_per_query=args.passages_per_query,
        shuffle=args.shuffle,
        drop_last=args.drop_last,
        pin_memory=True,

        # TrainingSettings
        epochs=args.epochs,
        use_amp=args.use_amp,
        num_gpus=args.num_gpus
    )

    return config





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ColBERT Training")

    # Dataset arguments
    dataset_args = parser.add_argument_group("Dataset Arguments")
    dataset_args.add_argument("--dataset-name", type=str, required=True, help="Name of the dataset")
    dataset_args.add_argument("--dataset-mode", type=str, required=True, choices=["QQP", "QPP"], help="Mode of the dataset")
    dataset_args.add_argument("--passages-path-train", type=str, required=True, help="Path to the training passages.tsv file")
    dataset_args.add_argument("--queries-path-train", type=str, required=True, help="Path to the training queries.tsv file")
    dataset_args.add_argument("--triples-path-train", type=str, required=True, help="Path to the training triples.tsv file")
    dataset_args.add_argument("--passages-path-val", type=str, help="Path to the validation passages.tsv file")
    dataset_args.add_argument("--queries-path-val", type=str, help="Path to the validation queries.tsv file")
    dataset_args.add_argument("--triples-path-val", type=str, help="Path to the validation triples.tsv file")

    # Dataloader arguments
    dataloader_args = parser.add_argument_group("Dataloader Arguments")
    dataloader_args.add_argument("--doc-maxlen", type=int, default=220, help="Maximum length of document input")
    dataloader_args.add_argument("--query-maxlen", type=int, default=32, help="Maximum length of query input")
    dataloader_args.add_argument("--train-workers", type=int, default=0, help="Number of workers for training data loading")
    dataloader_args.add_argument("--val-workers", type=int, default=0, help="Number of workers for validation data loading")
    dataloader_args.add_argument("--passages-per-query", type=int, default=10, help="Number of passages per query")
    dataloader_args.add_argument("--shuffle", action="store_true", help="Shuffle the data during loading")
    dataloader_args.add_argument("--drop-last", action="store_true", help="Drop the last batch if it's incomplete")
    dataloader_args.add_argument("--pin-memory", action="store_true", help="Pin memory for data loading")

    # Model arguments
    model_args = parser.add_argument_group("Model Arguments")
    model_args.add_argument("--backbone", type=str, help="Name of the backbone model or path to its checkpoint")
    model_args.add_argument("--dim", type=int, help="Size of the embedding vectors")
    model_args.add_argument("--dropout", type=float, help="Dropout rate")
    model_args.add_argument("--similarity", type=str, choices=["cosine", "L2"], default="cosine", help="Similarity function")
    model_args.add_argument("--normalize", action="store_true", help="Normalize the embeddings")

    # Training arguments
    training_args = parser.add_argument_group("Training Arguments")
    training_args.add_argument("--epochs", type=int, required=True, help="Number of training epochs")
    training_args.add_argument("--batch-size", type=int, required=True, help="Training batch size")
    training_args.add_argument("--accum-steps", type=int, default=1, help="Number of gradient accumulation steps")
    training_args.add_argument("--seed", type=int, default=125, help="Random seed")
    training_args.add_argument("--num-eval-per-epoch", type=int, default=1, help="Number of evaluation runs per epoch")
    training_args.add_argument("--checkpoints-per-epoch", type=int, default=1, help="Number of checkpoints to save per epoch")
    training_args.add_argument("--use-amp", action="store_true", help="Use automatic mixed precision training")
    training_args.add_argument("--num-gpus", type=int, default=0, help="Number of GPUs to use for training")
    training_args.add_argument("--checkpoints-path", type=str, default="/checkpoints", help="")
    training_args.add_argument("--tensorboard-path", type=str, default="/runs", help="")

    args = parser.parse_args()    
    train(args)
