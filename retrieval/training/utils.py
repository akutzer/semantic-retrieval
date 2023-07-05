import os
import random
from datetime import datetime
import logging

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from retrieval.configs import BaseConfig
from retrieval.models import ColBERT


def seed(seed: int = 125):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_run_name(args):
    time = datetime.now().isoformat(timespec="milliseconds")

    if "roberta" in args.backbone:
        backbone_name = "roberta"
    elif "colbertv2" in args.backbone:
        backbone_name = "colbertv2"
    elif "bert" in args.backbone:
        backbone_name = "bert"
    else:
        backbone_name = args.backbone

    run_name = f"{args.dataset_name}_{backbone_name}_{args.similarity}_{args.dim}_{time}"
    return run_name


def get_tensorboard_writer(run_name: str, path: str = "runs"):
    directory = f"{path}/{run_name}"
    writer = SummaryWriter(log_dir=directory)

    return writer


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
        checkpoint=args.checkpoint,

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
        lr=args.learning_rate,
        warmup_epochs=args.warmup_epochs,
        warmup_start_factor=args.warmup_start_factor,
        use_amp=args.use_amp,
        num_gpus=args.num_gpus,
    )

    return config


def load_optimizer_checkpoint(directory: str, optimizer: torch.optim.Optimizer):
    logging.basicConfig(level=logging.WARNING, format="[%(asctime)s][%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    optim_path = os.path.join(directory, "optimizer.pt")

    if os.path.exists(optim_path):
        state_dict = torch.load(optim_path)
        optimizer.load_state_dict(state_dict)
        logging.info("Loaded optimizer checkpoint!")
    else:
        logging.warning(
            f"Could not load optimizer checkpoint, because the path `{optim_path}` does not exist."
            " Returning the given optimizer."
        )
    
    return optimizer


def load_scheduler_checkpoint(directory: str, scheduler: torch.optim.lr_scheduler.LRScheduler):
    logging.basicConfig(level=logging.WARNING, format="[%(asctime)s][%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    scheduler_path = os.path.join(directory, "scheduler.pt")
    

    if os.path.exists(scheduler_path):
        state_dict = torch.load(scheduler_path)
        scheduler.load_state_dict(state_dict)
        logging.info("Loaded scheduler checkpoint!")
    else:
        logging.warning(
            f"Could not load scheduler, because the path `{scheduler_path}` does not exist."
            " Returning the given scheduler."
        )
    
    return scheduler

def load_grad_scaler_checkpoint(directory: str, scaler: torch.cuda.amp.GradScaler):
    logging.basicConfig(level=logging.WARNING, format="[%(asctime)s][%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    scaler_path = os.path.join(directory, "gradient_scaler.pt")

    if os.path.exists(scaler_path):
        state_dict = torch.load(scaler_path)
        scaler.load_state_dict(state_dict)
        logging.info("Loaded gradient scaler checkpoint!")
    else:
        logging.warning(
            f"Could not load gradient scaler checkpoint, because the path `{scaler_path}` does not exist."
            " Returning the given gradient scaler."
        )
    
    return scaler


def freeze_until_layer(colbert: ColBERT, layer: int):
    """
    Freezes the backbone weights up to a certain layer.
    """
    freeze_parameters = ["embeddings"] + [f"encoder.layer.{i}." for i in range(1, layer + 1)]

    for para_name, parameters in colbert.backbone.named_parameters():
        for freeze_para in freeze_parameters:
            if freeze_para in para_name:
                parameters.requires_grad = False
                break

    return colbert
