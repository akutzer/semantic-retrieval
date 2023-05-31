import random
from datetime import datetime

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from retrieval.configs import BaseConfig


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
    time = datetime.now().isoformat(timespec="seconds")

    if "roberta" in args.backbone:
        backbone_name = "roberta"
    elif "colbertv2" in args.backbone:
        backbone_name = "colbertv2"
    elif "bert" in args.backbone:
        backbone_name = "bert"
    else:
        backbone_name = args.backbone
    
    run_name = f"{args.dataset_name}_{backbone_name}_{time}"
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
        num_gpus=args.num_gpus
    )

    return config