import random
from datetime import datetime

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter


def seed(seed: int = 125):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

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