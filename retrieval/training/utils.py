import random
import datetime

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter


def seed(seed: int = 125):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_tensorboard_writer(config, dataset_name: str):
    # instantiation of TensorBoard logger
    iso_date = datetime.datetime.now().isoformat()[:-6]

    if "bert" in config.backbone_name_or_path:
        backbone_name = "bert"
    elif "roberta" in config.backbone_name_or_path:
        backbone_name = "roberta"
    elif "colbertv2" in config.backbone_name_or_path:
        backbone_name = "colbertv2"
    else:
        backbone_name = config.backbone_name_or_path

    directory = f"{dataset_name}-{iso_date}-{backbone_name}_{config.similarity}"
    writer = SummaryWriter(comment=directory)

    return writer