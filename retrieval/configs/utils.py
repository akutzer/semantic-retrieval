import os
import json
from dataclasses import asdict, is_dataclass

from retrieval.configs.configs import BaseConfig

    
def save_config(config: BaseConfig, config_path: str):
    # save the model's config if available
    if is_dataclass(config):
        config_dict = asdict(config)
        with open(config_path, "w", encoding="utf-8") as config_file:
            json.dump(config_dict, config_file)
        return True
        
    return False


def load_config(config_path: str):
    # load the model's config if available
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as config_file:
            config_dict = json.load(config_file)
        config = BaseConfig(**config_dict)
        return config

    return False