#!/usr/bin/env python3
from dataclasses import dataclass
from retrieval.configs.settings import *


@dataclass
class BaseConfig(TokenizerSettings, DocSettings, QuerySettings, DataLoaderSettings, TrainingSettings, ModelSettings):
    pass
