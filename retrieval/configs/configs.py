from dataclasses import dataclass
from retrieval.configs.settings import *


@dataclass
class BaseConfig(TokenizerSettings, DocSettings, QuerySettings, TrainingSettings, ModelSettings):
    pass
