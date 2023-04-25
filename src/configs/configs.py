from dataclasses import dataclass
from .settings import *


@dataclass
class BaseConfig(TokenizerSettings, DocSettings, QuerySettings):
    pass
