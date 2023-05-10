from abc import ABC, abstractmethod
from typing import Union, List


class IndexerInterface(ABC):
    @abstractmethod
    def index(self, path_to_passages: str, bsize: Union[None, int] = None):
        pass
    
    @abstractmethod
    def search(self, query):
        pass

    @abstractmethod
    def save(self, path):
        pass

    @abstractmethod
    def load(self, path):
        pass

    @abstractmethod
    def get_pid_embedding(self, pid):
        pass
