
from retrieval.indexing import ColBERTIndexer, FaissIndex


class ColBERTRetriever:

    def __init__(self, config, indicies_path):
        if indicies_path.endswith((".pt", ".pth")):
            pass
        elif indicies_path.endswith(".tsv"):
            pass
        else:
            raie ValueError
        self.inference = None
        self.indexer = ColBERTIndexer(config, device="cpu")

    def retrieve(query: str):
        Q = self.indexer.inference.query_from_text([query])
        # self.indexer.
        pass