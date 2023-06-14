#!/usr/bin/env python3
import time
import faiss

from retrieval.indexing.colbert_indexer import ColBERTIndexer


class FaissIndex:
    def __init__(self, dim, partitions, similarity="l2"):
        self.dim = dim
        self.partitions = partitions
        self.num_subquantizers = 8

        # self.quantizer, self.index = self._create_index()
        assert similarity in ["cosine", "l2"]
        if similarity == "l2":
            self.index = faiss.IndexFlatL2(self.dim)
        else:
            self.index = faiss.IndexFlatIP(self.dim)
        self.offset = 0

    def _create_index(self):
        quantizer = faiss.IndexFlatL2(self.dim)
        index = faiss.IndexIVFPQ(
            quantizer, self.dim, self.partitions, self.num_subquantizers, 8
        )
        # index = faiss.IndexIVFFlat(quantizer, self.dim, self.partitions)

        return quantizer, index

    def train(self, train_data):
        s = time.time()
        self.index.train(train_data)
        print(time.time() - s)

    def add(self, data):
        self.index.add(data)

    def search(self, query, k=4, nprobe=10):
        self.index.nprobe = nprobe
        return self.index.search(query, k)

    def save(self, output_path, nprobe=10):
        self.index.nprobe = nprobe
        faiss.write_index(self.index, output_path)


if __name__ == "__main__":
    from retrieval.configs import BaseConfig
    import numpy as np
    import random

    random.seed(125)
    np.random.seed(125)

    config = BaseConfig(
        dim=32, batch_size=16, accum_steps=1, similarity="cosine"  # "l2" or "cosine"
    )

    INDEX_PATH = "../../data/fandom-qa/witcher_qa/passages.train.indices.pt"

    indexer = ColBERTIndexer(config, device="cpu")
    indexer.load(INDEX_PATH)

    faiss_indexer = FaissIndex(config.dim, partitions=100, similarity=config.similarity)
    if not faiss_indexer.index.is_trained:
        faiss_indexer.train(indexer.embeddings.numpy())
    faiss_indexer.add(indexer.embeddings.numpy())

    # print(indexer.embeddings[:10].numpy())

    # D, I = faiss_indexer.search(indexer.embeddings[:1].numpy(), k=3, nprobe=1)
    # print(D, I, sep="\n", end="\n\n\n")

    D, I = faiss_indexer.search(indexer.embeddings[:10].numpy(), k=3, nprobe=1)
    print(D, I, sep="\n")
