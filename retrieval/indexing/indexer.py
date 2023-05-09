import faiss
import time
from tqdm import tqdm
import torch
from collections import defaultdict

from retrieval.configs import BaseConfig
from retrieval.data import Passages
from retrieval.models import ColBERTTokenizer, ColBERTInference



class ColBERTIndexer():
    def __init__(self, config, device="cpu"):
        self.config = config
        self.device = device
        
        self.load_model()
        self.embeddings = None
        self.iid2pid = dict()
        self.pid2iid = defaultdict(list)
        self.next_iid = 0
        
    def load_model(self):
        # TODO: implement correctly
        self.tokenizer = ColBERTTokenizer(self.config)
        self.inference = ColBERTInference(self.config, self.tokenizer, device=self.device)
        
    def index(self, path_to_passages: str, bsize: int = 16):
        passages = Passages(path_to_passages)
        embeddings = []

        with torch.inference_mode():
            # iterated batch-wise over the passage dataset and store their embeddings
            for i in tqdm(range(0, len(passages), bsize)):
                batch = passages[i:i+bsize].values.tolist()
                pids = passages[i:i+bsize].index.tolist()
                psgs_embs = self.inference.doc_from_text(batch)
                
                # store the embeddings
                embeddings.extend(psgs_embs)

                # update the iid2pid and pid2iid mappings
                for pid, psg_embs in zip(pids, psgs_embs):
                    # adds new iids in the range [next_iid, next_iid + n_embeddings)
                    start_iid, end_iid = self.next_iid, self.next_iid + psg_embs.shape[0]
                    new_iids = range(start_iid, end_iid)
                    self.pid2iid[pid].extend(new_iids)
                    self.iid2pid.update({iid: pid for iid in new_iids})
                    self.next_iid = end_iid

            # concatenate the single embeddings/matrices into a large embedding matrix
            self.embeddings = torch.cat(embeddings, dim=0)

    def save(self, path):
        parameters = {
            "iid2pid": self.iid2pid,
            "pid2iid": self.pid2iid,
            "embeddings": self.embeddings.cpu(),
        }
        torch.save(parameters, path)
    
    def load(self, path):
        parameters = torch.load(path)
        self.iid2pid = parameters["iid2pid"]
        self.pid2iid = parameters["pid2iid"]
        self.embeddings = parameters["embeddings"].to(self.device)
    
    def get_pid_embeddings(self, pid):
        iids = self.pid2iid[pid]
        emb = [self.embeddings[iid] for iid in iids]
        return torch.stack(emb, dim=0)


if __name__ == "__main__":
    import random
    import numpy as np

    SEED = 125
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


    config = BaseConfig(
        dim = 32,
        batch_size = 16,
        accum_steps = 1,
    )
    PATH = "../../data/fandom-qa/witcher_qa/passages.train.tsv"
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

    INDEX_PATH = "../../data/fandom-qa/witcher_qa/passages.train.indices.pt"
    IDX = 3

    indexer = ColBERTIndexer(config, device="cuda:0")
    indexer.index(PATH, bsize=16)
    indexer.save(INDEX_PATH)
    # print(indexer.pid2iid)
    print(indexer.iid2pid[IDX], indexer.embeddings[IDX])
    print(indexer.get_pid_embeddings(indexer.iid2pid[IDX]))

    indexer = ColBERTIndexer(config, device="cpu")
    indexer.load(INDEX_PATH)
    print(indexer.iid2pid[IDX], indexer.embeddings[IDX])
    print(indexer.get_pid_embeddings(indexer.iid2pid[IDX]))