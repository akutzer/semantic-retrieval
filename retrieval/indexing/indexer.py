import faiss
import time
from tqdm import tqdm
import torch
from collections import defaultdict

from retrieval.configs import BaseConfig
from retrieval.data import Passages
from retrieval.models import ColBERTTokenizer, ColBERTInference


DEVICE = "cuda:0"



class ColBERTIndexer():
    def __init__(self, config):
        self.config = config
        
        self.load_model()
        self.embeddings = None
        self.iid2pid = dict()
        self.pid2iid = defaultdict(list)
        self.next_iid = 0

        
    def load_model(self):
        # TODO: implement correctly
        self.tokenizer = ColBERTTokenizer(self.config)
        self.inference = ColBERTInference(self.config, self.tokenizer, device=DEVICE)
        
    def index(self, path_to_passages: str):
        passages = Passages(path_to_passages)
        bsize = 16
        embeddings = []
        for i in tqdm(range(0, len(passages), bsize)):
            batch = passages[i:i+bsize].values.tolist()
            pids = passages[i:i+bsize].index.tolist()
            psgs_embs = self.inference.doc_from_text(batch)
            for pid, psg_embs in zip(pids, psgs_embs):
                n_embs = psg_embs.shape[0]
                new_iids = list(range(self.next_iid, n_embs))
                self.pid2iid[pid].extend(new_iids)
                for iid in new_iids:
                    self.iid2pid[iid] = pid     
                embeddings.append(psg_embs.cpu())
                self.next_iid += n_embs 
            
    
        self.embeddings = torch.cat(embeddings, dim=0)
        print(self.embeddings.shape)
        print(len(self.iid2pid), self.next_iid)

    def store(self, path):
        parameters = {
            "iid2pid": self.iid2pid,
            "pid2iid": self.pid2iid,
            "embeddings": self.embeddings,
        }
        torch.save(parameters, path)
    
    def load(self, path):
        parameters = torch.load(path)
        self.iid2pid = parameters["iid2pid"]
        self.pid2iid = parameters["pid2iid"]
        self.embeddings = parameters["embeddings"]
    
    def get_pid_embeddings(self, pid):
        iids = self.pid2iid[pid]
        emb = [self.embeddings[iid] for iid in iids]
        return torch.stack(emb, dim=0)

            
        


class FaissIndex():
    def __init__(self, dim, partitions):
        self.dim = dim
        self.partitions = partitions

        self.quantizer, self.index = self._create_index()
        self.offset = 0

    def _create_index(self):
        quantizer = faiss.IndexFlatL2(self.dim)
        index = faiss.IndexIVFPQ(quantizer, self.dim, self.partitions, 16, 8)

        return quantizer, index

    def train(self, train_data):
        s = time.time()
        self.index.train(train_data)
        print(time.time() - s)

    def add(self, data):
        self.index.add(data)

    def save(self, output_path, nprobe=10):
        self.index.nprobe = nprobe
        faiss.write_index(self.index, output_path)


if __name__ == "__main__":
    config = BaseConfig(
        dim = 32,
        batch_size = 16,
        accum_steps = 1,
    )
    PATH = "../../data/fandom-qa/witcher_qa/passages.train.tsv"
    indexer = ColBERTIndexer(config)

    indexer.index(PATH)
    index_path = "../../data/fandom-qa/witcher_qa/passages.train.indices.pt"
    IDX = 3
    print(indexer.pid2iid)
    print(indexer.iid2pid[IDX], indexer.embeddings[IDX])
    print(indexer.get_pid_embeddings(indexer.iid2pid[IDX]))
    indexer.store(index_path)

    indexer = ColBERTIndexer(config)
    indexer.load(index_path)
    print(indexer.iid2pid[IDX], indexer.embeddings[IDX])
    print(indexer.get_pid_embeddings(indexer.iid2pid[IDX]))
    #print(q)