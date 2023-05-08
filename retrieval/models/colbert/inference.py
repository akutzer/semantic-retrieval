import torch
from typing import List
import math

from retrieval.configs import BaseConfig
from retrieval.models.colbert.colbert import ColBERT
from retrieval.models.colbert.tokenizer import ColBERTTokenizer


class ColBERTInference(ColBERT):
    def __init__(self, config: BaseConfig, tokenizer: ColBERTTokenizer, device="cpu"):
        super().__init__(config, tokenizer, device)
        self.eval()
    
    def query(self, input_ids, attention_mask, to_cpu: bool = False):
        with torch.inference_mode():
            Q = super().query(input_ids, attention_mask)

        # split the tensor of shape (B, L, Q_dim) into a list of d tensors with shape (L, Q_dim)
        Q = torch.chunk(Q.cpu() if to_cpu else Q, Q.shape[0], dim=0)
        return Q
    
    def doc(self, input_ids, attention_mask, to_cpu: bool = False):
        with torch.inference_mode():
            D, mask = super().doc(input_ids, attention_mask, return_mask=True)

            # remove the vectors representing the embedding of punctuation symbols
            D = [d[m].cpu() if to_cpu else d[m] for d, m in zip(D, mask)]

        return D
    
    def query_from_text(self, queries: List[str], to_cpu=False):
        # TODO: update batch_size, accum_steps and bucket_size in settings.py
        bsize = math.ceil(self.config.batch_size / self.config.accum_steps)
        Qs = []

        with torch.inference_mode():
            for i in range(0, len(queries), bsize):
                qry_batch = queries[i:i+bsize]
                Q = self.tokenizer.tensorize(qry_batch, mode="query")
                #print(Q)
                Q = self.query(*Q, to_cpu=to_cpu)
                Qs.extend(Q)

        return Qs
    
    def doc_from_text(self, doc: List[str], to_cpu=False):
        # TODO: update batch_size, accum_steps and bucket_size in settings.py
        bsize = math.ceil(self.config.batch_size / self.config.accum_steps)
        Ds = []

        with torch.inference_mode():
            for i in range(0, len(doc), bsize):
                doc_batch = doc[i:i+bsize]
                D = self.tokenizer.tensorize(doc_batch, mode="doc")
                D = self.doc(*D, to_cpu=to_cpu)
                Ds.extend(D)

        return Ds




if __name__ == "__main__":
    from tqdm import tqdm

    queries = ["Hi, how are you today?", "Wow, Where do you live?"]
    passages = ["I'm ... let me think ... great!", "Nowhere, brudi."]

    MODEL_PATH = "bert-base-uncased" # "../../../data/colbertv2.0/" or "bert-base-uncased" or "roberta-base"
    DEVICE = "cuda:0"

    config = BaseConfig(
        tok_name_or_path=MODEL_PATH,
        backbone_name_or_path=MODEL_PATH,
        similarity="l2",
        dim = 24,
        accum_steps=1
    )

    tokenizer = ColBERTTokenizer(config)
    colbert = ColBERTInference(config, tokenizer, device=DEVICE)

    Q = tokenizer.tensorize(queries, mode="query", return_tensors="pt")
    P = tokenizer.tensorize(passages, mode="doc", return_tensors="pt")

    qrys1 = colbert.query(*Q)
    qrys2 = colbert.query_from_text(queries)
    for qry1, qry2 in zip(qrys1, qrys2):
        # print(torch.max(qry1 - qry2))
        print(torch.allclose(qry1, qry2))

    psgs1 = colbert.doc(*P)
    psgs2 = colbert.doc_from_text(passages)
    for psg1, psg2 in zip(psgs1, psgs2):
        # print(torch.max(psg1 - psg2))
        print(torch.allclose(psg1, psg2))
