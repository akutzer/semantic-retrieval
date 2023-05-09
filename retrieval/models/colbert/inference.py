import torch
from typing import List, Union
import math

from retrieval.configs import BaseConfig
from retrieval.models.colbert.colbert import ColBERT
from retrieval.models.colbert.tokenizer import ColBERTTokenizer





class ColBERTInference(ColBERT):
    def __init__(self, config: BaseConfig, tokenizer: ColBERTTokenizer, device="cpu"):
        super().__init__(config, tokenizer, device)
        self.eval()
    
    def query(self, input_ids: torch.IntTensor, attention_mask: torch.BoolTensor, to_cpu: bool = False) -> List[torch.Tensor]:
        """
        Calculates the ColBERT embedding for a tokenized query.
        """
        with torch.inference_mode():
            Q = super().query(input_ids, attention_mask)
            
        if to_cpu:
            Q = Q.cpu()
        # split the tensor of shape (B, L, dim) into a list of d tensors with shape (1, L, dim)
        Q = [q.squeeze(0) for q in torch.split(Q, 1, dim=0)]
        return Q
    
    def doc(self, input_ids: torch.IntTensor, attention_mask: torch.BoolTensor, to_cpu: bool = False) -> List[torch.Tensor]:
        """
        Calculates the ColBERT embedding for a tokenized document/passage.
        """
        with torch.no_grad():
            D, mask = super().doc(input_ids, attention_mask, return_mask=True)
        
        if to_cpu:
            D, mask = D.cpu(), mask.cpu()

        # split the tensor of shape (B, L_pad, dim) into a list of d tensors with shape (L, dim)
        # while also removing padding & punctuation embeddings
        D = [d[m].squeeze(0) for d, m in zip(D, mask)]
        
        return D
    
    
    def query_from_text(self, query: Union[str, List[str]], bsize: Union[None, int] = None, to_cpu: bool = False) -> List[torch.Tensor]:
        """
        Calculates the ColBERT embedding for a query or list of queries represented as strings.
        """
        is_single_query = isinstance(query, str)
        if is_single_query:
            query = [query]
        Qs = []

        with torch.inference_mode():
            batches = self.tokenizer.tensorize(query, mode="query", bsize=bsize)

            for Q in batches:
                Q = self.query(*Q, to_cpu=to_cpu)
                Qs.extend(Q)

        return Qs[0] if is_single_query else Qs
    
    def doc_from_text(self, doc: Union[str, List[str]], bsize: Union[None, int] = None, to_cpu: bool = False) -> List[torch.Tensor]:
        """
        Calculates the ColBERT embedding for a document/passages or list of document/passages represented as strings.
        """
        is_single_doc = isinstance(doc, str)
        if is_single_doc:
            doc = [doc]
        Ds = []

        with torch.inference_mode():
            batches = self.tokenizer.tensorize(doc, mode="doc", bsize=bsize)

            for D in batches:
                D = self.doc(*D, to_cpu=to_cpu)
                Ds.extend(D)

        return Ds[0] if is_single_doc else Ds




if __name__ == "__main__":
    from tqdm import tqdm

    queries = ["Hi, how are you today?", "Wow, Where do you live?"]
    passages = ["I'm ... let me think ... great!", "Nowhere, brudi.", "ooohhh noooo..."]

    MODEL_PATH = "bert-base-uncased" # "../../../data/colbertv2.0/" or "bert-base-uncased" or "roberta-base"
    DEVICE = "cuda:0" # "cpu" or "cuda:0"
    BSIZE = 3

    config = BaseConfig(
        tok_name_or_path=MODEL_PATH,
        backbone_name_or_path=MODEL_PATH,
    )

    tokenizer = ColBERTTokenizer(config)
    colbert = ColBERTInference(config, tokenizer, device=DEVICE)

    
    Q = tokenizer.tensorize(queries, mode="query")
    qrys1 = colbert.query(*Q)
    qrys2 = colbert.query_from_text(queries, bsize=BSIZE)
    for qry1, qry2 in zip(qrys1, qrys2):
        print(torch.allclose(qry1, qry2), torch.max(qry1 - qry2).item())

    P = tokenizer.tensorize(passages, mode="doc")
    psgs1 = colbert.doc(*P)
    psgs2 = colbert.doc_from_text(passages, bsize=BSIZE)
    for psg1, psg2 in zip(psgs1, psgs2):
        print(torch.allclose(psg1, psg2), torch.max(psg1 - psg2).item())
