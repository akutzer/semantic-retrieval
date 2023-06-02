#!/usr/bin/env python3
import math
from typing import List, Union, Optional

import torch
from tqdm import tqdm

from retrieval.configs import BaseConfig
from retrieval.models.colbert.colbert import ColBERT
from retrieval.models.colbert.tokenizer import ColBERTTokenizer
from retrieval.models.colbert.load import (
    load_colbert_and_tokenizer,
    get_colbert_and_tokenizer,
)


class ColBERTInference:
    def __init__(
        self,
        colbert: ColBERT,
        tokenizer: ColBERTTokenizer,
        device: Union[str, torch.device] = "cpu",
    ):
        self.colbert = colbert
        self.tokenizer = tokenizer
        self.colbert.register_tokenizer(tokenizer)

        self.to(device)
        self.colbert.eval()

    def query(
        self,
        input_ids: torch.IntTensor,
        attention_mask: torch.BoolTensor,
        to_cpu: bool = False,
    ) -> List[torch.Tensor]:
        """
        Calculates the ColBERT embedding for a tokenized query.
        """
        with torch.inference_mode():
            Q = self.colbert.query(input_ids, attention_mask)

        if to_cpu:
            Q = Q.cpu()
        # split the tensor of shape (B, L, dim) into a list of d tensors with shape (1, L, dim)
        # Q = [q.squeeze(0) for q in torch.split(Q, 1, dim=0)]
        return Q

    def doc(
        self,
        input_ids: torch.IntTensor,
        attention_mask: torch.BoolTensor,
        to_cpu: bool = False,
    ) -> List[torch.Tensor]:
        """
        Calculates the ColBERT embedding for a tokenized document/passage.
        """
        with torch.inference_mode():
            D, mask = self.colbert.doc(
                input_ids.to(self.device, non_blocking=True),
                attention_mask.to(self.device, non_blocking=True),
                return_mask=True,
            )

        if to_cpu:
            D, mask = D.cpu(), mask.cpu()

        # split the tensor of shape (B, L_pad, dim) into a list of d tensors with shape (L, dim)
        # while also removing padding & punctuation embeddings
        D = [d[m].squeeze(0) for d, m in zip(D, mask)]

        return D

    def query_from_text(
        self,
        query: Union[str, List[str]],
        bsize: Optional[int] = None,
        to_cpu: bool = False,
        show_progress: bool = False,
    ) -> torch.Tensor:
        """
        Calculates the ColBERT embedding for a query or list of queries represented as strings.
        """
        is_single_query = isinstance(query, str)
        if is_single_query:
            query = [query]

        if bsize is None:
            bsize = len(query)

        with torch.inference_mode():
            device = "cpu" if to_cpu else self.device
            Qs = torch.empty(
                len(query),
                self.tokenizer.query_maxlen,
                self.colbert.out_features,
                device=device,
            )

            # iterator of batches which contain of a (B, L_q, D) shaped index tensor
            # and a (B, L_q) shaped attention mask tensor
            batches = self.tokenizer.tensorize(query, mode="query", bsize=bsize)

            if show_progress:
                total = math.ceil(len(query) / bsize) if bsize is not None else 1
                batches = tqdm(batches, total=total)

            for i, Q in enumerate(batches):
                Q = self.query(*Q, to_cpu=to_cpu)
                Qs[i : i + bsize] = Q
                # Qs.extend(Q)

        return Qs[0] if is_single_query else Qs

    def doc_from_text(
        self,
        doc: Union[str, List[str]],
        bsize: Optional[int] = None,
        to_cpu: bool = False,
        show_progress: bool = False,
    ) -> List[torch.Tensor]:
        """
        Calculates the ColBERT embedding for a document/passages or list of document/passages represented as strings.
        """
        is_single_doc = isinstance(doc, str)
        Ds = []

        with torch.inference_mode():
            batches = self.tokenizer.tensorize(doc, mode="doc", bsize=bsize)
            if bsize is None:
                batches = [batches]
            if show_progress:
                total = math.ceil(len(doc) / bsize) if bsize is not None else 1
                batches = tqdm(batches, total=total)

            for D in batches:
                D = self.doc(*D, to_cpu=to_cpu)
                Ds.extend(D)

        return Ds[0] if is_single_doc else Ds

    @classmethod
    def from_pretrained(
        cls, directory: str, device: Union[str, torch.device] = "cpu"
    ) -> "ColBERTInference":
        colbert, tokenizer = load_colbert_and_tokenizer(directory, device)
        model = cls(colbert, tokenizer)
        model.to(device)
        return model

    def to(self, device: Union[str, torch.device]) -> None:
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device
        self.colbert.to(device=device)


if __name__ == "__main__":
    from tqdm import tqdm

    queries = ["Hi, how are you today?", "Wow, Where do you live?"]
    passages = ["I'm ... let me think ... great!", "Nowhere, brudi.", "ooohhh noooo..."]

    MODEL_PATH = "bert-base-uncased"  # "bert-base-uncased" or "roberta-base"
    DEVICE = "cuda:0"  # "cpu" or "cuda:0"
    BSIZE = 3

    config = BaseConfig(
        tok_name_or_path=MODEL_PATH,
        backbone_name_or_path=MODEL_PATH,
    )

    model, tokenizer = get_colbert_and_tokenizer(config)
    colbert = ColBERTInference(model, tokenizer, device=DEVICE)

    # colbert = ColBERTInference.from_pretrained("../../../data/colbertv2.0/", device=DEVICE)
    # tokenizer = colbert.tokenizer
    # print(colbert)

    queries = queries[0]
    Q = tokenizer.tensorize(queries, mode="query")
    qrys1 = colbert.query(*Q)
    qrys2 = colbert.query_from_text(queries, bsize=BSIZE)
    if isinstance(queries, str):
        qrys2 = qrys2[None]
    for qry1, qry2 in zip(qrys1, qrys2):
        print(torch.allclose(qry1, qry2), torch.max(qry1 - qry2).item())

    P = tokenizer.tensorize(passages, mode="doc")
    psgs1 = colbert.doc(*P)
    psgs2 = colbert.doc_from_text(passages, bsize=BSIZE)
    for psg1, psg2 in zip(psgs1, psgs2):
        print(torch.allclose(psg1, psg2), torch.max(psg1 - psg2).item())
