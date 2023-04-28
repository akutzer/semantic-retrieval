import string

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer



class ColBERT(nn.Module):
    def __init__(self, config=None, device="cpu"):
        super().__init__()
        self.config = config
        self.backbone_config = AutoConfig.from_pretrained(config.backbone_name_or_path)
        self.device = device
        #self.to(device) 
          
        self.raw_tokenizer =AutoTokenizer.from_pretrained(config.tok_name_or_path)
        self.backbone = AutoModel.from_pretrained(config.backbone_name_or_path, config=self.backbone_config)
  
        self.hid_dim = self.backbone.config.hidden_size
        self.linear = nn.Linear(self.hid_dim, config.dim, bias=False)

        if self.config.skip_punctuation:
            self.skiplist = {w: True
                             for symbol in string.punctuation
                             for w in [symbol, self.raw_tokenizer.encode(symbol, add_special_tokens=False)[0]]}
        self.pad_token_id = self.raw_tokenizer.pad_token_id

    def forward(self, Q, D):
        # Q shape: (B, L_q)
        # D shape: (B, L_d)

        q_vec = self.query(*Q)
        # q_vec shape: (B, L_q, out_features)
        d_vec, d_mask = self.doc(*D, return_mask=True)
        # d_vec shape:  (B*psgs_per_qry, L_d, out_features)
        # d_mask shape: (B*psgs_per_qry, L_d)

        # Repeat each query encoding for every corresponding document
        q_vec_duplicated = q_vec.repeat_interleave(self.config.passages_per_query, dim=0).contiguous()
        similarities = self.similarity(q_vec_duplicated, d_vec, d_mask)

        return similarities

    def query(self, input_ids, attention_mask):
        input_ids, attention_mask = input_ids.to(self.device), attention_mask.to(self.device)

        # run query through the backbone, e.g. BERT, but drop the pooler output
        Q = self.backbone(input_ids, attention_mask=attention_mask)[0]
        # reduce the query vectors dimensionality
        Q = self.linear(Q)

        # normalize each vector
        Q = F.normalize(Q, p=2, dim=-1)

        return Q

    def doc(self, input_ids, attention_mask, return_mask=True):
        input_ids, attention_mask = input_ids.to(self.device), attention_mask.to(self.device)

        # run document through the backbone, e.g. BERT, but drop the pooler output
        D = self.backbone(input_ids, attention_mask=attention_mask)[0]
        # reduce the document vectors dimensionality
        D = self.linear(D)

        # normalize each vector
        D = F.normalize(D, p=2, dim=-1)

        # mask the vectors representing the embedding of punctuation symbols
        mask = torch.tensor(self.mask(input_ids, skiplist=self.skiplist), device=self.device, dtype=torch.bool)
        D = D * mask.unsqueeze(-1)

        return D, mask if return_mask else D
   
    def similarity(self, Q, D_padded, D_mask):
        # Q shape:        (B*psgs_per_qry, L_q, out_features)
        # D_padded shape: (B*psgs_per_qry, L_d, out_features)
        # D_mask shape:   (B*psgs_per_qry, L_d)

        if self.config.similarity.lower() == "l2":
            # calculate squared l2
            # we need to negate, since we later want to maximize the similarity,
            # and the closer they are, the smaller is the distance between two vectors
            sim = -1.0 * (Q.unsqueeze(2) - D_padded.unsqueeze(1)).pow(2).sum(dim=-1)
        
        elif self.config.similarity.lower() == "cosine":
            sim = (Q @ D_padded.mT)
            
        else:
            raise ValueError(f"Invalid similarity function {self.config.similarity} given. Must be either 'l2' or 'cosine'")
        
        # ignore the similarities for padding and punctuation tokens
        sim.mT[~D_mask] = float("-inf")

        # calculate the sum of maximum similarity (sms)
        # sim shape: (B*psgs_per_qry, L_q, L_d)
        sms = sim.max(dim=-1).values.sum(dim=-1)
        
        return sms

    def mask(self, input_ids, skiplist=[]):
        mask = [[(tok not in skiplist) and (tok != self.pad_token_id) for tok in sample] for sample in input_ids.cpu().tolist()]
        return mask



if __name__ == "__main__":
    from retrieval.configs import BaseConfig
    from retrieval.data import DataIterator
    from tqdm import tqdm

    model_path = "../../data/colbertv2.0/"
    model_path = "bert-base-uncased"
    config = BaseConfig(tok_name_or_path=model_path, backbone_name_or_path=model_path)


    triples_path = "/home/aaron/Documents/Studium/Informatik/6_Semester/KP BigData/semantic-retrieval/data/fandom-qa/harry_potter_qa/triples.train.tsv"
    queries_path = "/home/aaron/Documents/Studium/Informatik/6_Semester/KP BigData/semantic-retrieval/data/fandom-qa/harry_potter_qa/queries.train.tsv"
    passages_path = "/home/aaron/Documents/Studium/Informatik/6_Semester/KP BigData/semantic-retrieval/data/fandom-qa/harry_potter_qa/passages.train.tsv"


    data_iter = DataIterator(config, triples_path, queries_path, passages_path)
    device = "cuda:0"
    colbert = ColBERT(config, device=device)
    colbert.to(device=device)

    optimizer = torch.optim.AdamW(colbert.parameters(), lr=3e-4, maximize=True)


    for batch in tqdm(data_iter):
        optimizer.zero_grad()
        loss = 0
        for sub_batch in batch:
            q_tokens, q_masks, p_tokens, p_masks = sub_batch
            Q, P = (q_tokens, q_masks), (p_tokens, p_masks)

            
            out = colbert(Q, P)
            loss += out.mean().item()
            out.mean().backward()
            #loss += out.mean()
        #loss = loss / config.accum_steps
        
        optimizer.step()

        print(loss)


    print(colbert)

