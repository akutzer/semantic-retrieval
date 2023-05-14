#!/usr/bin/env python3
import os
import string

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer

from retrieval.configs import BaseConfig
from retrieval.models.colbert.tokenizer import ColBERTTokenizer

# suppresses the warnings when loading a model with unused parameters 
import logging
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)



class ColBERT(nn.Module):
    def __init__(self, config: BaseConfig, tokenizer: ColBERTTokenizer, device="cpu"):
        super().__init__()
        self.config = config
        self.backbone_config = self._load_model_config(config)
        self.device = device
          
        self.tokenizer = tokenizer
        self.backbone = AutoModel.from_pretrained(config.backbone_name_or_path, config=self.backbone_config)
        self.backbone.resize_token_embeddings(len(self.tokenizer))
  
        self.hid_dim = self.backbone.config.hidden_size
        self.linear = nn.Linear(self.hid_dim, self.config.dim, bias=False)
        if self._load_linear_weights():
            print("Successfully loaded weights for last linear layer!")

        if self.config.skip_punctuation:
            self.skiplist = {w: True
                             for symbol in string.punctuation
                             for w in [symbol, self.tokenizer.encode(symbol, mode="query", add_special_tokens=False)[0]]}
        self.pad_token_id = self.tokenizer.pad_token_id

        self.to(device=device)
        self.train()

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
        similarities = self.similarity(q_vec_duplicated, d_vec, d_mask, intra_batch=self.config.intra_batch_similarity)

        return similarities

    def query(self, input_ids, attention_mask):
        input_ids, attention_mask = input_ids.to(self.device), attention_mask.to(self.device)

        # run query through the backbone, e.g. BERT, but drop the pooler output
        Q = self.backbone(input_ids, attention_mask=attention_mask)[0]
        # reduce the query vectors dimensionality
        Q = self.linear(Q)

        # normalize each vector
        if self.config.normalize:
            Q = F.normalize(Q, p=2, dim=-1)

        return Q

    def doc(self, input_ids, attention_mask, return_mask=True):
        input_ids, attention_mask = input_ids.to(self.device), attention_mask.to(self.device)

        # run document through the backbone, e.g. BERT, but drop the pooler output
        D = self.backbone(input_ids, attention_mask=attention_mask)[0]
        # reduce the document vectors dimensionality
        D = self.linear(D)

        # normalize each vector
        if self.config.normalize:
            D = F.normalize(D, p=2, dim=-1)

        # mask the vectors representing the embedding of punctuation symbols
        mask = torch.tensor(self.mask(input_ids, skiplist=self.skiplist), device=self.device, dtype=torch.bool)
        D = D * mask.unsqueeze(-1)

        return D, mask if return_mask else D
   
    def similarity(self, Q, D_padded, D_mask=None, intra_batch=False):
        # Q shape:        (B*psgs_per_qry, L_q, out_features)
        # D_padded shape: (B*psgs_per_qry, L_d, out_features)
        # D_mask shape:   (B*psgs_per_qry, L_d)
        if not intra_batch:
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
            if D_mask:
                sim.mT[~D_mask] = float("-inf")

            # calculate the sum of maximum similarity (sms)
            # sim shape: (B*psgs_per_qry, L_q, L_d)
            sms = sim.max(dim=-1).values.sum(dim=-1)
        
        else:
            assert self.config.passages_per_query == 1

            B, L_q, out_features = Q.shape
            L_d = D_padded.shape[1]

            if self.config.similarity.lower() == "l2":
                # calculate squared l2-norm
                # we need to negate, since we later want to maximize the similarity,
                # and the closer they are, the smaller is the distance between two vectors
                #print(Q[:, None, :, None].shape, D_padded[None, :, None].shape)
                # TODO: try to improve this call, since it's extremly memory hungry
                sim = -1.0 * (Q[:, None, :, None] - D_padded[None, :, None]).pow(2).sum(dim=-1)
               
            elif self.config.similarity.lower() == "cosine":                
                Q = Q.reshape(B*L_q, out_features)
                D_padded = D_padded.reshape(B*L_d, out_features)

                sim = Q @ D_padded.T
                sim = sim.reshape(B, L_q, B, L_d).permute(0, 2, 1, 3)
                # sim shape: (B*psgs_per_qry, B*psgs_per_qry, L_q, L_d)
                
            else:
                raise ValueError(f"Invalid similarity function {self.config.similarity} given. Must be either 'l2' or 'cosine'")
            
            # ignore the similarities for padding and punctuation tokens
            if D_mask:
                D_mask = D_mask[None].repeat_interleave(B, dim=0)
                sim.mT[~D_mask] = float("-inf")

            # calculate the sum of maximum similarity (sms)
            # sim shape: (B*psgs_per_qry, B*psgs_per_qry, L_q, L_d)
            sms = sim.max(dim=-1).values.sum(dim=-1)
        
        return sms

    def mask(self, input_ids, skiplist=[]):
        mask = [[(tok not in skiplist) and (tok != self.pad_token_id) for tok in sample] for sample in input_ids.cpu().tolist()]
        return mask
    
    def _load_linear_weights(self):
        if not os.path.exists(self.config.backbone_name_or_path):
            return False

        for file in os.listdir(self.config.backbone_name_or_path):
            path_to_weights = os.path.join(self.config.backbone_name_or_path, file)
            if not os.path.isfile(path_to_weights):
                continue
            
            if "pytorch_model" in file:
                try:
                    with open(path_to_weights, mode="br") as f: 
                        parameters = torch.load(f)

                    if "linear.weight" in parameters.keys():
                        weights = parameters["linear.weight"]
                        # replace the weights if the number of input features is the
                        if weights.shape[-1] == self.linear.weight.shape[-1]:
                            self.linear.weight.data = weights[:self.config.dim]
                            return True

                except Exception as e:
                    print(f"Couldn't load linear weights: {e}")

        return False
    
    def _load_model_config(self, config):
        backbone_config = AutoConfig.from_pretrained(config.backbone_name_or_path)

        backbone_config.hidden_size = config.hidden_size
        backbone_config.num_hidden_layers = config.num_hidden_layers
        backbone_config.num_attention_heads = config.num_attention_heads
        backbone_config.intermediate_size = config.intermediate_size
        backbone_config.hidden_act = config.hidden_act
        backbone_config.hidden_dropout_prob = config.dropout
        backbone_config.attention_probs_dropout_prob = config.dropout

        return backbone_config





if __name__ == "__main__":
    from tqdm import tqdm
    from transformers import AutoTokenizer
    #from retrieval.configs import BaseConfig
    

    queries = ["How are you today?", "Where do you live?"]
    passages = ["I'm great!", "Nowhere brudi."]

    MODEL_PATH = "bert-base-uncased" # "../../../data/colbertv2.0/" or "bert-base-uncased" or "roberta-base"
    DEVICE = "cuda:0"
    EPOCHS = 25

    config = BaseConfig(
        tok_name_or_path=MODEL_PATH,
        backbone_name_or_path=MODEL_PATH,
        similarity="l2",
        intra_batch_similarity=True,
        epochs = EPOCHS,
        dim = 24,
        hidden_size = 768,
        num_hidden_layers = 12,
        num_attention_heads = 12,
        intermediate_size = 3072,
        hidden_act = "gelu",
        dropout = 0.1,
    )

    tokenizer = ColBERTTokenizer(config)
    colbert = ColBERT(config, tokenizer, device=DEVICE)

    optimizer = torch.optim.AdamW(colbert.parameters(), lr=4e-5)
    criterion = nn.CrossEntropyLoss()

    Q = tokenizer.tensorize(queries, mode="query", return_tensors="pt")
    P = tokenizer.tensorize(passages, mode="doc", return_tensors="pt")
    out = colbert(Q, P)


    for epoch in range(1, EPOCHS+1):
        optimizer.zero_grad()
        out = colbert(Q, P)
        B = out.shape[0]
        loss = criterion(out, torch.arange(0, B, device=DEVICE, dtype=torch.long))
        loss.backward()
        optimizer.step()
        print(loss.item())
    
    colbert.eval()
    with torch.no_grad():
        out = colbert(Q, P)
        print(out, F.softmax(out, dim=-1))
