#!/usr/bin/env python3
import os
import string
from typing import Union, Optional, Tuple
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel

from retrieval.configs import BaseConfig, save_config, load_config
from retrieval.models.colbert.tokenizer import ColBERTTokenizer



# suppresses the warnings when loading a model with unused parameters
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

# intialize logging messages
logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logging.basicConfig(level=logging.WARNING, format="[%(asctime)s][%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")



class ColBERT(nn.Module):
    def __init__(self, config: BaseConfig, device: Union[str, torch.device] = "cpu"):
        super().__init__()
        self.config = config
        self.skiplist = None
        self.pad_token_id = None
        
        # load backbone and load/initialize output layer
        self.backbone_config = ColBERT._load_model_config(self.config)
        self.backbone = AutoModel.from_pretrained(
            self.config.backbone_name_or_path, config=self.backbone_config
        )
        self.hid_dim = self.backbone.config.hidden_size
        self.out_features = self.config.dim
        self.linear = nn.Linear(self.hid_dim, self.out_features, bias=False)


        # old way of loading colbertv2
        if "colbertv2.0/" in self.config.backbone_name_or_path:
            logging.info("Detected usage of ColBERTv2 as backbone name! This is deprecated, but will continue to work!")
            self.config.checkpoint = self.config.backbone_name_or_path
            if self._load_linear_weights():
                logging.info("Successfully loaded weights for last ColBERTv2 layer!")     

        self.to(device=device)
        self.train()

    def forward(
        self,
        Q: Tuple[torch.IntTensor, torch.BoolTensor],
        D: Tuple[torch.IntTensor, torch.BoolTensor],
    ) -> torch.FloatTensor:
        """
        Calculates the embedding for the query and document tensor and returns
        their sum of maximal similarity.

        Input:
            Q: (IntTensor of shape (B*N_q, L_q), BoolTensor of shape (B*N_q, L_q))
            D: (IntTensor of shape (B*N_p, L_d), BoolTensor of shape (B*N_p, L_d))
            The first tensor contains the token ids while the second is the mask tensor
            needed for the transformer model.
        Return:
            FloatTensor of shape (B, max(N_q, N_d)) or (B, B)
            The first one is the case for QQP and QPP-style datasets and the second
            one for QP-style datasets with `intra_batch=True`

        B   - batch_size
        N   - number of queries or documents per batch (QQP: N_q = 2 and N_p = 1, QPP: N_q = 1 and N_p >= 2)
        L_q - number of tokens per query
        L_d - number of tokens per document
        """
        B_q, L_q = Q[0].shape
        B_d, L_d = D[0].shape
        B = min(B_q, B_d)

        # calculate the embeddings
        q_vec = self.query(*Q)  # shape: (B*N_q, L_q, F); F = out_features
        d_vec, d_mask = self.doc(*D, return_mask=True)
        # d_vec shape:  (B*N_p, L_d, F)
        # d_mask shape: (B*N_p, L_d)

        # reshapes the 3d Tensors into 4d Tensors of the shape (B, N_q, L_q, F)
        # this is done to utilize broadcasting later on in case of a QQP or QPP
        # style dataset, if its a QP dataset (so N_q =  N_p = 1), this operation
        # is more or less useless but does no harm
        q_vec = q_vec.reshape(B, -1, L_q, self.out_features)
        d_vec = d_vec.reshape(B, -1, L_d, self.out_features)
        d_mask = d_mask.reshape(B, -1, L_d)
        # in case of an QQP-style dataset, we need to manually expand dim=1, since
        # broadcasting does not work for slicing with masks
        if B_q > B_d:
            queries_per_passage = B_q // B_d
            d_mask = d_mask.expand(-1, queries_per_passage, -1)

        similarities = self.similarity(
            q_vec, d_vec, d_mask, intra_batch=self.config.intra_batch_similarity
        )

        return similarities

    def query(
        self, input_ids: torch.IntTensor, attention_mask: torch.BoolTensor
    ) -> torch.FloatTensor:
        """
        Calculates the embeddings for a batch of tokenized queries.
        Note: The returned embedding vectors are normalized.

        Input:
            input_ids     : IntTensor of shape (B, L_q)
            attention_mask: BoolTensor of shape (B, L_q)
        Returns:
            FloatTensor of shape: (B, L_q, F)

        B   - batch_size
        L_q - number of tokens per query
        F   - dimension of an embedding vector (number of features)
        """
        input_ids = input_ids.to(self.device, non_blocking=True)
        attention_mask = attention_mask.to(self.device, non_blocking=True)

        # run query through the backbone, e.g. BERT, but drop the pooler output
        Q = self.backbone(input_ids, attention_mask=attention_mask)[0]
        # reduce the query vectors dimensionality
        Q = self.linear(Q)

        # normalize each embedding vector
        if self.config.normalize:
            Q = F.normalize(Q, p=2, dim=-1)

        return Q

    def doc(
        self,
        input_ids: torch.IntTensor,
        attention_mask: torch.BoolTensor,
        return_mask: bool = True,
    ) -> Union[torch.FloatTensor, Tuple[torch.FloatTensor, torch.BoolTensor]]:
        """
        Calculates the embeddings for a batch of tokenized documents.
        Note: The returned embedding vectors are normalized.

        Input:
            input_ids     : IntTensor of shape (B, L_d)
            attention_mask: BoolTensor of shape (B, L_d)
        Returns:
            FloatTensor of shape: (B, L_d, F)
            BoolTensor of shape:  (B, L_d) if `return_mask=True`

        B   - batch_size
        L_d - number of tokens per document
        F   - dimension of an embedding vector (number of features)
        """
        input_ids = input_ids.to(self.device, non_blocking=True)
        attention_mask = attention_mask.to(self.device, non_blocking=True)

        # run document through the backbone, e.g. BERT, but drop the pooler output
        D = self.backbone(input_ids, attention_mask=attention_mask)[0]
        # reduce the document vectors dimensionality
        D = self.linear(D)

        # normalize each vector
        if self.config.normalize:
            D = F.normalize(D, p=2, dim=-1)

        # mask the vectors representing the embedding of punctuation symbols
        mask = self.mask(input_ids, skiplist=self.skiplist)
        D = D * mask.unsqueeze(-1)

        return D, mask if return_mask else D

    def similarity(
        self,
        Q: torch.FloatTensor,
        D: torch.FloatTensor,
        D_mask: Optional[torch.BoolTensor] = None,
        intra_batch: bool = False,
    ) -> torch.FloatTensor:
        """
        Calculates the sum of max similarites between the query embeddings and document embeddings.
        Depending on the config, either the negated squared L2 norm or the cosine similarity is used as a meassure of similarity.
        If for each query their is only one document given, the similarity can be calculated between the batches, if `intra_batch = True`.

        Input:
            Q      : torch.FloatTensor of shape (B, N, L_q, F)
            D      : torch.FloatTensor of shape (B, N, L_d, F)
            D_mask : torch.BoolTensor  of shape (B, N, L_d)
        Return:
            FloatTensor of shape (B, max(N_q, N_d)) or (B, B)
            The first one is the case for QQP and QPP-style datasets and the second
            one for QP-style datasets with `intra_batch=True`

        B   - batch_size
        N   - number of queries or documents per batch (QQP: N_q = 2 and N_p = 1, QPP: N_q = 1 and N_p >= 2)
        L_q - number of embeddings per query
        L_d - number of embeddings per document
        F   - dimension of an embedding vector (number of features)
        """
        Q = Q.to(dtype=D.dtype)
        if not intra_batch:
            if self.config.similarity.lower() == "l2":
                # calculate squared l2 norm
                # we need to negate, since we later want to maximize the similarity,
                # and the closer they are, the smaller is the distance between two vectors
                sim = -1.0 * torch.cdist(Q, D, p=2.0)
            elif self.config.similarity.lower() == "cosine":
                # since the vectors are already normed, calculating the dot product
                # gives the cosine similarity
                sim = Q @ D.mT  # shape: (B, N, L_q, L_d)
            else:
                raise ValueError(
                    f"Invalid similarity function `{self.config.similarity}` given. "
                    "Must be either `L2` or `cosine`"
                )

            # ignore the similarities for padding and punctuation tokens
            if D_mask is not None:
                sim.mT[~D_mask] = float("-inf")
            # calculate the sum of maximum similarity (SMS)
            sms = sim.max(dim=-1).values.mean(dim=-1)  # shape: (B, N)

        else:
            assert self.config.passages_per_query == 1
            assert Q.shape[1] == 1 and D.shape[1] == 1, f"{Q.shape}, {D.shape}"

            # remove dim=1, since it should be 1 in both cases
            Q, D = Q.squeeze(1), D.squeeze(1)
            if D_mask is not None:
                D_mask = D_mask.squeeze(1)

            B, L_q, F = Q.shape
            L_d = D.shape[1]

            if self.config.similarity.lower() == "l2":
                # calculate squared l2-norm
                # we need to negate, since we later want to maximize the similarity,
                # and the closer they are, the smaller is the distance between two vectors
                # Q shape: (B, 1, L_q,  1 , F)
                # D shape: (1, B,  1 , L_d, F)
                # TODO: try to improve this call, since it's extremly memory hungry
                sim = -1.0 * (Q[:, None, :, None] - D[None, :, None]).pow(2).sum(dim=-1)

            elif self.config.similarity.lower() == "cosine":
                Q = Q.reshape(B * L_q, F)
                D = D.reshape(B * L_d, F)

                sim = Q @ D.T
                sim = sim.reshape(B, L_q, B, L_d).permute(0, 2, 1, 3)  # shape: (B, B, L_q, L_d)

            else:
                raise ValueError(
                    f"Invalid similarity function {self.config.similarity} given. "
                    "Must be either 'l2' or 'cosine'"
                )

            # ignore the similarities for padding and punctuation tokens
            if D_mask is not None:
                D_mask = D_mask[None].repeat_interleave(B, dim=0)
                sim.mT[~D_mask] = float("-inf")
            # calculate the sum of maximum similarity (sms)
            sms = sim.max(dim=-1).values.mean(dim=-1)  # shape: (B, B)

        return sms

    def mask(
        self, input_ids: torch.IntTensor, skiplist: Optional[torch.IntTensor] = None
    ) -> torch.BoolTensor:
        """
        Returns a boolean mask of the same shape as the input, with
            - True,  if the token-id is neither a padding-token nor in the skiplist
            - False, otherwise.

        Input:
            input_ids: IntTensor of arbitrary shape
        Return:
            BoolTensor of the shape as the input_ids tensor
        """
        with torch.no_grad():
            is_pad_token = input_ids == self.pad_token_id

            if skiplist is not None:
                # applies logical or along the last dimension
                is_punctuation = torch.any((input_ids[..., None] == skiplist), dim=-1)
                mask = ~is_pad_token & ~is_punctuation
            else:
                mask = ~is_pad_token
        return mask
    
    def to(self, device: Optional[Union[str, torch.device]] = None, dtype: Optional[torch.dtype] = None) -> None:
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device
        if self.skiplist is not None:
            self.skiplist = self.skiplist.to(device=device)
        super().to(device=device, dtype=dtype)

    def register_tokenizer(self, tokenizer: ColBERTTokenizer) -> None:
        """
        Resizes the embedding matrix of the underlying backbone to fit the tokenizer.
        Initializes the skiplist and stores the pad-token-id needed for the masking
        of the document embeddings.
        """
        # resize the embedding matrix if necessary
        self.backbone.resize_token_embeddings(len(tokenizer))

        if self.config.skip_punctuation:
            self.skiplist = []
            # add ids of standalone punctuation symbols
            skiplist = [
                tokenizer.encode(symbol, mode="doc", add_special_tokens=False)[0]
                for symbol in string.punctuation
            ]
            # add ids of punctuation symbols trailing a word
            # in most cases these are the actual tokens that occur in a tokenized sentence
            skiplist += [
                tokenizer.encode("a" + symbol, mode="doc", add_special_tokens=False)[-1]
                for symbol in string.punctuation
            ]
            self.skiplist = torch.tensor(
                skiplist, device=self.device, dtype=torch.int32
            )

        self.pad_token_id = tokenizer.pad_token_id
    
    def save(self, save_directory: str) -> None:
        """
        Saves the models parameters (model.pt) and the config (colbert_config.json)
        in the given directory.
        """
        # create the directory if it doesn't exist
        os.makedirs(save_directory, exist_ok=True)

        # save the model state dict
        model_path = os.path.join(save_directory, "model.pt")
        torch.save(self.state_dict(), model_path)

        # save the model's config if available
        config_path = os.path.join(save_directory, "colbert_config.json")
        save_config(self.config, config_path)

    @classmethod
    def from_pretrained(
        cls, directory: str, device: Union[str, torch.device] = "cpu", config: Optional[BaseConfig] = None
    ) -> "ColBERT":
        # load the model's config if necessary
        if not isinstance(config, BaseConfig):
            config_path = os.path.join(directory, "colbert_config.json")
            config = load_config(config_path)
        # use the default config, if loading the model's config was unsuccessful
        # and the given config was None
        if not config:
            logging.warning("colbert_config.json does not exist, loading default config.")
            config = BaseConfig()

        # get randomly initialized model, it's parameters will be overridden later
        model = cls(config)

        # detects if the checkpoint is the official pretrained ColBERTv2 model
        colbertv2_checkpoint = "pytorch_model.bin" in os.listdir(directory)
        normal_checkpoint = "model.pt" in os.listdir(directory)

        if colbertv2_checkpoint:
            logging.info("Detected ColBERTv2 checkpoint. Loading the model!")
            model_path = os.path.join(directory, "pytorch_model.bin")
            
            # load the backbone using HuggingFace
            model.backbone = AutoModel.from_pretrained(
                directory, config=model.backbone_config
            )
            
            # override old checkpoint in the config
            config.checkpoint = directory

            # try to load the output layer
            if model._load_linear_weights():
                logging.info("Successfully loaded weights for last ColBERTv2 layer!")            
            
        elif normal_checkpoint:
            logging.info("Detected regular ColBERT checkpoint. Loading the model!")
            model_path = os.path.join(directory, "model.pt")
            # Load the state dict, ignoring size mismatch errors
            state_dict = torch.load(model_path, map_location=model.device)

            # extend the embedding matrix in case the checkpoint had a larger embedding matrix
            # this is the case if we load a RoBERTa checkpoint, since we
            # added 2 new tokens ([Q]/[D]-Token)
            if "backbone.embeddings.word_embeddings.weight" in state_dict:
                n_embs = state_dict["backbone.embeddings.word_embeddings.weight"].shape[0]
                model.backbone.resize_token_embeddings(n_embs)
            model.load_state_dict(state_dict, strict=True)

            # override old checkpoint in the config
            config.checkpoint = directory

        else:
            logging.warning(
                f"Could not load the model checkpoint `{model_path}`. Returning randomly initialized model."
            )
        
        model.to(device)
        return model


    def _load_linear_weights(self) -> bool:
        """
        Tries loading the last linear layer of a pretrained ColBERT implementation,
        which maps the BERT output vectors to the desired dimensionality (`self.out_features`).
        """
        if self.config.checkpoint is None or not os.path.exists(self.config.checkpoint):
            return False

        for file in os.listdir(self.config.checkpoint):
            path_to_weights = os.path.join(self.config.checkpoint, file)
            if not os.path.isfile(path_to_weights):
                continue
                
            if "pytorch_model" in file or ".pt" in file or ".pth" in file:
                try:
                    with open(path_to_weights, mode="br") as f:
                        parameters = torch.load(f, map_location=self.device)

                    if "linear.weight" in parameters.keys():
                        weights = parameters["linear.weight"]
                        # replace the weights if the number of input features is the same
                        if weights.shape[-1] == self.linear.weight.shape[-1]:
                            self.linear.weight.data = weights[: self.config.dim]
                            return True

                except Exception as e:
                    print(f"Couldn't load linear weights: {e}")
        
        return False

    @staticmethod
    def _load_model_config(config: BaseConfig) -> AutoConfig:
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
    queries = ["How are you today?", "Where do you live?"]
    passages = ["I'm great!", "Nowhere brudi."]

    MODEL_PATH = "roberta-base"  # "bert-base-uncased" or "roberta-base"
    DEVICE = "cuda:0"
    EPOCHS = 25

    config = BaseConfig(
        tok_name_or_path=MODEL_PATH,
        backbone_name_or_path=MODEL_PATH,
        similarity="cosine",
        intra_batch_similarity=False,
        epochs=EPOCHS,
        dim=24,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        dropout=0.1,
        passages_per_query=10,
    )

    
    tokenizer = ColBERTTokenizer(config)
    colbert = ColBERT(config, device=DEVICE)
    # colbert = ColBERT.from_pretrained("../../../data/colbertv2.0/", device=DEVICE, config=config)
    colbert.register_tokenizer(tokenizer)

    # colbert.save("testchen")
    # colbert = ColBERT.from_pretrained("testchen", device=DEVICE)
    # config = colbert.config
    # tokenizer = ColBERTTokenizer(config)
    # colbert.register_tokenizer(tokenizer)
    # print(colbert.linear.weight)
    # print(colbert_.linear.weight)

    optimizer = torch.optim.AdamW(colbert.parameters(), lr=3e-5)
    criterion = nn.CrossEntropyLoss()

    Q = tokenizer.tensorize(queries, mode="query", return_tensors="pt")
    P = tokenizer.tensorize(passages, mode="doc", return_tensors="pt")
    # out = colbert(Q, P)

    # QQP style:
    P = (P[0][:1], P[1][:1])

    # QPP style:
    # Q = (Q[0][:1], Q[1][:1])

    for epoch in range(1, EPOCHS + 1):
        optimizer.zero_grad()
        print(Q[0].shape, P[0].shape)
        out = colbert(Q, P)
        print(out.shape)
        B = out.shape[0]
        loss = criterion(out, torch.arange(0, B, device=DEVICE, dtype=torch.long))
        loss.backward()
        optimizer.step()
        print(loss.item())
        exit(0)

    colbert.eval()
    with torch.no_grad():
        out = colbert(Q, P)
        print(out, F.softmax(out, dim=-1))
