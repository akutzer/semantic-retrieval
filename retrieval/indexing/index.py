#!/usr/bin/env python3
from dataclasses import dataclass
import torch

from retrieval.data import Passages
from retrieval.models import ColBERTInference, load_colbert_and_tokenizer, inference_to_embedding
from retrieval.indexing.colbert_indexer import ColBERTIndexer



@dataclass
class IndexConfig:
    passages_path: str
    checkpoint_path: str
    index_path: str = ""
    batch_size: int = 8
    use_gpu: bool = False
    device: str = torch.device("cpu")
    dtype: torch.dtype = torch.float16    


def argparser2index_config(args):
    device = "cuda" if args.use_gpu and torch.cuda.is_available() else "cpu"

    if args.dtype.upper() == "FP16":
        dtype = torch.float16
    elif args.dtype.upper() == "FP32":
        dtype = torch.float32
    else:
        dtype = torch.float64

    config = IndexConfig(
        passages_path = args.passages_path,
        checkpoint_path = args.checkpoint_path,
        index_path = args.index_path,
        use_gpu= args.use_gpu,
        device=device,
        dtype=dtype,
        batch_size = args.batch_size
    )

    return config


def index(inference: ColBERTInference, config: IndexConfig, store: bool = False):
    # initialize the indexer
    device = "cuda" if config.use_gpu and torch.cuda.is_available() else "cpu"
    indexer = ColBERTIndexer(inference, device=device, dtype=config.dtype)

    # load the data
    passages = Passages(config.passages_path)
    data = passages.values().tolist()
    pids = passages.keys().tolist()

    # start indexing
    indexer.index(data, pids, bsize=config.batch_size)

    # store the indexing
    if store:
        indexer.save(config.index_path)

    return indexer



if __name__ == "__main__":
    import argparse

    # enable TensorFloat32 tensor cores for float32 matrix multiplication if available
    torch.set_float32_matmul_precision("high")


    parser = argparse.ArgumentParser(description="ColBERT Indexing")
    parser.add_argument("--passages-path", type=str, required=True, help="Path to the TSV-file containing the passages")
    parser.add_argument("--checkpoint-path", type=str, required=True, help="Path to ColBERT Checkpoint which should be used for indexing")
    parser.add_argument("--index-path", type=str, required=True, help="Path to where the computed indices should be saved to")
    parser.add_argument("--use-gpu", action="store_true", help="Use GPU for indexing (recommended)")
    parser.add_argument("--dtype", type=str, default="FP16", choices=["FP16", "FP32", "FP64"], help="Floating-point precision of the indices")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for used during tee indexation")
    parser.add_argument("--embedding-only", action="store_true", help="This used only the word embedding layer of the ColBERT model")

    args = parser.parse_args()
    config = argparser2index_config(args)
    print(config)
 
    # instantiate model
    colbert, tokenizer = load_colbert_and_tokenizer(config.checkpoint_path)
    inference = ColBERTInference(colbert, tokenizer)

    if args.embedding_only:
        inference = inference_to_embedding(inference, just_word_emb=False, layer_norm=True)
    
    # run the indexation
    index(inference, config, store=True)
