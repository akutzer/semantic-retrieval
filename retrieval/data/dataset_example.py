from tqdm import tqdm
import numpy as np
np.random.seed(125)

from retrieval.configs import BaseConfig
from retrieval.models import ColBERTTokenizer
from retrieval.data.dataset import TripleDataset
from retrieval.data.dataloader import BucketIterator




# config = BaseConfig(passages_per_query=1)
# triples_path = "../../data/fandom-qa/witcher_qa/triples.train.tsv"
# queries_path = "../../data/fandom-qa/witcher_qa/queries.train.tsv"
# passages_path = "../../data/fandom-qa/witcher_qa/passages.train.tsv"
# dataset = TripleDataset(config, triples_path, queries_path, passages_path, mode="QPP")

# # for i, triple in enumerate(tqdm(dataset)):
# #     qid, pid_pos, *pid_neg = triple
# #     query, psg_pos, *psg_neg = dataset.id2string(triple)
# #     # print(triple, query, psg_pos, psg_neg, sep="\n", end="\n\n")



# config = BaseConfig(passages_per_query=10)
# triples_path = "../../data/ms_marco_v2.1/train/triples.train.tsv"
# queries_path = "../../data/ms_marco_v2.1/train/queries.train.tsv"
# passages_path = "../../data/ms_marco_v2.1/train/passages.train.tsv"
# dataset = TripleDataset(config, triples_path, queries_path, passages_path, mode="QPP")

# for i, triple in enumerate(tqdm(dataset)):
#     qid, pid_pos, *pid_neg = triple
#     query, psg_pos, *psg_neg = dataset.id2string(triple)
    # print(triple, query, psg_pos, psg_neg, sep="\n", end="\n\n")



config = BaseConfig(
        bucket_size=16*4,
        batch_size=32,
        accum_steps=2,
        passages_per_query=10)
triples_path = "../../data/ms_marco_v1.1/train/triples.train.tsv"
queries_path = "../../data/ms_marco_v1.1/train/queries.train.tsv"
passages_path = "../../data/ms_marco_v1.1/train/passages.train.tsv"

dataset = TripleDataset(config, triples_path, queries_path, passages_path, mode="QPP")
tokenize = ColBERTTokenizer(config)
data_iter = BucketIterator(config, dataset, tokenize)



data_iter.shuffle()
for i, bucket in enumerate(tqdm(data_iter)):
    for batch in bucket:
        Q, P = batch
        (q_tokens, q_masks), (p_tokens, p_masks) = Q, P
        # print(q_tokens.shape, p_tokens.shape)
        # print(tokenize.decode(q_tokens[0]), tokenize.decode(p_tokens[0]), tokenize.decode(p_tokens[-1]))
        # exit(0)