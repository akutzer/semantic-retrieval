#!/usr/bin/env python3
from retrieval.tokenization import DocTokenizer, QueryTokenizer
from retrieval.data import DataIterator
from retrieval.configs import BaseConfig

from tqdm import tqdm



if __name__ == "__main__":
    base_config = BaseConfig(tok_name_or_path="../data/colbertv2.0/")
    doc_tokenizer = DocTokenizer(base_config)
    query_tokenizer = QueryTokenizer(base_config)

    triples_path = "/home/aaron/Documents/Studium/Informatik/6_Semester/KP BigData/semantic-retrieval/data/fandom-qa/harry_potter_qa/triples.train.tsv"
    queries_path = "/home/aaron/Documents/Studium/Informatik/6_Semester/KP BigData/semantic-retrieval/data/fandom-qa/harry_potter_qa/queries.train.tsv"
    passages_path = "/home/aaron/Documents/Studium/Informatik/6_Semester/KP BigData/semantic-retrieval/data/fandom-qa/harry_potter_qa/passages.train.tsv"


    data_iter = DataIterator(base_config, triples_path, queries_path, passages_path)
    #data_iter.shuffle()
    EPOCHS = 10


    for epoch in range(1, EPOCHS+1):
        data_iter.reset()
        data_iter.shuffle()

        for batch in tqdm(data_iter):
            for sub_batch in batch:
                q_tokens, q_masks, p_tokens, p_masks = sub_batch
                #print(q_tokens.is_pinned())
                q_tokens.to("cuda:0")
                q_masks.to("cuda:0")
                p_tokens.to("cuda:0")
                p_masks.to("cuda:0")
                #exit(0)
    
    exit(0)
    out = doc_tokenizer.encode(["Wow this looks cool!!!! :D", "What does look so cool?"], add_special_tokens=True)
    print(out)

    out = doc_tokenizer.tensorize(["Wow this looks cool!!!! :D", "What does look so cool?"])
    print(out)

    """out = query_tokenizer.tokenize(["Wow this looks cool!!!! :D", "What does look so cool?"], add_special_tokens=True )
    print(out)"""