#!/usr/bin/env python3
from tokenization import DocTokenizer, QueryTokenizer
from configs import BaseConfig



if __name__ == "__main__":
    base_config = BaseConfig()
    doc_tokenizer = DocTokenizer(base_config)
    query_tokenizer = QueryTokenizer(base_config)

    out = doc_tokenizer.encode(["Wow this looks cool!!!! :D", "What does look so cool?"], add_special_tokens=True)
    print(out)

    out = doc_tokenizer.tensorize(["Wow this looks cool!!!! :D", "What does look so cool?"])
    print(out)

    """out = query_tokenizer.tokenize(["Wow this looks cool!!!! :D", "What does look so cool?"], add_special_tokens=True )
    print(out)"""