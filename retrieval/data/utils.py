import os

from retrieval.configs import BaseConfig
from retrieval.data.dataset import TripleDataset



def load_dataset(config: BaseConfig, path: str, mode: str):
    assert mode.upper() in ["QQP", "QPP"]

    if not os.path.exists(path):
        raise ValueError(f"Path: `{path}` does not exist!")
    

    triples_path, queries_path, passages_path  = "", "", ""
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if not os.path.isfile(file_path):
            continue
            
        if file_path.endswith(".tsv"):
            if "triples" in file:
                triples_path = file_path
            elif "queries" in file:
                queries_path = file_path
            elif "passages" in file:
                passages_path = file_path
    
    dataset = TripleDataset(config, triples_path, queries_path, passages_path, mode=mode)
    return dataset



if __name__ == "__main__":
    config = BaseConfig()
    dataset = load_dataset(config, "../../data/fandoms_qa/harry_potter/train", mode="QQP")
    print(dataset)