import json
import csv



class Triples:
    def __init__(self, path=None, psgs_per_qry=None):
        self.path = path
        self.__load_file(path, psgs_per_qry)
    
    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def __load_file(self, path, psgs_per_qry=None):
        if path.endswith((".csv", ".tsv")):
            self.data = self.__load_tsv(path, psgs_per_qry)
        
        elif path.endswith(".json"):
            self.data = self.__load_json(path, psgs_per_qry)

        return self.data

    def __load_tsv(self, path, psgs_per_qry=None):
        delimiter = "\t" if path.endswith(".tsv") else ","

        triples = []
        with open(path, mode="r", encoding="utf-8", newline="") as file:
            reader = csv.reader(file, delimiter=delimiter)
            for line in reader:
                qid, *pids = line
                qid = int(qid)
                pids = list(map(int, filter(str.isdigit, pids[:psgs_per_qry])))
                triples.append([qid] + pids)
        
        return triples

    def __load_json(self, path, psgs_per_qry=None):
        with open(path, mode="r", encoding="utf-8") as file:
            triples = json.load(file)
        
        if psgs_per_qry is not None:
            triples[:] = [triple[:psgs_per_qry + 1] for triple in triples]
            
        return triples