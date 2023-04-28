import json
import csv



class Passages:
    def __init__(self, path=None):
        self.path = path
        self.__load_file(path)
    
    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data.items())

    def __getitem__(self, key):
        return self.data[key]

    def keys(self):
        return self.data.keys()

    def values(self):
        return self.data.values()

    def items(self):
        return self.data.items()

    def __load_file(self, path):
        if path.endswith((".csv", ".tsv")):
            self.data = self.__load_tsv(path)
        
        elif path.endswith(".json"):
            self.data = self.__load_json(path)

        return self.data

    def __load_tsv(self, path):
        delimiter = "\t" if path.endswith(".tsv") else ","

        passages = {}
        with open(path, mode="r", encoding="utf-8", newline="") as file:
            reader = csv.reader(file, delimiter=delimiter)
            for line in reader:
                pid, passage, *_ = line
                pid = int(pid)
                passages[pid] = passage
        
        return passages

    def __load_json(self, path):
        with open(path, mode="r", encoding="utf-8") as file:
            passages = json.load(file)
        return passages