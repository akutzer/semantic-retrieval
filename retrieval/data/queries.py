#!/usr/bin/env python3
import pandas as pd

class Queries:
    def __init__(self, path=None):
        self.path = path
        self._load_file(path)
    
    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data.items())

    def __getitem__(self, key):
        return self.data.loc[key]

    def keys(self):
        return self.data.index

    def values(self):
        return self.data.values

    def items(self):
        return self.data.iteritems()

    def _load_file(self, path):
        if path.endswith((".csv", ".tsv")):
            self.data = self._load_tsv(path)
        
        elif path.endswith(".json"):
            self.data = self._load_json(path)

        return self.data

    def _load_tsv(self, path):
        delimiter = "\t" if path.endswith(".tsv") else ","
        queries = pd.read_csv(path, delimiter=delimiter, index_col=False, header=None)
        queries = queries.set_index(0, drop=False)[1]
        self._rename_df(queries)
        return queries
    
    def _load_json(self, path):
        queries = pd.read_json(path, typ="series")
        self._rename_df(queries)
        return queries
    
    def _rename_df(self, df: pd.DataFrame):
        df.rename_axis("QID", inplace=True)
        return df



if __name__ == "__main__":
    path = "../../data/fandom-qa/witcher_qa/queries.train.tsv"

    queries = Queries(path=path)
    print(queries.data)
    print(len(queries))
    print(queries[0])
