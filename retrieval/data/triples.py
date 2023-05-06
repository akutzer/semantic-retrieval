#!/usr/bin/env python3
import warnings
import pandas as pd
import sklearn


class Triples:
    def __init__(self, path: str, mode: str, psgs_per_qry: int = None):
        self.path = path

        mode = mode.lower()
        assert mode in ["qqp", "qpp"], f"Mode must be either `QQP` or `QPP`, but was given as: {mode}"
        self.mode = mode

        if self.mode == "qqp" and psgs_per_qry is not None:
            warnings.warn("psgs_per_qry argument will be ignored if mode = `QQP`", DeprecationWarning)

        self._load_file(path, mode, psgs_per_qry)
    
    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    def __getitem__(self, idx):
        return self.data.iloc[idx].values.tolist()
    
    def _load_file(self, path, mode, psgs_per_qry=None):
        if path.endswith((".csv", ".tsv")):
            self.data = self._load_tsv(path, psgs_per_qry)
        
        elif path.endswith(".json"):
            self.data = self._load_json(path, psgs_per_qry)

        return self.data
    
    def _load_tsv(self, path, psgs_per_qry=None):
        delimiter = "\t" if path.endswith(".tsv") else ","

        triples = pd.read_csv(path, delimiter=delimiter, header=None)
        if psgs_per_qry is not None and self.mode == "qpp":
            triples = triples.iloc[:, :psgs_per_qry+1]
        self._rename_df(triples)
 
        return triples
    
    def _load_json(self, path, psgs_per_qry=None):
        triples = pd.read_json(path)
        if psgs_per_qry is not None and self.mode == "qpp":
            triples = triples.iloc[:, :psgs_per_qry+1]
        self._rename_df(triples)
        
        return triples
    
    def _rename_df(self, df: pd.DataFrame, psgs_per_qry=None):
        if psgs_per_qry is None:
            psgs_per_qry = df.shape[-1] - 2
 
        if self.mode == "qqp":
            # names = ["QID⁺"] + ["QID⁻"] * psgs_per_qry + ["PID"]
            names = ["QID⁺", "QID⁻", "PID"]
        else:
            names = ["QID", "PID⁺"] + ["PID⁻"] * psgs_per_qry

        names = {i: name for i, name in enumerate(names)}
        df.rename(names, axis=1, inplace=True)

        return df

    def shuffle(self, reset_index=False):
        self.data = sklearn.utils.shuffle(self.data)
        if reset_index:
            self.data = self.data.reset_index(drop=True)



if __name__ == "__main__":
    path = "../../data/fandom-qa/witcher_qa/triples.train.json"

    triples = Triples(path=path, mode="QPP", psgs_per_qry=None)
    print(triples[0], triples.data.loc[0].tolist())
    triples.shuffle(reset_index=False)
    print(triples[0], triples.data.loc[0].tolist())
    triples.shuffle(reset_index=True)
    print(triples[0], triples.data.loc[0].tolist())

    # print(triples.data)
    # print(len(triples))
    # print(triples[-4])
