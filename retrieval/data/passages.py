#!/usr/bin/env python3
import pandas as pd



class Passages:
    def __init__(self, path=None, ignore_wid=True):
        self.path = path
        self.ignore_wid = ignore_wid
        self.data = pd.Series(dtype="object")
        self._load_file(path)

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        data = self.data.iloc[:, 0] if self.ignore_wid else self.data
        return iter(data.items())

    def __getitem__(self, key):
        item = self.data.loc[key]
        if self.ignore_wid:
            if isinstance(key, list):
                return item.iloc[:, 0] #.tolist()
            else:
                return item[0]
        return item

    def keys(self):
        return  self.data.iloc[:, 0].index if self.ignore_wid else self.data.index

    def values(self):
        return self.data.iloc[:, 0].values if self.ignore_wid else self.data.values

    def items(self):
        return self.data.iloc[:, 0].items() if self.ignore_wid else self.data.items()

    def _load_file(self, path):
        if path.endswith((".csv", ".tsv")):
            self.data = self._load_tsv(path, drop_nan=True)

        elif path.endswith(".json"):
            raise DeprecationWarning()
            # self.data = self._load_json(path)

        return self.data

    def _load_tsv(self, path, drop_nan=False):
        delimiter = "\t" if path.endswith(".tsv") else ","
        passages = pd.read_csv(path, delimiter=delimiter, index_col=False)

        self._replace_nan(passages, drop_nan)
        pid, passage, wid, *_ = passages.columns
        passages = passages.set_index(pid, drop=False)[[passage, wid]]
        self._rename_axis(passages)
        return passages

    # def _load_json(self, path, drop_nan=False):
    #     passages = pd.read_json(path, typ="series")
    #     self._replace_nan(passages, drop_nan)
    #     self._rename_axis(passages)
    #     return passages

    def _replace_nan(self, series: pd.Series, drop_nan=False):
        if drop_nan:
            series.replace("", pd.NaT, inplace=True)
            series.dropna(inplace=True)  # drop rows with NaN/NaT values
        else:
            series.fillna("", inplace=True)

        return series

    def _rename_axis(self, series: pd.Series):
        series.rename_axis("PID", inplace=True)
        return series

    def pid2string(self, pid, skip_non_existing=False):
        if isinstance(pid, list):
            return [
                self[p] for p in pid if (not skip_non_existing) or p in self.keys()
            ]
        else:
            return (
                self[pid] if (not skip_non_existing) or pid in self.keys() else None
            )


if __name__ == "__main__":
    path = "../../data/ms_marco/ms_marco_v1_1/train/passages.tsv"
    passages = Passages(path=path)
    print(passages.data, end="\n\n")

    path = "../../data/fandoms_qa/harry_potter/train/passages.tsv"
    passages = Passages(path=path)
    print(passages.data, end="\n\n")
    
    print(len(passages))
    print(passages.values())
    print(passages[173654], type(passages[173654]))
    print(passages[[173654, 173655]], type(passages[[173654, 173655]]))
    print(passages.pid2string([173654, 2 * len(passages)], skip_non_existing=True))
