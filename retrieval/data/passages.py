#!/usr/bin/env python3
import logging
import pandas as pd


# TODO: Add WID support
class Passages:
    def __init__(self, path=None):
        self.path = path
        self.data = pd.Series(dtype="object")
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
            raise DeprecationWarning()
            # self.data = self._load_json(path)

        return self.data

    def _load_tsv(self, path, drop_nan=False):
        logging.basicConfig(level=logging.WARNING, format="[%(asctime)s][%(levelname)s] %(message)s", datefmt="%y-%m-%d %H:%M:%S")
        logging.warning("Passages currently drops the WID column!")
        delimiter = "\t" if path.endswith(".tsv") else ","
        passages = pd.read_csv(path, delimiter=delimiter, index_col=False)

        self._replace_nan(passages, drop_nan)
        pid, passage, *_ = passages.columns
        # convert the pandas.DataFrame with the columns QID and query
        # into a pandas.Series
        passages = passages.set_index(pid, drop=False)[passage]
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
                self.data[p] for p in pid if (not skip_non_existing) or p in self.keys()
            ]
        else:
            return (
                self.data[pid]
                if (not skip_non_existing) or pid in self.keys()
                else None
            )


if __name__ == "__main__":
    path = "../../data/ms_marco/ms_marco_v1_1/train/passages.tsv"
    passages = Passages(path=path)
    print(passages.data, end="\n\n")

    path = "../../data/fandoms_qa/harry_potter/train/passages.tsv"
    passages = Passages(path=path)
    print(passages.data, end="\n\n")

    print(len(passages))
    print(passages[0])
    print(passages.pid2string([0, 2 * len(passages)], skip_non_existing=True))
