#!/usr/bin/env python3
import pandas as pd


class Queries:
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
        return self.data.items()

    def _load_file(self, path):
        if path.endswith((".csv", ".tsv")):
            self.data = self._load_tsv(path)

        elif path.endswith(".json"):
            raise DeprecationWarning()
            # self.data = self._load_json(path)

        return self.data

    def _load_tsv(self, path, drop_nan=False):
        delimiter = "\t" if path.endswith(".tsv") else ","
        queries = pd.read_csv(path, delimiter=delimiter, index_col=False)

        self._replace_nan(queries, drop_nan)
        qid, query, *_ = queries.columns
        # convert the pandas.DataFrame with the columns QID and query
        # into a pandas.Series
        queries = queries.set_index(qid, drop=False)[query]
        self._rename_axis(queries)
        return queries

    # def _load_json(self, path, drop_nan=False):
    #     queries = pd.read_json(path, typ="series")
    #     self._replace_nan(queries, drop_nan)
    #     self._rename_axis(queries)
    #     return queries

    def _replace_nan(self, series: pd.Series, drop_nan=False):
        if drop_nan:
            series.replace("", pd.NaT, inplace=True)
            series.dropna(inplace=True)  # drop rows with NaN/NaT values
        else:
            series.fillna("", inplace=True)

        return series

    def _rename_axis(self, series: pd.Series):
        series.rename_axis("QID", inplace=True)
        return series

    def qid2string(self, qid, skip_non_existing=False):
        if isinstance(qid, list):
            return [
                self.data[q] for q in qid if (not skip_non_existing) or q in self.keys()
            ]
        else:
            return (
                self.data[qid]
                if (not skip_non_existing) or qid in self.keys()
                else None
            )


if __name__ == "__main__":
    # path = "../../data/ms_marco/ms_marco_v2_1/train/queries.tsv"
    # queries = Queries(path=path)
    # print(queries.data, end="\n\n")

    path = "../../data/fandoms_qa/fandoms_all/all/queries.tsv"
    queries = Queries(path=path)
    print(queries.data, end="\n\n")

    print(queries[623686])
    print(queries[271434])

    print(len(queries))
    print(queries[0])
    print(queries.qid2string([0, 2 * len(queries)], skip_non_existing=True))
