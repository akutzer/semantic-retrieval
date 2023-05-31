import pandas as pd
from retrieval.models.basemodels import TfIdf
import numpy as np
import swifter
import os
import pandas as pd

SEARCH_PATH = "../../data/"

def positionPidQid(best_ind_in_m, pid, qid):
    return best_ind_in_m[qid,pid]

def passageFromTriple(triple, df_passages, df_queries):
    passages = df_passages.iloc[triple[1:]]['passage'].values
    print(passages)
    tf_idf = TfIdf(passages)
    k = len(passages)
    queries = [df_queries.iloc[triple[0]]['query']]
    print('-----')
    print(queries)

    best_ind_all = tf_idf.answerQuestion(queries,k)
    best_ind_in_m = np.argsort(best_ind_all, axis=1)

    print(best_ind_in_m.shape)
    print(best_ind_in_m[0,0])
    return best_ind_in_m[0,0]


def statistics(path, passages_from_triples=False):
    print(path)

    df_passages = pd.read_csv(path + '/passages.tsv', sep ='\t')
    df_queries = pd.read_csv(path + '/queries.tsv', sep ='\t')
    df_triple = pd.read_csv(path + '/triples.tsv', sep ='\t')

    if passages_from_triples:
        df_triple['pos+10'] = df_triple.swifter.apply(lambda row : passageFromTriple(row, df_passages, df_queries), axis=1)

    else:
        passages = df_passages.passage.values
        queries = df_queries['query'].values
        tf_idf = TfIdf(passages)
        k = len(passages)
        best_ind_all = tf_idf.answerQuestion(queries,k)
        best_ind_in_m = np.argsort(best_ind_all, axis=1)
        
        df_triple['pos+'] = df_triple.swifter.apply(lambda row : positionPidQid(best_ind_in_m, row["PID"], row["QID+"]), axis=1)
        df_triple['pos-'] = df_triple.swifter.apply(lambda row : positionPidQid(best_ind_in_m, row["PID"], row["QID-"]), axis=1)

        mrrplus = df_triple['pos+'].apply(lambda x: 1.0/(x+1)).sum() / len(df_triple)
        mrrminus = df_triple['pos-'].apply(lambda x: 1.0/(x+1)).sum() / len(df_triple)
        
        

        print('describe pos+')
        print(f'MRR pos+: {mrrplus}')
        print(df_triple['pos+'].describe())

        print('describe pos-')
        print(f'MRR pos-: {mrrminus}')
        print(df_triple['pos+'].describe())

        print('----------------------------')


if __name__ == "__main__":
    for root, dirs, files in os.walk(SEARCH_PATH):
    
        # is a target folder
        if 'passages.tsv' in files and ('MS' in root ):
            continue
            print(statistics(root, passages_from_triples=True))
        elif 'passages.tsv' in files and ('/all' in root ):
            statistics(root, passages_from_triples=False)