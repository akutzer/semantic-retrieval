#!/usr/bin/env python3
import os

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from retrieval.data import Queries, Passages, Triples
from retrieval.models.basemodels.metrics import Metrics

'''
#   Calculates the dot product between every paragraph tf-idf vector and the query tf-idf vector and chooses the best . 
#   (X@Q.T, where X is matrix of all paragraph vectors and Q is the query matrix)
'''

FOLDERS = ['C:/Daten/Florian/Dev/Python/semantic-retrieval/data/fandom_qa/harry_potter_qa_small',
           # ,'../../../data/fandom_qa/witcher_qa'
           ]

# number of paragraphs that are acceptable for a question
K = 5

# def read_df_from_tsv(file, column_names):
#     return pd.read_csv(file, sep='\t', header=None).rename(columns=dict(enumerate(column_names)))

class TfIdf():

    def __init__(self, passages):
        self.vectorizer, self.X = self.tfIDFCreatorFromArr(arr=passages)
        self.X = self.X#.toarray()


    def tfIDFCreatorFromArr(self, arr, min_df=5):
        vectorizer = TfidfVectorizer(strip_accents='unicode', min_df=min_df)
        X = vectorizer.fit_transform(arr)
        return vectorizer, X


    def answerQuestion(self, questions, k):
        Q = self.vectorizer.transform(questions).T
        M = self.X @ Q
        M = M.toarray()
        max_ind = np.argsort(-M, axis=0)

        return max_ind[:k, :].T

    def best_and_worst_pids(self, questions, topk, bottomk):
        Q = self.vectorizer.transform(questions).T
        M = self.X @ Q
        M = M.toarray()
        max_ind = np.argsort(-M, axis=0)

        return max_ind[:topk, :].T, max_ind[-bottomk:, :].T


if __name__ == "__main__":
    # #
    tf_idf = TfIdf(passages=["halger gr eehe heh er e", " gaeh rh4hz 4 4 sr 4wuz45z shae5z "
                                    , " gaeh rh4hz 4 sdgsg shae5z ", " gaeh rhGEWgwegw45z shae5z "
                                    , " gaeh rh4hz 4 sdgsg shae5z ", " gaeh rhGEWgwegw45z shae5z "
                                    , " gaeh rh4hz 4 sdgsg shae5z ", " gaeh rhGEWgwegw45z shae5z "
                                    , " gaeh rh4hz 4 sdgsg shae5z ", " gaeh rhGEWgwegw45z shae5z "
                                    , "hallo", " gaeh rhGEWgwegw45z shae5z aaron god snape"
                                 ])
    # # queries = ["hallo ? ", "syasuu a6g f? ", "ssrjusrjusrg f? ","srtzurursg f? "]
    # # best_k = tf_idf.batchBestKPIDs(3, queries)
    # # print(best_k)
    # tf_idf = TfIdf(folders=FOLDERS)
    print(tf_idf.answerQuestion(["who killed severus snape"], 5))
    print(tf_idf.answerQuestion(["what is god"], 5))
    print(tf_idf.answerQuestion(["is aaron god"], 5))
    print(tf_idf.answerQuestion(["does god love me"], 5))
    # tf_idf.evaluate_folders(8, 3)
    # tf_idf.printMeanWallAndCPUTime()