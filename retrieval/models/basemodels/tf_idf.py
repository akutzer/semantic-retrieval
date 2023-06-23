#!/usr/bin/env python3
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

'''
#   Calculates the dot product between every paragraph tf-idf vector and the query tf-idf vector and chooses the best . 
#   (X@Q.T, where X is matrix of all paragraph vectors and Q is the query matrix)
'''

class TfIdf():

    def __init__(self, passages, mapping_rowInd_pid=None):
        self.vectorizer, self.X = self.tfIDFCreatorFromArr(arr=passages)
        self.X = self.X.T # shape: (D, N_doc)
        self.mapping_row_pid = mapping_rowInd_pid

    def mapRowIndexToPid(self, ind):
        if not self.mapping_row_pid:
            return ind
        else:
            return self.mapping_row_pid[ind]

    
    def tfIDFCreatorFromArr(self, arr, min_df=5):
        vectorizer = TfidfVectorizer(strip_accents='unicode', min_df=min_df)
        X = vectorizer.fit_transform(arr)
        return vectorizer, X


    def answerQuestion(self, questions, k):
        Q = self.vectorizer.transform(questions) # shape: (B, D)
        M = (Q @ self.X).toarray() # shape: (B, N_doc)
        max_ind = np.argsort(-M, axis=-1)

        vecfun = np.vectorize(self.mapRowIndexToPid)
        max_ind = vecfun(max_ind)

        return max_ind[:, :k]
    
    def batchBestKPIDs(self, questions, k):
        Q = self.vectorizer.transform(questions) # shape: (B, D)
        M = (Q @ self.X).toarray() # shape: (B, N_doc)
        max_ind = np.argsort(-M, axis=-1)

        best_k = max_ind[:, :k]
        scores = np.take_along_axis(M, best_k, axis=-1)

        vecfun = np.vectorize(self.mapRowIndexToPid)
        best_k_pids = vecfun(best_k)

        return scores, best_k_pids

    def best_and_worst_pids(self, questions, topk, bottomk):
        Q = self.vectorizer.transform(questions) # shape: (B, D)
        M = (Q @ self.X).toarray() # shape: (B, N_doc)
        max_ind = np.argsort(-M, axis=-1)

        vecfun = np.vectorize(self.mapRowIndexToPid)
        max_ind = vecfun(max_ind)

        return max_ind[:, :topk], max_ind[:, -bottomk:]


if __name__ == "__main__":
    data = [
        "halger gr eehe heh er e", " gaeh rh4hz 4 4 sr 4wuz45z shae5z "
        , " gaeh rh4hz 4 sdgsg shae5z ", " gaeh rhGEWgwegw45z shae5z "
        , " gaeh rh4hz 4 sdgsg shae5z ", " gaeh rhGEWgwegw45z shae5z "
        , " gaeh rh4hz 4 sdgsg shae5z ", " gaeh rhGEWgwegw45z shae5z "
        , " gaeh rh4hz 4 sdgsg shae5z ", " gaeh rhGEWgwegw45z shae5z "
        , "hallo", " gaeh rhGEWgwegw45z shae5z aaron god snape"
    ]
    row2pid = dict(enumerate(range(len(data))))

    tf_idf = TfIdf(passages=data, mapping_rowInd_pid=row2pid)

    print(tf_idf.answerQuestion(["who killed severus snape"], 5))
    print(tf_idf.answerQuestion(["what is god", "is aaron god", "does god love me"], 5))
    print(tf_idf.best_and_worst_pids(["who killed severus snape"], 2, 2))
