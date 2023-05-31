import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from retrieval.data.queries import Queries
from retrieval.data.passages import Passages
from retrieval.data.triples import Triples
from retrieval.data.dataset import TripleDataset
from retrieval.models.basemodels.metrics import Metrics 
from retrieval.configs import BaseConfig
import time




'''
#   Calculates the dot product between every paragraph tf-idf vector and the query tf-idf vector and chooses the best . 
#   (X@Q.T, where X is matrix of all paragraph vectors and Q is the query matrix)
'''

class TfIdf():

    def tfIDFCreatorFromArr(self, arr, min_df=5):
        vectorizer = TfidfVectorizer(strip_accents='unicode', min_df=min_df)
        X = vectorizer.fit_transform(arr)
        return vectorizer, X


    def __init__(self, passages):
        self.vectorizer, self.X = self.tfIDFCreatorFromArr(arr=passages)
        self.X = self.X.toarray()



    def answerQuestion(self, questions, k):
        Q = self.vectorizer.transform(questions).T
        M = self.X@Q
        max_ind = np.argsort(-M, axis=0)

        return max_ind[:k, :].T






if __name__ == "__main__":
    tf_idf = TfIdf(['hallo', "ich ist bin klaus", 'igor ist ein igel', 'salami schmeckt gut auf pizza'])
    print(tf_idf.answerQuestion(['klaus bin wer?','igor ist wer', 'pizza salami'],2))
