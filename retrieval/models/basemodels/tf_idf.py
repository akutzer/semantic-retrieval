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

FOLDERS = ['../../../data/fandoms_qa/harry_potter/all'
           #,'../../../data/fandom_qa/witcher_qa'
           ]

# number of paragraphs that are acceptable for a question
K = 5

# def read_df_from_tsv(file, column_names):
#     return pd.read_csv(file, sep='\t', header=None).rename(columns=dict(enumerate(column_names)))

class TfIdf():

    def __init__(self, passages=None, folders=None, combine_paragraphs=True, mode="qpp"):
        if folders:
            self.folders = folders
            self.passage_files = [x + '/' + y for x in folders for y in os.listdir(x) if 'passages' in y and '.tsv' in y]
            self.question_files = [x + '/' + y for x in folders for y in os.listdir(x) if 'queries' in y and '.tsv' in y]
            self.triple_files = [x + '/' + y for x in folders for y in os.listdir(x) if 'triples' in y and '.tsv' in y]
            self.folder_objects = [(Passages(passage_file), Queries(question_file), Triples(triple_file, mode)) for passage_file,question_file, triple_file in zip(self.passage_files, self.question_files, self.triple_files)]
            self.paragraphs = []
            passages_objs = list(zip(*self.folder_objects))[0]
            for passage_obj in passages_objs:
                self.paragraphs = self.paragraphs + list(passage_obj.values())

            if passages:
                print("Passages and folders are defined. passages are ignored!")
        elif passages:
            self.paragraphs = passages
        else:
            print("Either passages or folders has to be defined!")


        self.metrics = Metrics()
        self.vectorizer, self.X = self.tfIDFCreatorFromArr(self.paragraphs)
        self.X = self.X.T #.toarray()


    # def __init__(cls, passages):
    #     #cls.vectorizer, cls.X = cls.tfIDFCreatorFromArr(passsages)
    #     #cls.X = cls.X.toarray()
    #     #cls.vectorizer = TfidfVectorizer(strip_accents='unicode', min_df=10)
    #     print("fromPassages created")
    #     cls.vectorizer, cls.X = cls.tfIDFCreatorFromArr(arr=passages)
    #     cls.X = cls.X.toarray()



    def answerQuestion(self, question, k):
        #measure time
        self.metrics.startCPUTime()
        self.metrics.startWallTime()

        Q = self.vectorizer.transform([question])
        M = Q @ self.X
        max_ind = np.argsort(-M.toarray(), axis=-1)

        # topk_idcs = 
        # max_ind = np.argsort(-M, axis=0).flatten()
        # print(type(max_ind), max_ind.shape)
        # best_k = [(max_ind[i],self.paragraphs[max_ind[i]]) for i in range(k)]

        # measure time
        self.metrics.stopCPUTime(1)
        self.metrics.stopWallTime(1)

        return max_ind[:, :k]




    def tfIDFCreatorFromArr(self, arr, min_df=10):
        vectorizer = TfidfVectorizer(strip_accents='unicode', min_df=min_df)
        X = vectorizer.fit_transform(arr)
        return vectorizer, X


    def batchBestKPIDs(self, k, Q):
        # order is the same i googled it
        #row_pid_mapping = dict(enumerate(passages.keys()))
        #col_qid_mapping = dict(enumerate(queries.keys()))

        # create tf-idf vectorizer and matrix
        #vectorizer, X = self.tfIDFCreatorFromArr(passages.values())


        '''input: embeddings of passages and batch of queries
        returns: '''
        Q = self.vectorizer.transform(Q)
        M = Q @ self.X

        max_ind = np.argsort(-M.toarray(), axis=-1)
        best_k = max_ind[:,:k]
        scores = np.take_along_axis(M, best_k, axis=-1)

        return scores.toarray(), best_k


    def evaluate_folders(self,k,beta):
        for i in range(len(self.folders)):
            paragraphs, queries, qpp_triples = self.folder_objects[i]

            #measuring time
            query_count = len(qpp_triples)
            self.metrics.startCPUTime()
            self.metrics.startWallTime()

            # order is the same i googled it
            row_pid_mapping = dict(enumerate(paragraphs.keys()))
            col_qid_mapping = dict(enumerate(queries.keys()))

            # create tf-idf vectorizer and matrix
            vectorizer, X = self.tfIDFCreatorFromArr(paragraphs.values())

            Q = vectorizer.transform(queries.values()).T
            
            Q = Q.toarray()
            X = X.toarray()
            M = X@Q

            self.metrics.evaluateDatasetIntoM(M, row_pid_mapping, col_qid_mapping, dataset_name=self.folders[i], qpp_triples=qpp_triples)

            #measuring time
            self.metrics.stopCPUTime(query_count)
            self.metrics.stopWallTime(query_count)

            self.metrics.printStatistics(k, beta)

    def printMeanWallAndCPUTime(self):
        self.metrics.printMeanWallAndCPUTime()


if __name__ == "__main__":
    #
    # tf_idf = TfIdf(passages=["halger gr eehe heh er e", " gaeh rh4hz 4 4 sr 4wuz45z shae5z "
    #                                 , " gaeh rh4hz 4 sdgsg shae5z ", " gaeh rhGEWgwegw45z shae5z "
    #                                 , " gaeh rh4hz 4 sdgsg shae5z ", " gaeh rhGEWgwegw45z shae5z "
    #                                 , " gaeh rh4hz 4 sdgsg shae5z ", " gaeh rhGEWgwegw45z shae5z "
    #                                 , " gaeh rh4hz 4 sdgsg shae5z ", " gaeh rhGEWgwegw45z shae5z "
    #                                 , "hallo", " gaeh rhGEWgwegw45z shae5z "
    #                              ])
    # queries = ["hallo ? ", "syasuu a6g f? ", "ssrjusrjusrg f? ","srtzurursg f? "]
    # best_k = tf_idf.batchBestKPIDs(3, queries)
    # print(best_k)
    tf_idf = TfIdf(folders=FOLDERS)
    print(tf_idf.answerQuestion("who killed severus snape", 5))
    print(tf_idf.answerQuestion("what is god", 5))
    print(tf_idf.answerQuestion("is aaron god", 5))
    print(tf_idf.answerQuestion("does god love me", 5))
    tf_idf.evaluate_folders(8, 3)
    tf_idf.printMeanWallAndCPUTime()