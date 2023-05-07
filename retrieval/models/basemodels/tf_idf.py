import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from retrieval.data.queries import Queries
from retrieval.data.passages import Passages
from retrieval.data.triples import Triples
from retrieval.data.dataset import TripleDataset
from retrieval.models.basemodels.metrics import Metrics 
from retrieval.configs import BaseConfig




'''
#   Calculates the dot product between every paragraph tf-idf vector and the query tf-idf vector and chooses the best . 
#   (X@Q.T, where X is matrix of all paragraph vectors and Q is the query matrix)
'''

FOLDERS = ['../../../data/fandom-qa/harry_potter_qa'
           #'../../../data/fandom_qa/witcher_qa'
           ]

# number of paragraphs that are acceptable for a question
K = 5

# def read_df_from_tsv(file, column_names):
#     return pd.read_csv(file, sep='\t', header=None).rename(columns=dict(enumerate(column_names)))

class TfIdf():
    def __init__(self, folders, combine_paragraphs=True, mode="qpp"):
        self.folders = folders
        self.passage_files = [x + '/' + y for x in folders for y in os.listdir(x) if 'passages' in y and '.tsv' in y]
        self.question_files = [x + '/' + y for x in folders for y in os.listdir(x) if 'queries' in y and '.tsv' in y]
        self.triple_files = [x + '/' + y for x in folders for y in os.listdir(x) if 'triples' in y and '.tsv' in y]

        self.folder_objects = [(Passages(passage_file), Queries(question_file), Triples(triple_file, mode)) for passage_file,question_file, triple_file in zip(self.passage_files, self.question_files, self.triple_files)]

        self.paragraphs = None
        if combine_paragraphs:
            self.paragraphs = np.array([passages.values() for passages in list(zip(*self.folder_objects))[0]]).flatten()

        self.vectorizer, self.X = self.tfIDFCreatorFromArr(self.paragraphs)


    def answerQuestion(self, question):
        Q = self.vectorizer.transform([question]).T
        X = self.X.toarray()
        M = X@Q

        max_ind = np.argmax(M)
        return max_ind, self.paragraphs[max_ind]




    def tfIDFCreatorFromArr(self, arr, min_df=10):
        vectorizer = TfidfVectorizer(strip_accents='unicode', min_df=min_df)
        X = vectorizer.fit_transform(arr)
        return vectorizer, X
    
    
    def metrics(self,k):
        for i in range(len(self.folders)):
            paragraphs, queries, qpp_triples = self.folder_objects[i]

            # order is the same i googled it
            row_pid_mapping = dict(enumerate(paragraphs.keys()))
            col_qid_mapping = dict(enumerate(queries.keys()))

            # create tf-idf vectorizer and matrix
            vectorizer, X = self.tfIDFCreatorFromArr(paragraphs.values())

            Q = vectorizer.transform(queries.values()).T
            
            Q = Q.toarray()
            X = X.toarray()
            M = X@Q

            metrics = Metrics(M, row_pid_mapping, col_qid_mapping, dataset_name=self.folders[i], qpp_triples=qpp_triples)
            metrics.printStatistics(6,1)





if __name__ == "__main__":
    tf_idf = TfIdf(FOLDERS)
    print(tf_idf.answerQuestion("who killed severus snape"))
    # tf_idf.metrics(5)
