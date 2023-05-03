import pandas as pd
import os 
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import swifter
from retrieval.data.queries import Queries
from retrieval.data.passages import Passages
from retrieval.data.triples import Triples
from retrieval.models.basemodels.metrics import Metrics 



'''
#   Evaluates test data by using tf-idf as a base model. Measurement: is one of the best -k- predicted answers the correct one?
#   Calculates the dot product between every paragraph tf-idf vector and the query tf-idf vector and chooses the largest -k- scores. 
#   (X@q_vec.T, where X is matrix of all paragraph vectors and q_vec is the query vector)
'''

FOLDERS = [# '../../../data/fandom-qa/harry_potter_qa'
           '../../../data/fandom-qa/witcher_qa_2'
           ]

# number of paragraphs that are acceptable for a question
K = 5

def read_df_from_tsv(file, column_names):
    return pd.read_csv(file, sep='\t', header=None).rename(columns=dict(enumerate(column_names)))


def tfIDFCreatorFromArr(arr, min_df=10):
    vectorizer = TfidfVectorizer(strip_accents='unicode', min_df=min_df)
    X = vectorizer.fit_transform(arr)
    return vectorizer, X


def tfIDF(k, FOLDERS):
    passage_files = [x + '/' + y for x in FOLDERS for y in os.listdir(x) if 'passages' in y and '.tsv' in y]
    question_files = [x + '/' + y for x in FOLDERS for y in os.listdir(x) if 'queries' in y and '.tsv' in y]
    triple_files = [x + '/' + y for x in FOLDERS for y in os.listdir(x) if 'triples' in y and '.tsv' in y]

    for i in range(len(FOLDERS)):
        #p_df = read_df_from_tsv(passage_files[i], ['PID', 'paragraph'])
        #q_df = read_df_from_tsv(question_files[i], ['QID', 'query'])
        #t_df = read_df_from_tsv(triple_files[i], ['QID', 'PID+', 'PID-'])
        
        # read files into dataframe
        paragraphs = Passages(passage_files[i])
        queries = Queries(question_files[i])
        qpp_triples = Triples(triple_files[i])

        # order is the same i googled it
        row_pid_mapping = dict(enumerate(paragraphs.keys()))
        col_qid_mapping = dict(enumerate(queries.keys()))

        # create tf-idf vectorizer and matrix
        vectorizer, X = tfIDFCreatorFromArr(paragraphs.values())

        Q = vectorizer.transform(queries.values()).T
        
        Q = Q.toarray()
        X = X.toarray()
        M = X@Q

        metrics = Metrics(M, row_pid_mapping, col_qid_mapping, dataset_name=FOLDERS[i], qpp_triples=qpp_triples)
        metrics.isInBestK(5)


def main():
    tfIDF(K, FOLDERS)

if __name__ == "__main__":
    main()
