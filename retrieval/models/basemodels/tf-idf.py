import pandas as pd
import os 
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import swifter


'''
#   Evaluates test data by using tf-idf as a base model. Measurement: is one of the best -k- predicted answers the correct one?
#   Calculates the dot product between every paragraph tf-idf vector and the query tf-idf vector and chooses the largest -k- scores. 
#   (X@q_vec.T, where X is matrix of all paragraph vectors and q_vec is the query vector)
'''

FOLDERS = ['../../../data/fandom-qa/harry_potter_qa'
           #,'../../data/fandom-qa/witcher_qa_2'
           ]

# number of paragraphs that are acceptable for a question
K = 5

def read_df_from_tsv(file, column_names):
    return pd.read_csv(file, sep='\t', header=None).rename(columns=dict(enumerate(column_names)))


def getKBestMatchingIndicesForQuestion(question, vectorizer, X, k):
    q_vec = vectorizer.transform([question])
    dot_products = (X@q_vec.T).toarray().flatten()
    return np.argsort(dot_products)[::-1][:k]


def mapToInternalID(df, arr, type='PID'):
    if type == 'PID':
        return df['PID'].iloc[arr].to_numpy()
    elif type == 'QID':
        return df['QID'].iloc[arr].to_numpy()


def tfIDFCreatorFromArr(arr, min_df=10):
    vectorizer = TfidfVectorizer(strip_accents='unicode', min_df=min_df)
    X = vectorizer.fit_transform(arr)
    return vectorizer, X


def isPairInTriples(qid,pid, t_df):
    return t_df.loc[(t_df['QID'] == qid) & (t_df['PID+'] == pid)].any().any()


def printStatistics(result_dfs, k):
    for i, q_df in enumerate(result_dfs):
        hits = len(q_df.loc[q_df['best_k_match'] >= 0])
        total = len(q_df)
        name_dataset = FOLDERS[i].split('/')[-1]
        print(f"In the dataset {name_dataset} tf_idf successfully found the correct passage in the top {k} matches {100.0*hits/total}% of the time")

def getResultDfs(k, FOLDERS):
    passage_files = [x + '/' + y for x in FOLDERS for y in os.listdir(x) if 'passages' in y and '.tsv' in y]
    question_files = [x + '/' + y for x in FOLDERS for y in os.listdir(x) if 'queries' in y and '.tsv' in y]
    triple_files = [x + '/' + y for x in FOLDERS for y in os.listdir(x) if 'triples' in y and '.tsv' in y]

    result_dfs = []
    for i in range(len(FOLDERS)):
        # read files into dataframe
        p_df = read_df_from_tsv(passage_files[i], ['PID', 'paragraph'])
        q_df = read_df_from_tsv(question_files[i], ['QID', 'query'])
        t_df = read_df_from_tsv(triple_files[i], ['QID', 'PID+', 'PID-'])

        # create tf-idf vectorizer and matrix
        vectorizer, X = tfIDFCreatorFromArr(p_df['paragraph'])
        # q_df = q_df.iloc[:100,:]

        # get best k paragraph matches for question
        q_df["best_k_PID"] = q_df['query'].swifter.apply(lambda q: mapToInternalID(p_df, getKBestMatchingIndicesForQuestion(q ,vectorizer, X, k), 'PID'))
        # q_df['best_k_match'] = q_df.apply(lambda x : any([isPairInTriples(x[0], y, t_df) for y in x[2]]), axis=1)

        # get position of the correct question in the best k paragraphs else -1
        q_df['best_k_match'] = q_df.swifter.apply(lambda x : (lambda arr : -1 if arr[0].size == 0 else arr[0][0]) (np.array([isPairInTriples(x[0], y, t_df) for y in x[2]]).nonzero()) , axis=1)
        
        result_dfs.append(q_df)
    return result_dfs

def main():
    printStatistics(getResultDfs(K, FOLDERS), K)

if __name__ == "__main__":
    main()
