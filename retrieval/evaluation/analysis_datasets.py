import os
import pandas as pd

SEARCH_PATH = "../../data/"


# key is column , k is number of most occurring question words
def getDistributionQuestionWords(df, key, k):
    c = df[key].apply(lambda x : x.split(' ')[0]).value_counts(normalize=True)

    i = 0
    for x,y in c.items():
        if i >= k:
            break
        i = i + 1
        print(f'prob of *{x}*: {y}')


def printStatistics(path):
    df_passages = pd.read_csv(path + '/passages.tsv', sep ='\t')
    df_queries = pd.read_csv(path + '/queries.tsv', sep ='\t')
    df_triple = pd.read_csv(path + '/triples.tsv', sep ='\t')

    print('--------------------------------------------')
    print(f'statistics for folder {path}:')
    print(f'# triples: {len(df_triple)}')
    print(f'# passages: {len(df_passages)}')

    median_passage_words = df_passages['passage'].apply(lambda x : len(x.split(' '))).median()
    print(f'median passage words: {median_passage_words}')
    
    print(f'# queries: {len(df_queries)}')

    median_query_words = df_queries['query'].apply(lambda x : len(x.split(' '))).median()
    print(f'median query words: {median_query_words}')


    getDistributionQuestionWords(df_queries, 'query', 5)


if __name__ == "__main__":
    for root, dirs, files in os.walk(SEARCH_PATH):
        
        # is a target folder
        if 'passages.tsv' in files and ('/all' in root or 'MSMARCO' in root):
            printStatistics(root)