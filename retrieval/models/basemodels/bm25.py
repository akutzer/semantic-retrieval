import math
import numpy as np
import pandas as pd
import os
from multiprocessing import Pool, cpu_count
import swifter

FOLDERS = ['../../../data/fandom_qa/harry_potter_qa_small',
            #'../../data/fandom-qa/harry_potter_qa'
           #,'../../data/fandom-qa/witcher_qa_2'
           ]

# number of paragraphs that are acceptable for a question
K = 5

"""
All of these algorithms have been taken from the paper:
Trotmam et al, Improvements to BM25 and Language Models Examined
Here we implement all the BM25 variations mentioned. 
"""


class BM25:
    def __init__(self, corpus, tokenizer=None):
        self.corpus_size = 0
        self.avgdl = 0
        self.doc_freqs = []
        self.idf = {}
        self.doc_len = []
        self.tokenizer = tokenizer

        if tokenizer:
            corpus = self._tokenize_corpus(corpus)

        nd = self._initialize(corpus)
        self._calc_idf(nd)

    def _initialize(self, corpus):
        nd = {}  # word -> number of documents with word
        num_doc = 0
        for document in corpus:
            self.doc_len.append(len(document))
            num_doc += len(document)

            frequencies = {}
            for word in document:
                if word not in frequencies:
                    frequencies[word] = 0
                frequencies[word] += 1
            self.doc_freqs.append(frequencies)

            for word, freq in frequencies.items():
                try:
                    nd[word]+=1
                except KeyError:
                    nd[word] = 1

            self.corpus_size += 1

        self.avgdl = num_doc / self.corpus_size
        return nd

    def _tokenize_corpus(self, corpus):
        pool = Pool(cpu_count())
        tokenized_corpus = pool.map(self.tokenizer, corpus)
        return tokenized_corpus

    def _calc_idf(self, nd):
        raise NotImplementedError()

    def get_scores(self, query):
        raise NotImplementedError()

    def get_batch_scores(self, query, doc_ids):
        raise NotImplementedError()

    def get_top_n(self, query, documents, n=5):

        assert self.corpus_size == len(documents), "The documents given don't match the index corpus!"

        scores = self.get_scores(query)
        top_n = np.argsort(scores)[::-1][:n]
        return [documents[i] for i in top_n]


    def get_top_n_ind(self, query, documents, n=5):

        assert self.corpus_size == len(documents), "The documents given don't match the index corpus!"

        scores = self.get_scores(query)
        top_n = np.argsort(scores)[::-1][:n]
        return top_n


class BM25Okapi(BM25):
    def __init__(self, corpus, tokenizer=None, k1=1.5, b=0.75, epsilon=0.25):
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        super().__init__(corpus, tokenizer)

    def _calc_idf(self, nd):
        """
        Calculates frequencies of terms in documents and in corpus.
        This algorithm sets a floor on the idf values to eps * average_idf
        """
        # collect idf sum to calculate an average idf for epsilon value
        idf_sum = 0
        # collect words with negative idf to set them a special epsilon value.
        # idf can be negative if word is contained in more than half of documents
        negative_idfs = []
        for word, freq in nd.items():
            idf = math.log(self.corpus_size - freq + 0.5) - math.log(freq + 0.5)
            self.idf[word] = idf
            idf_sum += idf
            if idf < 0:
                negative_idfs.append(word)
        self.average_idf = idf_sum / len(self.idf)

        eps = self.epsilon * self.average_idf
        for word in negative_idfs:
            self.idf[word] = eps

    def get_scores(self, query):
        """
        The ATIRE BM25 variant uses an idf function which uses a log(idf) score. To prevent negative idf scores,
        this algorithm also adds a floor to the idf value of epsilon.
        See [Trotman, A., X. Jia, M. Crane, Towards an Efficient and Effective Search Engine] for more info
        :param query:
        :return:
        """
        score = np.zeros(self.corpus_size)
        doc_len = np.array(self.doc_len)
        for q in query:
            q_freq = np.array([(doc.get(q) or 0) for doc in self.doc_freqs])
            score += (self.idf.get(q) or 0) * (q_freq * (self.k1 + 1) /
                                               (q_freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)))
        return score

    def get_batch_scores(self, query, doc_ids):
        """
        Calculate bm25 scores between query and subset of all docs
        """
        assert all(di < len(self.doc_freqs) for di in doc_ids)
        score = np.zeros(len(doc_ids))
        doc_len = np.array(self.doc_len)[doc_ids]
        for q in query:
            q_freq = np.array([(self.doc_freqs[di].get(q) or 0) for di in doc_ids])
            score += (self.idf.get(q) or 0) * (q_freq * (self.k1 + 1) /
                                               (q_freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)))
        return score.tolist()

def read_df_from_tsv(file, column_names):
    return pd.read_csv(file, sep='\t', header=None).rename(columns=dict(enumerate(column_names)))

def mapToInternalID(df, arr, type='PID'):
    if type == 'PID':
        return df['PID'].iloc[arr].to_numpy()
    elif type == 'QID':
        return df['QID'].iloc[arr].to_numpy()

def isPairInTriples(qid,pid, t_df):
    return t_df.loc[(t_df['QID'] == qid) & (t_df['PID+'] == pid)].any().any()


def printStatistics(result_dfs, k):
    for i, q_df in enumerate(result_dfs):
        hits = len(q_df.loc[q_df['best_k_match'] >= 0])
        total = len(q_df)
        name_dataset = FOLDERS[i].split('/')[-1]
        print(f"In the dataset {name_dataset} bm25 successfully found the correct passage in the top {k} matches {100.0*hits/total}% of the time")


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

        # tokenize corpus
        tokenized_corpus = [doc.split(" ") for doc in p_df['paragraph']]
        bm25 = BM25Okapi(tokenized_corpus)
        q_df["best_k_PID"] = q_df['query'].swifter.apply(lambda q: mapToInternalID(p_df, bm25.get_top_n_ind(q.split(" "), p_df['paragraph'], n=k), 'PID'))
        q_df['best_k_match'] = q_df.swifter.apply(lambda x : (lambda arr : -1 if arr[0].size == 0 else arr[0][0]) (np.array([isPairInTriples(x[0], y, t_df) for y in x[2]]).nonzero()) , axis=1)
        result_dfs.append(q_df)
    return result_dfs


def main():
    printStatistics(getResultDfs(K, FOLDERS), K)
    '''
    corpus = [
        "Hello there good man!",
        "It is quite windy in London",
        "How is the weather today?"
    ]
    tokenized_corpus = [doc.split(" ") for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    print(bm25)
    query = "windy London"
    tokenized_query = query.split(" ")
    doc_scores = bm25.get_scores(tokenized_query)
    print(bm25.get_top_n_ind(tokenized_query, corpus, n=2))
    # array([0.        , 0.93729472, 0.        ])'''

if __name__ == "__main__":
    main()