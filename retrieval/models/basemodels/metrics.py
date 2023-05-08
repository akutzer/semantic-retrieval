from retrieval.data.triples import Triples 
import numpy as np
import torch
from torch.nn.modules.transformer import Transformer
from torchprofile import profile_macs
import time

class Metrics:
    def __init__(self):
        self.M = None
        # CPU/Execution Time
        self.wall_start_time = 0
        self.cpu_start_time = 0
        self.wall_times = []
        self.cpu_times = []

    def evaluateDatasetIntoM(self, M, row_pid_mapping, col_qid_mapping, dataset_name, qpp_triples:Triples=None, pqq_triples:Triples=None):
        self.M = np.array(M)
        self.row_pid_mapping = row_pid_mapping
        self.col_qid_mapping = col_qid_mapping
        self.qpp_triples = qpp_triples
        self.pqq_triples = pqq_triples
        self.dataset_name = dataset_name

        # best paragraphs for queries
        self.best_matches_for_query = self.bestMatches(True)

        # best queries for paragraphs
        self.best_matches_for_paragraph = self.bestMatches(False)

        # inverse of mapping
        self.pid_row_mapping = {v: k for k, v in row_pid_mapping.items()}
        self.qid_col_mapping = {v: k for k, v in col_qid_mapping.items()}


    # position where element would be if sorted
    # either columnwise (passages=True) or rowwise (passages=False)
    def bestMatches(self, passages=True):
        if passages:
            return np.argsort(np.argsort(-self.M, axis=0), axis=0)
        else:
            return np.argsort(np.argsort(-self.M, axis=1), axis=1)
     

    # for every triple returns position where tf idf ranked it (only position array)
    def positionOfPositive(self, pqq=False):
        pos = []
        if pqq:
            row_ind = 0
            col_ind = 1
            best_matches = self.best_matches_for_paragraph
        else:  # qpp
            row_ind = 1
            col_ind = 0
            best_matches = self.best_matches_for_query

        for triple in self.qpp_triples:
            row = self.pid_row_mapping[triple[row_ind]]
            col = self.qid_col_mapping[triple[col_ind]]
            pos.append(best_matches[row, col])

        return pos


    def positionOfNegative(self, pqq=False):
        pos = []
        if pqq:
            row_ind = 0
            col_ind = 2
            best_matches = self.best_matches_for_paragraph
        else:  # qpp
            row_ind = 2
            col_ind = 0
            best_matches = self.best_matches_for_query

        for triple in self.qpp_triples:
            row = self.pid_row_mapping[triple[row_ind]]
            col = self.qid_col_mapping[triple[col_ind]]
            pos.append(best_matches[row, col])

        return pos


    def isInBestK(self,k, qpp=True):
        pos_arr = self.positionOfPositive()
        hits = np.count_nonzero(np.array(pos_arr) < k)
        return 1.0 * hits / len(pos_arr)


    def meanIndexPositive(self):
        pos_arr = self.positionOfPositive()
        return np.average(pos_arr)


    def meanIndexNegative(self):
        pos_arr = self.positionOfNegative()
        return np.average(pos_arr)


    def meanReciprocalRank(self):
        # all 3 metrics are between 0 and 1
        # should be big
        mrr_pos = 1/(self.meanIndexPositive()+1)
        mrr_neg = 1 - 1/(self.meanIndexNegative()+1)
        return mrr_pos, mrr_neg


    def linearPositionScore(self):
        num_p, _ = self.M.shape
        lps_pos = (self.meanIndexPositive()-num_p)/-num_p
        lps_neg = self.meanIndexNegative()/num_p
        return lps_pos, lps_neg, (lps_pos+lps_neg)/2


    def confusionMatrix(self):
        num_p, num_q = self.M.shape
        pos_arr_pos = self.positionOfPositive()
        pos_arr_neg = self.positionOfNegative()
        true_positives = np.count_nonzero(np.array(pos_arr_pos) <= int(num_p/2))
        false_positive = num_p - true_positives
        true_negatives = np.count_nonzero(np.array(pos_arr_neg) >= int(num_p/2))
        false_negatives = num_p - true_negatives
        return [[true_positives, false_positive], [false_negatives, true_negatives]]


    def precisionAndRecall(self):
        cm = self.confusionMatrix()
        TP, FP = cm[0]
        FN, TN = cm[1]
        return TP/(TP + FP), TP/(TP + FN)


    def FBetaScore(self, beta = 1):
        # equal to F1Score for beta = 1
        precision, recall = self.precisionAndRecall()
        return (beta**2 + 1)*(precision * recall)/(beta**2 * precision + recall)


    def printTriples(self):
        for triple in self.qpp_triples:
            print(triple)

    #TODO FLOPS implmentieren
    def FLOPsPerAnswerRetrieval(self):
        embed_size = 512
        num_tokens = 30

        model = Transformer(embed_size)
        inputs = (
            torch.randn(num_tokens, 1, embed_size),
            torch.randn(num_tokens, 1, embed_size),
        )

        macs = profile_macs(model, inputs)
        print('transformer: {:.4g} G'.format(macs / 1e9))

    #TODO add start and end bevor and after function call
    def startCPUTime(self):
        self.cpu_start_time = time.process_time()


    def startWallTime(self):
        self.wall_start_time = time.time()


    def stopCPUTime(self):
        elapsed_time = time.process_time() - self.cpu_start_time
        self.cpu_times.append(elapsed_time)


    def stopWallTime(self):
        elapsed_time = time.time() - self.wall_start_time
        self.wall_times.append(elapsed_time)


    def meanWallAndCPUTimePerAnswerRetrieval(self):
        '''in miliseconds'''
        mwallt = 0
        mcput = 0
        print(self.wall_times)
        print(self.cpu_times)
        if self.wall_times:
            mwallt = np.average(self.wall_times)
        if self.cpu_times:
            mcput = np.average(self.cpu_times)
        return mwallt, mcput

    def printStatistics(self, k, beta):
        if not self.M is None:
            print("dataset name:", self.dataset_name)
            print(f"correct passage in top {k}: ", 100*self.isInBestK(k), "percent")
            print("precision/recall", self.precisionAndRecall())
            print(f"F-beta-score (beta = {beta})", self.FBetaScore(beta))
            print("mean-reciprocal-rank(pos, neg):", self.meanReciprocalRank())
            print("linear-position-score(pos, neg, combined):", self.linearPositionScore())
        else:
            print("no dataset was evaluated!")

    def printMeanWallAndCPUTime(self):
        wall_time, cpu_time = self.meanWallAndCPUTimePerAnswerRetrieval()
        if wall_time > 0:
            print("mean wall time: ", wall_time)
        else:
            print("no data about wall time!")
        if cpu_time > 0:
            print("mean cpu time: ", cpu_time)
        else:
            print("no data about cpu time!")



    






