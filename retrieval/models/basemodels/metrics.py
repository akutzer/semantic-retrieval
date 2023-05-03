from retrieval.data.triples import Triples 
import numpy as np

class Metrics:
    def __init__(self, M, row_pid_mapping, col_qid_mapping, dataset_name, qpp_triples:Triples=None, pqq_triples:Triples=None):
        self.M = np.array(M)
        self.row_pid_mapping = row_pid_mapping
        self.col_qid_mapping = col_qid_mapping
        self.qpp_triples = qpp_triples
        self.pqq_triples = pqq_triples
        self.dataset_name = dataset_name
        self.pid_row_mapping = {v: k for k, v in row_pid_mapping.items()}
        self.qid_col_mapping = {v: k for k, v in col_qid_mapping.items()}


    def __bestMatches__(self, passages=True):
        if passages:
            return np.argsort(np.argsort(-self.M, axis=0), axis=0)
        else:
            return np.argsort(np.argsort(-self.M, axis=1), axis=1)
    
    def positionOfPositive(self, pqq=False):
        pos = []
        if pqq:
            row_ind = 0
            col_ind = 1
        else: # qpp 
            row_ind = 1
            col_ind = 0

        
        best_matches_M = self.__bestMatches__(passages=True)
            
        for triple in self.qpp_triples:
            row = self.pid_row_mapping[triple[row_ind]]
            col = self.qid_col_mapping[triple[col_ind]] 
            pos.append(best_matches_M[row,col])

        return pos
    


        
    def isInBestK(self,k, qpp=True):
        pos_arr = self.positionOfPositive()
        hits = np.count_nonzero(np.array(pos_arr) < k)
        print(f"In the dataset {self.dataset_name} tf_idf successfully found the correct passage in the top {k} matches {100.0*hits/len(pos_arr)}% of the time")





