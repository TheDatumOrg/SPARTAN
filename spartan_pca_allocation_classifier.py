from spartan.spartan_pca_bit_allocation import SpartanPCAAllocation
from distance import mindist,matching_distance,hist_euclidean_dist,pairwise_distance,pairwise_histogram_distance
from distance_vectorized import symbol_vectorized,hamming_vectorized,symbol_weighted,mindist_vectorized,mindist_weighted,l1_mindist

import numpy as np

class SpartanPCAAllocationClassifier:
    def __init__(self,
                 alphabet_size=[16,8,4,4,4,4,4,4],
                 window_size=0,
                 word_length=8,
                 binning_method='equi-depth',
                 allocation_method='doubling',
                 assignment_policy = 'direct',
                 metric = 'matching',
                 max_bit=3,
                 min_bit=1,
                 delta=0,
                 bit_budget = 16
                 ):
        self.alphabet_size = alphabet_size
        self.window_size = window_size
        self.word_length = word_length
        self.binning_method = binning_method
        self.allocation_method = allocation_method
        self.assignment_policy = assignment_policy
        self.bit_budget = bit_budget
        self.metric = metric
        self.max_bit = max_bit
        self.min_bit = min_bit
        self.delta = delta

        self.spartan = SpartanPCAAllocation(
            alphabet_size=alphabet_size,
            window_size = window_size,
            word_length = word_length,
            binning_method = binning_method,
            assignment_policy = assignment_policy,
            max_bit = max_bit,
            min_bit = min_bit,
            delta = delta,
            bit_budget=self.bit_budget
        )
    def fit(self,X,y=None):
        self._y = y

        self.train_mean = np.mean(X)
        self.train_std = np.std(X)

        train_X = (X - self.train_mean) / self.train_std


        word_length = min(self.word_length, X.shape[-1], X.shape[0])
        
        if word_length < self.word_length:
            self.word_length = word_length
            self.spartan.word_length = self.word_length

            if isinstance(self.alphabet_size, list): # direct allocation
                alpha_size = np.mean(np.log2(self.alphabet_size))
                alpha_size = int(2**alpha_size)
                self.spartan.alphabet_size = [alpha_size for i in range(self.word_length)]
                print("shrink the alphabet size: ", self.spartan.alphabet_size)


        X_transform = self.spartan.fit_transform(train_X)
        self.train_words = np.expand_dims(X_transform,axis=1)

        return self

    def predict(self,X):
        pred_X = (X - self.train_mean) / self.train_std

        self.pred_words = np.expand_dims(self.spartan.transform(pred_X),axis=1)

        if self.metric in ['editdistance','euclidean']:
            dist_mat = pairwise_distance(self.pred_words,self.train_words,symmetric=False,metric = self.metric)
        elif self.metric in ['hist_euclidean','boss']:
            dist_mat = pairwise_histogram_distance(self.predict_hist,self.train_hist,symmetric=False,metric=self.metric)
        elif self.metric in ['symbol']:
            pred_X = np.squeeze(self.pred_words,axis=1)
            train_X = np.squeeze(self.train_words,axis=1)

            dist_mat = symbol_vectorized(pred_X,train_X)
        elif self.metric in ['matching']:
            pred_X = np.squeeze(self.pred_words,axis=1)
            train_X = np.squeeze(self.train_words,axis=1)

            dist_mat = hamming_vectorized(pred_X,train_X)
        elif self.metric in ['mindist']:
            pred_X = np.squeeze(self.pred_words,axis=1)
            train_X = np.squeeze(self.train_words,axis=1)

            breakpoints = self.spartan.mindist_breakpoints

            dist_mat = mindist_vectorized(pred_X,train_X,breakpoints)
        elif self.metric in ['mindist_weighted']:
            pred_X = np.squeeze(self.pred_words,axis=1)
            train_X = np.squeeze(self.train_words,axis=1)

            breakpoints = self.spartan.mindist_breakpoints
            weights = self.spartan.evcr[:self.word_length]

            dist_mat = mindist_weighted(pred_X,train_X,breakpoints,weights)
        elif self.metric in ['l1_mindist']:
            pred_X = np.squeeze(self.pred_words,axis=1)
            train_X = np.squeeze(self.train_words,axis=1)

            breakpoints = self.spartan.mindist_breakpoints

            dist_mat = l1_mindist(pred_X,train_X,breakpoints)
        elif self.metric in ['symbol_weighted']:
            weights = self.spartan.evcr[:self.word_length]
            
            pred_X = np.squeeze(self.pred_words,axis=1)
            train_X = np.squeeze(self.train_words,axis=1)

            dist_mat = symbol_weighted(pred_X,train_X,weights)
        ind = np.argmin(dist_mat,axis=1)
        ind = ind.T
        pred = self._y[ind]

        return pred