from spartan.spartan import SPARTAN
from distance import pairwise_distance,pairwise_histogram_distance
from distance_vectorized import hamming_vectorized,symbol_vectorized,symbol_weighted,hamming_weighted,mindist_vectorized,sax_mindist,mindist_minmax,spartan_pca_mindist,euclidean_vectorized,boss_vectorized,cosine_similarity_vectorized,kl_divergence
import numpy as np

class SPARTANClassifier:
    def __init__(self,
                 alphabet_size=[16,8,4,4,4,4,4,4],
                 window_size=0,
                 word_length=8,
                 binning_method='equi-depth',
                 allocation_method='doubling',
                 assignment_policy = 'direct',
                 remove_repeat_words=False,
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
        self.remove_repeat_words = remove_repeat_words
        self.bit_budget = bit_budget
        self.metric = metric
        self.max_bit = max_bit
        self.min_bit = min_bit
        self.delta = delta

        self.spartan = SPARTAN(
            alphabet_size=alphabet_size,
            window_size = window_size,
            word_length = word_length,
            binning_method = binning_method,
            assignment_policy = assignment_policy,
            remove_repeat_words=remove_repeat_words,
            max_bit = max_bit,
            min_bit = min_bit,
            delta = delta,
            bit_budget=self.bit_budget
        )
    def fit(self,X,y=None):
        self._y = y

        self._mean = np.mean(X)
        self._std = np.std(X)

        self._X = X
        train_X = (X - self._mean) / self._std


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
        if self.window_size == 0:
            self.train_words = np.expand_dims(X_transform,axis=1)
        else:
            self.train_words = X_transform
        self.evcr = self.spartan.pca.explained_variance_ratio_
        return self

    def predict(self,X):
        pred_X = (X - self._mean) / self._std
        X_transform = self.spartan.transform(pred_X)
        if self.window_size ==0 or self.window_size == X.shape[1]:
            self.pred_words = np.expand_dims(self.spartan.transform(pred_X),axis=1)
        else:
            self.pred_words = X_transform
        # self.pred_words = self.pred_words.astype(np.uint32)
        
        if self.metric in ['editdistance']:
            dist_mat = pairwise_distance(self.pred_words,self.train_words,symmetric=False,metric = self.metric)
        elif self.metric in ['boss']:
            dist_mat =  boss_vectorized(self.spartan.pred_histogram,self.spartan.train_histogram)
        elif self.metric in ['hist_euclidean']:
            dist_mat =  euclidean_vectorized(self.spartan.pred_histogram,self.spartan.train_histogram)
        elif self.metric in ['cosine_similarity']:
            dist_mat = cosine_similarity_vectorized(self.spartan.pred_histogram,self.spartan.train_histogram)
        elif self.metric in ['kl']:
            dist_mat = kl_divergence(self.spartan.pred_histogram,self.spartan.train_histogram)
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
        elif self.metric in ['sax_mindist']:
            pred_X = np.squeeze(self.pred_words,axis=1)
            train_X = np.squeeze(self.train_words,axis=1)

            breakpoints = self.spartan.mindist_breakpoints
            # dist_mat = sax_mindist(pred_X,train_X,breakpoints)
            dist_mat = spartan_pca_mindist(pred_X,train_X,breakpoints)
        elif self.metric in ['mindist_minmax']:
            pred_X = np.squeeze(self.pred_words,axis=1)
            train_X = np.squeeze(self.train_words,axis=1)

            breakpoints = self.spartan.mindist_breakpoints
            dist_mat = mindist_minmax(pred_X,train_X,breakpoints)
        elif self.metric in ['symbol_weighted']:
            weights = self.evcr[:self.word_length]
            
            pred_X = np.squeeze(self.pred_words,axis=1)
            train_X = np.squeeze(self.train_words,axis=1)

            dist_mat = symbol_weighted(pred_X,train_X,weights)
        elif self.metric in ['euclidean']:
            pred_X = np.squeeze(self.pred_words,axis=1)
            train_X = np.squeeze(self.train_words,axis=1)

            dist_mat = euclidean_vectorized(pred_X,train_X)

        self.dist_mat = dist_mat
        ind = np.argmin(dist_mat,axis=1)
        ind = ind.T
        pred = self._y[ind]

        return pred