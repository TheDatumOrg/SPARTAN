from sfa.sfa_fast import SFAFast

import math
import numpy as np

import multiprocessing
import time
import sys

from distance import mindist,matching_distance,hist_euclidean_dist,pairwise_distance,pairwise_histogram_distance
from distance_vectorized import symbol_vectorized,hamming_vectorized,sax_mindist,mindist_minmax,euclidean_vectorized,boss_vectorized,cosine_similarity_vectorized,kl_divergence
from numba import prange,njit

class SFADictionaryClassifier:
    def __init__(
        self,
        word_length=8,
        alphabet_size=4,
        window_size=12,
        norm=False,
        binning_method="equi-depth",
        anova=False,
        variance=False,
        bigrams=False,
        skip_grams=False,
        remove_repeat_words=False,
        lower_bounding=True,
        save_words=False,
        feature_selection="none",
        max_feature_count=256,
        p_threshold=0.05,
        random_state=None,
        return_sparse=True,
        return_pandas_data_series=False,
        n_jobs=-1,
        metric = 'mindist'
    ):
        self.words = []
        self.breakpoints = []

        # we cannot select more than window_size many letters in a word
        self.word_length = word_length

        self.alphabet_size = alphabet_size
        self.window_size = window_size
        self.norm = norm
        self.lower_bounding = lower_bounding
        

        self.remove_repeat_words = remove_repeat_words

        self.save_words = save_words

        self.binning_method = binning_method
        self.anova = anova
        self.variance = variance

        self.bigrams = bigrams
        self.skip_grams = skip_grams
        self.n_jobs = n_jobs

        self.n_instances = 0
        self.series_length = 0
        self.letter_bits = 0

        # Feature selection part
        self.feature_selection = feature_selection
        self.max_feature_count = max_feature_count
        self.feature_count = 0
        self.relevant_features = None

        # feature selection is applied based on the chi-squared test.
        self.p_threshold = p_threshold

        self.return_sparse = return_sparse
        self.return_pandas_data_series = return_pandas_data_series

        self.random_state = random_state

        if self.n_jobs < 1 or self.n_jobs > multiprocessing.cpu_count():
            n_jobs = multiprocessing.cpu_count()
        else:
            n_jobs = self.n_jobs

        self.metric = metric

        from numba import set_num_threads

        set_num_threads(n_jobs)

    def fit(self,X,y=None):
        self._y = y

        # remember class labels
        self.classes_ = np.unique(y)
        self.n_classes_ = self.classes_.shape[0]
        self._class_dictionary = {}

        if self.window_size ==0:
            window_size = X.shape[1]
        else:
            window_size = self.window_size
        self.inverse_sqrt_win_size = (
            1.0 / math.sqrt(window_size) if not self.lower_bounding else 1.0
        )

        for index, class_val in enumerate(self.classes_):
            self._class_dictionary[class_val] = index

        self.word_length = min(self.word_length, X.shape[-1], X.shape[0])

        self.sfa = SFAFast(
            window_size=self.window_size,
            word_length=self.word_length,
            alphabet_size=self.alphabet_size,
            norm=self.norm,
            binning_method=self.binning_method,
            anova=self.anova,
            variance=self.variance,
            bigrams=self.bigrams,
            skip_grams=self.skip_grams,
            remove_repeat_words=self.remove_repeat_words,
            lower_bounding=self.lower_bounding,
            save_words=self.save_words,
            feature_selection=self.feature_selection,
            max_feature_count=self.max_feature_count,
            p_threshold=self.p_threshold,
            random_state=self.random_state,
            return_sparse=self.return_sparse,
            return_pandas_data_series=self.return_pandas_data_series,
            n_jobs=self.n_jobs
        )

        
        X_transform = self.sfa.fit_transform(X,y)

        self.breakpoints = self.sfa.breakpoints
        X_words = self.sfa.words

        X_words_indices = self.words_to_indices(X_words)


        self.train_words = X_words
        self.train_word_indices = X_words_indices
        self.train_hist = X_transform.toarray()
        # print(self.train_hist[1])

        if self.n_jobs < 1 or self.n_jobs > multiprocessing.cpu_count():
            n_jobs = multiprocessing.cpu_count()
        else:
            n_jobs = self.n_jobs

        # super raises numba import exception if not available
        # so now we know we can use numba

        from numba import set_num_threads

        set_num_threads(n_jobs)

        return self

    def fit_transform(self,X,y=None):
        self._y = y

        self.word_length = min(self.word_length, X.shape[-1], X.shape[0])

        # remember class labels
        self.classes_ = np.unique(y)
        self.n_classes_ = self.classes_.shape[0]
        self._class_dictionary = {}

        for index, class_val in enumerate(self.classes_):
            self._class_dictionary[class_val] = index

        self.sfa = SFAFast(
            window_size=self.window_size,
            word_length=self.word_length,
            alphabet_size=self.alphabet_size,
            norm=self.norm,
            binning_method=self.binning_method,
            anova=self.anova,
            variance=self.variance,
            bigrams=self.bigrams,
            skip_grams=self.skip_grams,
            remove_repeat_words=self.remove_repeat_words,
            lower_bounding=self.lower_bounding,
            save_words=self.save_words,
            feature_selection=self.feature_selection,
            max_feature_count=self.max_feature_count,
            p_threshold=self.p_threshold,
            random_state=self.random_state,
            return_sparse=self.return_sparse,
            return_pandas_data_series=self.return_pandas_data_series,
            n_jobs=self.n_jobs
        )

        
        X_transform = self.sfa.fit_transform(X,y)

        self.breakpoints = self.sfa.breakpoints
        X_words = self.sfa.words
        X_words_indices = self.words_to_indices(X_words)
        # print(X_words_indices.shape)

        self.train_words = X_words
        self.train_word_indices = X_words_indices
        self.train_hist = X_transform

        return X_transform

    def predict(self,X):
        X_transform = self.sfa.transform(X)

        self.predict_hist = X_transform.toarray()

        self.pred_words = self.sfa.words
        self.pred_word_indices = self.words_to_indices(self.pred_words)

        if self.metric in ['editdistance']:

            if self.metric == 'matching':
                dist_mat = hamming_vectorized(self.predict_words_bps[:,0,:],self.train_words_bps[:,0,:])
            else:    
                dist_mat = pairwise_distance(self.predict_words_bps,self.train_words_bps,symmetric=False,metric = self.metric)
            # dist_mat = pairwise_distance(self.pred_word_indices,self.train_word_indices,symmetric=False,metric = self.metric)
        elif self.metric in ['hist_euclidean']:
            # dist_mat = pairwise_histogram_distance(self.predict_hist,self.train_hist,symmetric=False,metric=self.metric)
            dist_mat = euclidean_vectorized(self.predict_hist,self.train_hist)
        elif self.metric in ['cosine_similarity']:
            dist_mat = cosine_similarity_vectorized(self.predict_hist,self.train_hist)
        elif self.metric in ['boss']:
            dist_mat = boss_vectorized(self.predict_hist,self.train_hist)
        elif self.metric in ['kl']:
            dist_mat = kl_divergence(self.predict_hist,self.train_hist)
        elif self.metric in ['symbol']:
            pred_X = np.squeeze(self.pred_word_indices,axis=1)
            train_X = np.squeeze(self.train_word_indices,axis=1)

            dist_mat = symbol_vectorized(pred_X,train_X)
        elif self.metric in ['hamming', 'matching']:
            pred_X = np.squeeze(self.pred_word_indices,axis=1)
            train_X = np.squeeze(self.train_word_indices,axis=1)

            dist_mat = hamming_vectorized(pred_X,train_X)
        elif self.metric in ['sax_mindist']:
            pred_X = np.squeeze(self.pred_word_indices,axis=1)
            train_X = np.squeeze(self.train_word_indices,axis=1)

            breakpoints = self.sfa.mindist_breakpoints
            print(breakpoints)

            dist_mat = sax_mindist(pred_X,train_X,breakpoints)
        
        elif self.metric in ['symbol']:
            pred_X = np.squeeze(self.pred_word_indices,axis=1)
            train_X = np.squeeze(self.train_word_indices,axis=1)

            dist_mat = symbol_vectorized(pred_X,train_X)
        elif self.metric in ['euclidean']:
            pred_X = np.squeeze(self.pred_word_indices,axis=1)
            train_X = np.squeeze(self.train_word_indices,axis=1)

            dist_mat = euclidean_vectorized(pred_X,train_X)
        elif self.metric == 'matching_symbol':
            pred_X = np.squeeze(self.pred_word_indices,axis=1)
            train_X = np.squeeze(self.train_word_indices,axis=1)

            dist_mat_1 = hamming_vectorized(pred_X,train_X)
            dist_mat_2 = symbol_vectorized(pred_X,train_X)
            dist_mat = None
        self.dist_mat = dist_mat
        if self.metric == 'matching_symbol':

            print(self.metric)
            self.pred_dist_mat1 = dist_mat_1
            self.pred_dist_mat2 = dist_mat_2
            ind_1 = np.argmin(dist_mat_1,axis=1)
            ind_2 = np.argmin(dist_mat_2,axis=1)
            
            ind_1 = ind_1.T
            ind_2 = ind_2.T
            pred_1 = self._y[ind_1]
            pred_2 = self._y[ind_2]

            return (pred_1, pred_2)
        else:
            self.pred_dist_mat = dist_mat
            ind = np.argmin(dist_mat,axis=1)

            # if self.metric =='matching':
            #     ind = np.argmax(dist_mat,axis=1)

            ind = ind.T
            pred = self._y[ind]

            return pred


        # self.pred_dist_mat = dist_mat

        # ind = np.argmin(dist_mat,axis=1)

        # # if self.metric =='matching':
        # #     ind = np.argmax(dist_mat,axis=1)

        # ind = ind.T
        # pred = self._y[ind]

        # return pred

    # def pairwise_distance(self,X_wordlists,Y_wordlists=None,symmetric = True):
    #     if Y_wordlists is None:
    #         Y_wordlists=X_wordlists
    #     X_samples = X_wordlists.shape[0]
    #     Y_samples = Y_wordlists.shape[0]

    #     pairwise_matrix = np.zeros((X_samples,Y_samples))

    #     for i in range(X_samples):
    #         curr_x = X_wordlists[i]
    #         for j in range(Y_samples):
    #             if symmetric and j < i:
    #                 pairwise_matrix[i,j] = pairwise_matrix[j,i]
    #             else:
    #                 if self.metric == 'mindist':
    #                     x = curr_x
    #                     y = Y_wordlists[j]

    #                     breakpoints= self.breakpoints

    #                     pairwise_matrix[i,j] = self.sfa_mindist(x,y,breakpoints,self.series_length,self.word_length)
    #                 elif self.metric == 'matching':
    #                     x = curr_x
    #                     y = Y_wordlists[j]

    #                     pairwise_matrix[i,j] = matching_distance(x,y)

    #                 elif self.metric == 'histogram':
    #                     pairwise_matrix[i,j] = hist_euclidean_dist(self.predict_hist[i],self.train_hist[j])

    #     return pairwise_matrix


    def sfa_cell(self,r,c,breakpoints):
        partial_dist = 0
        breakpoints_i = breakpoints
        if np.abs(r-c) <= 1:
            partial_dist = 0
        else:
            partial_dist = breakpoints_i[int(max(r,c) - 1)] - breakpoints_i[int(min(r,c))]
        return partial_dist

    def sfa_mindist(self,word_list1,word_list2,breakpoints,n,w):
        sum = 0
        for i , (word1, word2) in enumerate(zip(word_list1,word_list2)):
            partial_dist=0
            for i,(q,c) in enumerate(zip(word1,word2)):
                check = (self.sfa_cell(q,c,breakpoints[i]))**2
                if check == np.inf:
                    print((q,c))
                partial_dist = partial_dist + (self.sfa_cell(q,c,breakpoints[i]))**2
            sum = sum + (np.sqrt(n/w)*np.sqrt(partial_dist))

        return sum
    def words_to_indices(self,X_wordslist):
        n_instances = len(X_wordslist)
        n_words = len(X_wordslist[0])
        word_bits = self.sfa.word_bits
        letter_bits = self.sfa.letter_bits

        letter = 2**letter_bits
        
        word_indices = np.zeros((n_instances,n_words,self.word_length))
        for i, instance in enumerate(X_wordslist):
            for j, word in enumerate(instance):
                word = X_wordslist[i,j]

                for k in range(self.word_length):
                    ind = word % letter
                    word_indices[i,j,k] = ind
                    word = word // letter

        return word_indices
        