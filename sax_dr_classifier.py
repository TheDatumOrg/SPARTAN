from sax.sax_dr import SAXDR
from distance import mindist,matching_distance,hist_euclidean_dist,pairwise_distance,pairwise_histogram_distance
from distance_vectorized import symbol_vectorized, hamming_vectorized, euclidean_vectorized
import scipy.stats

import sys
import numpy as np

class SAXDRDictionaryClassifier():
    def __init__(self,
        word_length=8,
        alphabet_size=4,
        window_size=12,
        remove_repeat_words=False,
        save_words=False,
        metric = 'mindist'):
        
        self.word_length = word_length
        self.alphabet_size = alphabet_size
        self.window_size = window_size
        self.remove_repeat_words = remove_repeat_words
        self.save_words = save_words
        self.metric = metric

    def fit(self,X,y=None):

        self.word_length = min(self.word_length, X.shape[-1], X.shape[0])
        
        self.sax = SAXDR(
            word_length=self.word_length,
            alphabet_size=self.alphabet_size,
            window_size=self.window_size,
            remove_repeat_words=self.remove_repeat_words,
            save_words=self.save_words
        )

        self.train_data = X
        self.train_bags = self.sax.transform(X)

        self.train_hist = self.sax.histogram
        self.train_words_bps = self.sax.bp_words

        self.series_length = X.shape[1]
        self.breakpoints = self.sax.breakpoints
        self.train_words_bps = self.sax.bp_words
        # self.train_dist_mat = self.pairwise_distance(self.train_words_bps)
        self._y = y

        return self 
    def predict(self,X):
        self.test_data = X
        self.predict_bags = self.sax.transform(X)
        self.predict_hist = self.sax.histogram
        self.predict_words_bps = self.sax.bp_words

        print(self.train_words_bps.shape)
        print(self.predict_words_bps.shape)
        print(self.predict_words_bps[:5])

        if self.metric in ['matching','editdistance']:
            if self.metric == 'matching':
                dist_mat = hamming_vectorized(self.predict_words_bps[:,0,:],self.train_words_bps[:,0,:])
            else:    
                dist_mat = pairwise_distance(self.predict_words_bps,self.train_words_bps,symmetric=False,metric = self.metric)

        elif self.metric in ['hist_euclidean','boss']:
            dist_mat = pairwise_histogram_distance(self.predict_hist,self.train_hist,symmetric=False,metric=self.metric)
        
        elif self.metric in ['symbol']:
            pred_X = np.squeeze(self.predict_words_bps,axis=1)
            train_X = np.squeeze(self.train_words_bps,axis=1)

            dist_mat = symbol_vectorized(pred_X,train_X)

        elif self.metric == 'matching_symbol':
            pred_X = np.squeeze(self.predict_words_bps,axis=1)
            train_X = np.squeeze(self.train_words_bps,axis=1)

            dist_mat_1 = hamming_vectorized(pred_X,train_X)
            dist_mat_2 = symbol_vectorized(pred_X,train_X)


        elif self.metric == 'own_dist':

            dist_mat = self.sax.distance(self.train_words_bps, self.predict_words_bps, int(self.word_length/2), X.shape[1])

            ed = euclidean_vectorized(scipy.stats.zscore(self.test_data, axis=-1), scipy.stats.zscore(self.train_data, axis=-1))

            ed += 1e-8

            # err = (self.test_data[3] - self.train_data[0])
            print("dst: ", dist_mat)
            print("ed: ", ed)
            # print(self.predict_words_bps[3], self.train_words_bps[0])
            
            # print(self.train_data[0], self.test_data[3])
            print((dist_mat/ed)[:5, :5])

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
    def transform(self,X):
        return self.sax.transform(X)

    def fit_transform(self,X,y=None):
        self.fit(X,y)
        X_transform = self.transform(X)

        return X_transform
    
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
    #                     x = curr_x + 1
    #                     y = Y_wordlists[j] + 1

    #                     breakpoints = [sys.float_info.min] + self.breakpoints

    #                     pairwise_matrix[i,j] = mindist(x,y,breakpoints,self.series_length,self.word_length)
    #                 elif self.metric == 'matching':
    #                     x = curr_x
    #                     y = Y_wordlists[j]
    #                     pairwise_matrix[i,j] = matching_distance(x,y)

    #                 elif self.metric == 'hist':
    #                     x = curr_x
    #                     y = Y_wordlists[j]
    #                     pairwise_matrix[i,j] = hist_euclidean_dist(self.predict_hist[i],self.train_hist[j])
    #     return pairwise_matrix