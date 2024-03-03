from sax.sax_vfd import SAXVFD
from distance import mindist,matching_distance,hist_euclidean_dist,pairwise_distance,pairwise_histogram_distance
from distance_vectorized import symbol_vectorized, hamming_vectorized, euclidean_vectorized

import sys
import numpy as np
import scipy.stats

class SAXVFDDictionaryClassifier():
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
        # self.sax = SAXVFD(
        #     word_length=self.word_length,
        #     alphabet_size=self.alphabet_size,
        #     window_size=self.window_size,
        #     remove_repeat_words=self.remove_repeat_words,
        #     save_words=self.save_words
        # )
        self.word_length = min(self.word_length, X.shape[-1], X.shape[0])
        
        self.train_data = X
        # self.train_bags = self.sax.transform(X)

        # self.train_hist = self.sax.histogram
        # self.train_words_bps = self.sax.bp_words

        # self.series_length = X.shape[1]
        # self.breakpoints = self.sax.breakpoints
        # self.train_words_bps = self.sax.bp_words
        # # self.train_dist_mat = self.pairwise_distance(self.train_words_bps)
        self._y = y

        # return self 


    def _opt_feat_vec(self):

        train_num = len(self.train_data)
        test_num = len(self.test_data)

        if train_num >= 50:
            N = min(int(0.1*train_num), test_num)
        else:
            N = min(int(0.5*train_num), test_num)

        select_train_idx = np.random.choice(train_num, size=N, replace=False)
        select_test_idx  = np.random.choice(test_num,  size=N, replace=False)
        select_train = self.train_data[select_train_idx]
        select_test  = self.test_data[select_test_idx]

        self.sax_train = SAXVFD(
            # word_length=self.word_length,
            word_length=self.word_length,
            alphabet_size=self.alphabet_size,
            window_size=self.window_size,
            remove_repeat_words=self.remove_repeat_words,
            save_words=self.save_words
        )

        self.sax_train.transform(select_train)
        select_train_bp = self.sax_train.bp_words
        
        print(select_train_bp.shape)
        print(select_train_bp[0])

        self.sax_test = SAXVFD(
            # word_length=self.word_length,
            word_length=self.word_length,
            alphabet_size=self.alphabet_size,
            window_size=self.window_size,
            remove_repeat_words=self.remove_repeat_words,
            save_words=self.save_words
        )

        self.sax_test.transform(select_test)
        select_test_bp = self.sax_test.bp_words

        # print(select_train_bp[0].shape, select_test_bp[0].shape)

        # print(np.array(self.sax_train.breakpoints).dtype)
        
        sax_param = (self.sax_train.breakpoints, select_train.shape[1], 3) # bkps, n, w
        opt_feat_name = self._tlb(select_train_bp, select_test_bp, select_train, select_test, sax_param)
        
        return opt_feat_name

    def _tlb(self, word1, word2, series1, series2, sax_param):

        n_instances, _, n_feat = word1.shape 
        word1 = word1.reshape(n_instances, 1, -1, 18)
        word2 = word2.reshape(n_instances, 1, -1, 18)

        euc_dist = np.sqrt(np.sum((scipy.stats.zscore(series1,axis=1) - scipy.stats.zscore(series2,axis=1))**2, axis=(1)))
        
        feat_name = np.array(['max', 'min', 'mean', 'median', 'var', 
                              'skew', 'slope', 'range', 'IQR', 'entropy', 
                              'mean_sec_deri_central', 'apEn', 'mean_abs_ch', 'sampEn',
                              'abs_sum_of_ch', 'kurtosis','abs_en', 'bEn'])

        tlb_val = []
        for i in range(len(feat_name)):

            min_dist = pairwise_distance(word1[:,:,:,i],word2[:,:,:,i],symmetric=False,metric = 'mindist', sax_param=sax_param)
            min_dist = np.diagonal(min_dist)

            tlb_val.append(np.mean(min_dist/euc_dist+1e-8))


            # print("Feature Name: \t", feat_name[i])
            # print("symbols: \t", word1[0,:,:,i],word2[0,:,:,i])
            # print("min_dist: \t", np.around(min_dist, decimals=2), round(np.mean(min_dist),2))
            # print("euc_dist: \t", np.around(euc_dist, decimals=2), round(np.mean(euc_dist),2))
            # print("tlb: \t", np.around(min_dist/euc_dist, decimals=2), round(np.mean(min_dist/euc_dist), 2))
            
        ind = np.argsort(tlb_val)
        
        return feat_name[ind[-4:]]

    def predict(self,X):

        self.test_data = X

        opt_feat_name = self._opt_feat_vec()

        print(opt_feat_name)
        self.sax = SAXVFD(
            word_length=self.word_length,
            alphabet_size=self.alphabet_size,
            window_size=self.window_size,
            remove_repeat_words=self.remove_repeat_words,
            save_words=self.save_words,
            feat_list=opt_feat_name,
            return_feat_freq=True
        )

        self.train_bags = self.sax.transform(self.train_data)
        self.train_hist = self.sax.histogram
        self.train_words_bps = self.sax.bp_words

        self.sax.return_feat_freq=False
        self.predict_bags = self.sax.transform(self.test_data)
        self.predict_hist = self.sax.histogram
        self.predict_words_bps = self.sax.bp_words

        print(self.train_words_bps.shape)
        print(self.predict_words_bps.shape)
        # print(self.predict_words_bps[:5])

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
            
            pred_X = np.squeeze(self.predict_words_bps,axis=1)
            train_X = np.squeeze(self.train_words_bps,axis=1)

            dist_mat = self.sax.distance(train_X, pred_X, self.word_length, X.shape[1], k=4)
            ed = euclidean_vectorized(scipy.stats.zscore(self.test_data, axis=-1), scipy.stats.zscore(self.train_data, axis=-1))

            print(dist_mat/ed)

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