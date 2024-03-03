from sklearn.decomposition import PCA

import sys
import numpy as np

import time
import numba
import multiprocessing

binning_methods = {"equi-depth", "equi-width", "information-gain", "kmeans", "quantile"}

class SpartanPCA:
    def __init__(self,
                 alphabet_size=4,
                 window_size=0,
                 word_length=8,
                 binning_method='equi-depth',
                 remove_repeat_words = False,
                 n_jobs=-1):
        self.alphabet_size=alphabet_size
        self.window_size = window_size
        self.word_length = word_length
        self.binning_method = binning_method
        self.remove_repeat_words = remove_repeat_words
        self.n_jobs = n_jobs

    #     if self.n_jobs < 1 or self.n_jobs > multiprocessing.cpu_count():
    #         n_jobs = multiprocessing.cpu_count()
    #     else:
    #         n_jobs = self.n_jobs

    #     from numba import set_num_threads

    #     set_num_threads(n_jobs)
    def transform(self,X):

        n_instances, series_length = X.shape

        window_size = self.window_size
        if self.window_size == 0:
            window_size = series_length

        num_windows_per_inst = series_length - window_size + 1

        if self.window_size == 0 or self.window_size == X.shape[1]:
            X_transform = self.pca.transform(X)

            kept_components = X_transform[:,0:self.word_length]
            words = self.generate_words(kept_components,self.breakpoints)
        else:
            split = X[:,np.arange(window_size)[None,:] + np.arange(num_windows_per_inst)[:,None]]

            flat_split = np.reshape(split,(-1,split.shape[2]))
            split_transorm = self.pca.transform(flat_split)

            breakpoints = self.binning(split_transorm)

            self.breakpoints = breakpoints
            flat_words = self.generate_words(split_transorm,breakpoints)

            words = np.reshape(flat_words,(n_instances,num_windows_per_inst,self.word_length)) 
            self.pred_histogram = self.bag_to_hist(self.create_bags(words))


        return words
    def fit_transform(self,X,y=None):
        self.pca = PCA(n_components=self.word_length,svd_solver='full')
        # self.pca = PCA(n_components=self.word_length)

        self._y = y
        self._X = X

        n_instances, series_length = X.shape

        window_size = self.window_size
        if self.window_size == 0:
            window_size = series_length

        num_windows_per_inst = series_length - window_size + 1
        split = X[:,np.arange(window_size)[None,:] + np.arange(num_windows_per_inst)[:,None]]

        if self.window_size != 0 and self.window_size != series_length:
            
            print(split.shape)
            flat_split = np.reshape(split,(-1,split.shape[2]))
            split_transorm = self.pca.fit_transform(flat_split)
            # split_transorm = np.reshape(split_transorm,flat_split.shape)
            print(split_transorm.shape)
            

            breakpoints = self.binning(split_transorm)

            self.breakpoints = breakpoints
            flat_words = self.generate_words(split_transorm,breakpoints)

            words = np.reshape(flat_words,(n_instances,num_windows_per_inst,self.word_length))
            self.train_histogram = self.bag_to_hist(self.create_bags(words))           
        else: 

            X_transform = self.pca.fit_transform(X)
            self.evcr = self.pca.explained_variance_ratio_

            kept_components = X_transform[:,0:self.word_length]

            self.breakpoints = self.binning(kept_components)

            words = self.generate_words(kept_components,self.breakpoints)

        return words

    def create_bags(self,wordslists):
        n_instances,n_words_per_inst,word_length = wordslists.shape

        remove_repeat_words = self.remove_repeat_words
        wordslists = wordslists.astype(np.int32)
        bags = []
        last_word = None
        for i in range(n_instances):
            bag = {}
            wordlist = wordslists[i]
            for j in range(n_words_per_inst):
                word = wordlist[j]
                # print(word)
                text = ''.join(map(str,word))
                if (not remove_repeat_words) or (text != last_word):
                    bag[text] = bag.get(text, 0) + 1 

                last_word = text
            bags.append(bag)

        return bags

    def bag_to_hist(self,bags):
        n_instances = len(bags)

        word_length = self.word_length

        possible_words = self.alphabet_size ** word_length
        word_to_num = [np.base_repr(i,base=self.alphabet_size) for i in range(possible_words)]

        word_to_num = ['0'*(word_length - len(word)) + word for word in word_to_num]
        all_win_words = np.zeros((n_instances,possible_words))

        # print(word_to_num)

        for j in range(n_instances):
            bag = bags[j]
            # print(bag)

            for key in bag.keys():
                v = bag[key]
                # print(type(key))
                # print(key)

                # n = np.asarray(word_to_num[word_to_num is key]).nonzero()[0]
                n = word_to_num.index(key)
                # print(n)
                # print(np.asarray(word_to_num == key).nonzero())
                all_win_words[j,n] = v
        return all_win_words

    def generate_words(self,pca,breakpoints):
        words = np.zeros((pca.shape[0],self.word_length),dtype=np.float64)
        for a in range(pca.shape[0]):
            for i in range(self.word_length):
                words[a,i] = np.digitize(pca[a,i],breakpoints[i],right=True)
        
        return words

    def binning(self,pca):
        if self.binning_method == 'equi-depth' or self.binning_method == 'equi-width':
            breakpoints = self._mcb(pca)
        return breakpoints
    def _mcb(self,pca):
        breakpoints = np.zeros((self.word_length,self.alphabet_size))
        mindist_breakpoints = np.zeros((self.word_length,self.alphabet_size+1))

        pca = np.round(pca,4)
        for letter in range(self.word_length):
            column = np.sort(pca[:,letter])
            column_min = np.min(column)
            column_max = np.max(column)
            mindist_breakpoints[letter,0] = sys.float_info.min
            bin_index = 0

            #use equi-depth binning
            if self.binning_method == "equi-depth":
                target_bin_depth = len(pca) / self.alphabet_size

                for bp in range(self.alphabet_size - 1):
                    bin_index += target_bin_depth 
                    breakpoints[letter,bp] = column[int(bin_index)]
                    mindist_breakpoints[letter,bp+1] = column[int(bin_index)]

            
            #equi-width binning aka equi-frequency binning
            elif self.binning_method == "equi-width":
                target_bin_width = (column[-1] - column[0]) / self.alphabet_size

                for bp in range(self.alphabet_size - 1):
                    breakpoints[letter,bp] = (bp + 1) * target_bin_width + column[0]
                    mindist_breakpoints[letter,bp+1] = (bp + 1) * target_bin_width + column[0]
            
        breakpoints[:, self.alphabet_size - 1] = sys.float_info.max
        mindist_breakpoints[:,self.alphabet_size] = sys.float_info.max

        self.mindist_breakpoints = mindist_breakpoints
        return breakpoints