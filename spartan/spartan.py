from sklearn.decomposition import PCA

import os
import sys
import numpy as np

sys.path.append("../")
from tools import dynamic_bit_allocation, dynamic_bit_allocation_update

binning_methods = {"equi-depth", "equi-width", "information-gain", "kmeans", "quantile"}

class SPARTAN:
    def __init__(self,
                 alphabet_size=[8,4,4,2],
                 window_size=0,
                 word_length=4,
                 binning_method='equi-depth',
                 remove_repeat_words = False,
                 assignment_policy = 'direct',
                 bit_budget = 16,
                 allocation=False,
                 max_bit=3,
                 min_bit=1,
                 delta=0):
        
        if isinstance(alphabet_size,int):
            self.alphabet_size = [alphabet_size]*word_length
        else:    
            self.alphabet_size = alphabet_size
        self.window_size = window_size
        self.word_length = word_length
        self.binning_method = binning_method
        self.remove_repeat_words = remove_repeat_words
        self.assignment_policy = assignment_policy
        self.bit_budget = bit_budget
        self.max_bit = max_bit
        self.min_bit = min_bit
        self.delta = delta
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
        self.pca = PCA()
        self._X = X
        self._y = y


        n_instances, series_length = X.shape
        window_size = self.window_size
        if self.window_size == 0:
            window_size = series_length

        num_windows_per_inst = series_length - window_size + 1
        split = X[:,np.arange(window_size)[None,:] + np.arange(num_windows_per_inst)[:,None]]


        if num_windows_per_inst == 1:
            X_transform = self.pca.fit_transform(X)
        elif self.window_size != 0 and self.window_size != series_length:
            
            flat_split = np.reshape(split,(-1,split.shape[2]))
            split_transorm = self.pca.fit_transform(flat_split)
        
        # print(self.pca.explained_variance_ratio_)
        self.evcr = self.pca.explained_variance_ratio_

        if self.assignment_policy == 'log':
            assigned_evc = self.evcr[0:self.word_length]

            alphabet_size = np.log2(assigned_evc*100)
            alphabet_size[alphabet_size < 2] = 2
            alphabet_size = alphabet_size.astype(np.int32)
            print(alphabet_size)
            self.alphabet_size = alphabet_size
        elif self.assignment_policy == 'budget':
            assigned_evc = self.evcr[0:self.word_length]

            avg_allocation = self.bit_budget // self.word_length

            if 2**(avg_allocation+2) > n_instances:
                max_allocation = avg_allocation+1
            else:
                max_allocation = avg_allocation+3
            
            candidates = list(permutations_w_constraints(self.word_length,self.bit_budget,min_value=1,max_value=max_allocation))
            candidates = np.array(candidates)

            rewards = (assigned_evc[None,:] * candidates)
            rewards = rewards - ((0.15*(candidates-1)) * rewards)

            score = np.sum(rewards,axis=1)

            max_candidate_ind = np.argmax(score)

            allocation = candidates[max_candidate_ind]
            print(allocation)
            alphabet_size = 2**allocation
            print(alphabet_size)
            self.alphabet_size = alphabet_size
        elif self.assignment_policy == 'log_budget':
            candidates = list(permutations_w_constraints(self.word_length,self.bit_budget,min_value=1,max_value=self.bit_budget))
            candidates = np.array(candidates)

            assigned_evc = self.evcr[0:self.word_length]
            log_evc = np.log2(assigned_evc*100)
            score = np.sum(log_evc * candidates,axis=1)

            score = score / ((np.var(candidates,axis=1) + 1))
            max_candidate_ind = np.argmax(score)

            allocation = candidates[max_candidate_ind]
            print(allocation)
            alphabet_size = 2**allocation
            print(alphabet_size)
            self.alphabet_size = alphabet_size
        elif self.assignment_policy == 'direct':
            self.alphabet_size = self.alphabet_size
        elif self.assignment_policy == 'dp':
            if isinstance(self.alphabet_size, list):
                self.alphabet_size = self.alphabet_size[0]

            total_bit = int(np.log2(self.alphabet_size)*self.word_length)
            assigned_evc = self.evcr[0:self.word_length]

            assigned_evc = assigned_evc / np.sum(assigned_evc)
            DP_reward, bit_arr = dynamic_bit_allocation_update(total_bit, assigned_evc, self.min_bit, self.max_bit, delta=self.delta)
            self.alphabet_size = [int(2**bit_arr[i]) for i in range(len(bit_arr))]
            print("dp result: ", self.alphabet_size)
        

        if num_windows_per_inst == 1:

            kept_components = X_transform[:,0:self.word_length]

            self.breakpoints = self.binning(kept_components)

            words = self.generate_words(kept_components,self.breakpoints)
        else:
            breakpoints = self.binning(split_transorm)

            self.breakpoints = breakpoints
            flat_words = self.generate_words(split_transorm,breakpoints)

            words = np.reshape(flat_words,(n_instances,num_windows_per_inst,self.word_length))
            self.train_histogram = self.bag_to_hist(self.create_bags(words)) 

        return words

    def generate_words(self,pca,breakpoints):
        words = np.zeros((pca.shape[0],self.word_length))
        for a in range(pca.shape[0]):
            for i in range(self.word_length):
                words[a,i] = np.digitize(pca[a,i],breakpoints[i],right=True)
        
        return words

    def binning(self,pca):
        if self.binning_method == 'equi-depth' or self.binning_method == 'equi-width':
            breakpoints = self._mcb(pca)
        return breakpoints
    def _mcb(self,pca):
        breakpoints = []
        mindist_breakpoints =[]

        pca = np.round(pca,4)
        for letter in range(self.word_length):
            column = np.sort(pca[:,letter])
            column_min = np.min(column)
            bin_index = 0
            
            letter_alphabet_size = self.alphabet_size[letter]
            breakpoint_i = np.zeros(letter_alphabet_size)
            mindist_breakpoint_i = np.zeros(letter_alphabet_size)

            mindist_breakpoint_i[0] = column_min

            #use equi-depth binning
            if self.binning_method == "equi-depth":
                target_bin_depth = len(pca) / letter_alphabet_size

                for bp in range(letter_alphabet_size - 1):
                    bin_index += target_bin_depth 
                    breakpoint_i[bp] = column[int(bin_index)]
                    mindist_breakpoint_i[bp+1] = column[int(bin_index)]
            
            #equi-width binning aka equi-frequency binning
            elif self.binning_method == "equi-width":
                target_bin_width = (column[-1] - column[0]) / letter_alphabet_size

                for bp in range(letter_alphabet_size - 1):
                    breakpoint_i[bp] = (bp + 1) * target_bin_width + column[0]
            breakpoint_i[letter_alphabet_size - 1] = sys.float_info.max
            breakpoints.append(breakpoint_i)
            mindist_breakpoints.append(mindist_breakpoint_i)
            
        # breakpoints[:, self.alphabet_size - 1] = sys.float_info.max
        self.mindist_breakpoints = mindist_breakpoints
        return breakpoints
    
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

        possible_words = self.alphabet_size[0] ** word_length
        word_to_num = [np.base_repr(i,base=self.alphabet_size[0]) for i in range(possible_words)]

        word_to_num = ['0'*(word_length - len(word)) + word for word in word_to_num]
        all_win_words = np.zeros((n_instances,possible_words))

        # print(word_to_num)

        for j in range(n_instances):
            bag = bags[j]
            # print(bag)

            for key in bag.keys():
                v = bag[key]

                n = word_to_num.index(key)

                all_win_words[j,n] = v
        return all_win_words

def permutations_w_constraints(n_perm_elements, sum_total, min_value, max_value):
    # base case
    if n_perm_elements == 1:
        if (sum_total <= max_value) & (sum_total >= min_value):
            yield (sum_total,)
    else:
        for value in range(min_value, max_value + 1):
            for permutation in permutations_w_constraints(
                n_perm_elements - 1, sum_total - value, min_value, max_value
            ):
                if value >= permutation[0]:
                    yield (value,) + permutation