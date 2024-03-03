from sklearn.decomposition import PCA

import os
import sys
import numpy as np

sys.path.append("../")
from tools import dynamic_bit_allocation, dynamic_bit_allocation_update

binning_methods = {"equi-depth", "equi-width", "information-gain", "kmeans", "quantile"}

class SpartanPCAAllocation:
    def __init__(self,
                 alphabet_size=[8,4,4,2],
                 window_size=0,
                 word_length=4,
                 binning_method='equi-depth',
                 assignment_policy = 'direct',
                 bit_budget = 16,
                 max_bit=3,
                 min_bit=1,
                 delta=0):
        self.alphabet_size = alphabet_size
        self.window_size = window_size
        self.word_length = word_length
        self.binning_method = binning_method
        self.assignment_policy = assignment_policy
        self.bit_budget = bit_budget
        self.max_bit = max_bit
        self.min_bit = min_bit
        self.delta = delta
    def transform(self,X):
        X_transform = self.pca.transform(X)

        kept_components = X_transform[:,0:self.word_length]
        words = self.generate_words(kept_components,self.breakpoints)

        return words
    def fit_transform(self,X,y=None):
        self.pca = PCA()

        self._y = y
        self._X = X

        n_instances, series_length = X.shape

        window_size = self.window_size
        if self.window_size == 0:
            window_size = series_length

        num_windows_per_inst = series_length - window_size + 1
        split = X[:,np.arange(window_size)[None,:] + np.arange(num_windows_per_inst)[:,None]]

        X_transform = self.pca.fit_transform(X)

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

            # path = f'./candidates/candidates_b{self.bit_budget}_w{self.word_length}.npy'
            # if os.path.isfile(path):
            #     candidates = np.load(path)
            # else:
            #     candidates = list(permutations_w_constraints(self.word_length,self.bit_budget,min_value=1,max_value=max_allocation))
            #     candidates = np.array(candidates)
            #     np.save(path,candidates)
            candidates = list(permutations_w_constraints(self.word_length,self.bit_budget,min_value=1,max_value=max_allocation))
            candidates = np.array(candidates)
            # print(candidates.shape)
            # print(candidates)
            rewards = (assigned_evc[None,:] * candidates)
            rewards = rewards - ((0.15*(candidates-1)) * rewards)
            # 0.15*(candidates-1) * reward
            # 0.15*(candidates-1) * evc * candidates
            score = np.sum(rewards,axis=1)
            # print(score)
            # print(np.var(candidates,axis=1))
            # print(score)
            max_candidate_ind = np.argmax(score)

            allocation = candidates[max_candidate_ind]
            print(allocation)
            alphabet_size = 2**allocation
            print(alphabet_size)
            self.alphabet_size = alphabet_size
        elif self.assignment_policy == 'log_budget':
            candidates = list(permutations_w_constraints(self.word_length,self.bit_budget,min_value=1,max_value=self.bit_budget))
            candidates = np.array(candidates)
            # print(candidates.shape)
            # print(candidates)
            assigned_evc = self.evcr[0:self.word_length]
            log_evc = np.log2(assigned_evc*100)
            score = np.sum(log_evc * candidates,axis=1)
            # print(score)
            score = score / ((np.var(candidates,axis=1) + 1))
            # print(np.var(candidates,axis=1))
            # print(score)
            max_candidate_ind = np.argmax(score)

            allocation = candidates[max_candidate_ind]
            print(allocation)
            alphabet_size = 2**allocation
            print(alphabet_size)
            self.alphabet_size = alphabet_size
        elif self.assignment_policy == 'quantiles':
            assigned_evc = self.evcr[0:self.word_length]
            quantiles = np.zeros_like(assigned_evc)
            quantiles = np.quantile(self.evcr,q=[0.75,0.5,0.25])
        elif self.assignment_policy == 'direct':
            self.alphabet_size = self.alphabet_size

        elif self.assignment_policy == 'dp':
            
            # print(self.alphabet_size, self.word_length)
            if isinstance(self.alphabet_size, list):
                self.alphabet_size = self.alphabet_size[0]

            total_bit = int(np.log2(self.alphabet_size)*self.word_length)
            assigned_evc = self.evcr[0:self.word_length]
            
            # DP_reward, bit_arr = dynamic_bit_allocation(total_bit, assigned_evc, self.min_bit, self.max_bit)

            assigned_evc = assigned_evc / np.sum(assigned_evc)
            DP_reward, bit_arr = dynamic_bit_allocation_update(total_bit, assigned_evc, self.min_bit, self.max_bit, delta=self.delta)
            self.alphabet_size = [int(2**bit_arr[i]) for i in range(len(bit_arr))]
            # self.alphabet_size = [8, 8, 4, 4, 4, 4, 2, 2]
            print("dp result: ", self.alphabet_size)
        

        kept_components = X_transform[:,0:self.word_length]

        self.breakpoints = self.binning(kept_components)

        words = self.generate_words(kept_components,self.breakpoints)

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
        # breakpoints = np.zeros((self.word_length,self.alphabet_size))

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

# def generate_candidates(length,sum):
#     if length == 1:
#         yield (sum,)
#     else:
#         for value in range(sum + 1):
#             for permutation in generate_candidates(length-1,sum-value):
#                 yield (value,) + permutation
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