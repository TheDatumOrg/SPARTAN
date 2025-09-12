import os

NUM_JOBS = "1"
# set cpu usage limit for runtime experiments
os.environ["OMP_NUM_THREADS"] = NUM_JOBS # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = NUM_JOBS # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = NUM_JOBS # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = NUM_JOBS # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = NUM_JOBS # export NUMEXPR_NUM_THREADS=1
os.environ['BLIS_NUM_THREADS'] = NUM_JOBS

import argparse
import time

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})

from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA


parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data", required=False, default="data/cbf_gen/")

arguments = parser.parse_args()

data_path = arguments.data

window_size =-1 # single pattern
word_length = 8
bit_budget = 16

sampling_rate = 0.05
repeat_num = 5


def dynamic_alphabet_allocation(total_bit, EV, lamda=0.5):

    def regularization_term(x, ev_value, avg_bit, lamda=0.5, pos=1.0):
        
        return -lamda * (x-avg_bit)**2 * ev_value


    K = len(EV)
    N = total_bit
    A = int(N/K)
    DP = np.zeros((K+1,N+1))
    min_bit = 1
    max_bit = int(np.max(EV) * N)
    alloc = np.zeros_like(DP).astype(np.int32) + N # store the num of bits for each component

    # init
    for i in range(0, K+1):
        for j in range(0, N+1):
            
            DP[i][j] = -1e9

    DP[0][0] = 0

    # non-recursive
    for i in range(1, K+1):
        for j in range(0, N+1):
            
            max_reward = -1e9

            for x in range(min_bit, max_bit+1):

                if j - x >= 0 and x <= alloc[i-1][j-x]:  
                    
                    current_reward = DP[i-1][j-x] + x*EV[i-1] + regularization_term(x, EV[i-1], A, lamda, i/K)

                    if current_reward > max_reward:

                        alloc[i][j] = x
                        max_reward = current_reward
                        DP[i][j] = current_reward

    def print_sol(alloc, K, N):
        
        bit_arr = []  
        unused_bit = N
        for i in range(K, 1, -1):
            bit_arr.append(alloc[i][unused_bit])
            unused_bit -= alloc[i][unused_bit]

        bit_arr.append(unused_bit)
        return bit_arr
    
    bit_arr = print_sol(alloc, K, N)

    assert np.sum(bit_arr) == N

    return DP[K][N], bit_arr[::-1]


def pca_allocation_fit(X, word_length, bit_budget, downsample=1.0, svd_solver='full'):
    
    X = X.reshape(-1, X.shape[-1])
    pca = PCA(n_components=word_length,svd_solver=svd_solver)

    # downsampling with approximation
    if downsample < 1.0:
        sampling_num = min(max(int(np.ceil(len(X)*downsample)), 10), 1000)
        random_indices = np.random.choice(X.shape[0], sampling_num, replace=False)
        pca.fit(X[random_indices])
    else:
        pca.fit(X)
        
    evcr = pca.explained_variance_ratio_
    assigned_evc = evcr[0:word_length]

    assigned_evc = assigned_evc / np.sum(assigned_evc) # normalize

    # alphabet allocation
    DP_reward, bit_arr = dynamic_alphabet_allocation(total_bit=bit_budget, 
                                                     EV=assigned_evc, 
                                                     lamda=0.5)
    
    # transform all data (training)
    X_transform = pca.transform(X)
    X_transform = X_transform[:word_length]

    return pca

def pca_inference(X, pca, word_length):
    X = X.reshape(-1, X.shape[-1])
    X_transform = pca.transform(X)
    return X_transform[:, :word_length]

def paa_fit(X, word_length):
    
    X = X.reshape(-1, X.shape[-1])
    num_step = int(X.shape[-1] / word_length) * word_length # integer multiples
    X_new = X[:, :num_step]

    X_split = X_new.reshape(len(X), word_length, int(num_step / word_length))
    X_transform = np.mean(X_split, axis=-1) # mean for each segment

    return X_transform

def paa_inference(X, word_length):

    X = X.reshape(-1, X.shape[-1])
    num_step = int(X.shape[-1] / word_length) * word_length # integer multiples
    X_new = X[:, :num_step]

    X_split = X_new.reshape(len(X), word_length, int(num_step / word_length))
    X_transform = np.mean(X_split, axis=-1) # mean for each segment
    return X_transform

def dft_fit(X, word_length):
    
    X= X.reshape(-1,X.shape[-1])

    # dft approximation
    X_ffts = np.fft.fft(X,axis=1)

    reals = np.real(X_ffts)  
    imags = np.imag(X_ffts)  

    dft = np.zeros((len(X), word_length))
    dft[:, 0::2] = reals[:, 0 : word_length // 2]
    dft[:, 1::2] = imags[:, 0 : word_length // 2]

    return dft


methods = ['PAA','FFT','PCA+Allocation','PCA+Allocation_randomized', 'PCA+Allocation_randomized_Sampling']

result_dir = './result/scaling_runtime'
os.makedirs(result_dir, exist_ok=True)

results = pd.DataFrame()

scaling_dataset = ['N100', 'N1000', 'N10000', 'N100000', 'N1000000', 'L128', 'L1280', 'L12800', 'L128000']

for scaling_itr in scaling_dataset:
    
    if scaling_itr[0] == 'N': # varying number of series
        
        ts_num = int(scaling_itr[1:])
        X_train = np.load(os.path.join(data_path, f'cbf_train_X_1M.npy'))
        y_train = np.load(os.path.join(data_path, f'cbf_train_Y_1M.npy')).astype(int)
        X_test = np.load(os.path.join(data_path, f'cbf_test_X.npy'))
        y_test = np.load(os.path.join(data_path, f'cbf_test_Y.npy'))

        assert X_train.shape == (1e6, 128) and X_test.shape == (900, 128)
        assert y_train.shape == (1e6, )    and y_test.shape == (900, )

        X_train, y_train = X_train[:ts_num], y_train[:ts_num]
        
        # check data class imbalance from sampling
        if ts_num // 3 == 0:
            assert np.sum(y_train) == int(ts_num/3)*3
        if ts_num // 3 == 1:
            assert np.sum(y_train) == int(ts_num/3)*3
        if ts_num // 3 == 2:
            assert np.sum(y_train) == int(ts_num/3)*3 + 1

    elif scaling_itr[0] == 'L': # varying ts length

        ts_len = int(scaling_itr[1:])
        X_train = np.load(os.path.join(data_path, f'cbf_train_X_1000.npy'))
        y_train = np.load(os.path.join(data_path, f'cbf_train_Y_1000.npy')).astype(int)
        X_test = np.load(os.path.join(data_path, f'cbf_test_X.npy'))
        y_test = np.load(os.path.join(data_path, f'cbf_test_Y.npy'))

        assert X_train.shape == (1000, 128) and X_test.shape == (900, 128)
        assert y_train.shape == (1000, )    and y_test.shape == (900, )

        rep_num = int(ts_len / 128)
        X_train = np.tile(X_train, (1,rep_num))
        X_test  = np.tile(X_test, (1,rep_num))
        assert X_train.shape == (1000, ts_len) and X_test.shape == (900, ts_len)

    
    # Normalize Labels
    label_encode = LabelEncoder()
    y_train_transformed = label_encode.fit_transform(y_train)
    y_test_transformed = label_encode.transform(y_test)

    num_train = len(X_train)

    print(scaling_itr, X_train.shape, X_test.shape)

    train_means = np.mean(X_train,axis=1,keepdims=True)
    train_stds = np.std(X_train,axis=1,keepdims=True)
    test_means = np.mean(X_test,axis=1,keepdims=True)
    test_stds = np.std(X_test,axis=1,keepdims=True)

    train_stds[train_stds == 0] = 1
    test_stds[test_stds == 0] = 1

    X_train_transform = (X_train - train_means) / train_stds
    X_test_transform = (X_test - test_means) / test_stds

    # check sliding window
    if window_size == -1:
        dataset_win = X_train_transform.shape[1]
        num_windows_per_inst = 1
    else:
        dataset_win = window_size
        num_windows_per_inst = X_train_transform.shape[1] - window_size + 1

    X_train_split = X_train_transform[:,np.arange(dataset_win)[None,:] + np.arange(num_windows_per_inst)[:,None]]
    X_test_split = X_test_transform[:,np.arange(dataset_win)[None,:] + np.arange(num_windows_per_inst)[:,None]]

    for method in methods:
        print(method)
        train_time = 0.0
        pred_time = 0.0
        for j in range(repeat_num):
            
            # training
            fit_start = time.time()
            if method == 'PAA':
                paa = paa_fit(X_train_split, word_length)
            elif method =='FFT':
                dft = dft_fit(X_train_split, word_length)
            elif method == 'PCA+Allocation':
                pca = pca_allocation_fit(X_train_split,
                                         word_length,
                                         bit_budget,
                                         svd_solver='full')
            elif method == 'PCA+Allocation_randomized':
                pca = pca_allocation_fit(X_train_split,
                                         word_length,
                                         bit_budget,
                                         svd_solver='randomized')
            elif method.startswith('PCA+Allocation_randomized_Sampling'):
                pca = pca_allocation_fit(X_train_split,
                                         word_length,
                                         bit_budget, 
                                         downsample=sampling_rate, 
                                         svd_solver='randomized')
                
            fit_end = time.time()

            # inference
            pred_start = time.time()
            if method == 'PAA':
                paa = paa_fit(X_test_split, word_length)
            elif method =='FFT':
                dft = dft_fit(X_test_split, word_length)
            elif method in ['PCA+Allocation', 'PCA+Allocation_randomized', 'PCA+Allocation_randomized_Sampling']:
                X_transform = pca_inference(X_test_split, pca, word_length)
            pred_end= time.time()

            train_time_iter = fit_end - fit_start
            pred_time_iter = pred_end - pred_start

            train_time += train_time_iter
            pred_time += pred_time_iter

        train_time = train_time / repeat_num
        pred_time = pred_time / repeat_num
        time_record = pd.DataFrame([{'dataset': f'CBF_{scaling_itr}',
                                     'method':method,
                                     'train_time':train_time,
                                     'pred_time':pred_time, 
                                     'total_time':(train_time+pred_time)}])
        results = pd.concat([results,time_record],ignore_index=True)

results.to_csv(os.path.join(result_dir, f'test_cbf_runtime_results_comparison_{len(methods)}methods.csv'),index=False)