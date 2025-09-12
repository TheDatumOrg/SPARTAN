import os

NUM_JOBS = "1"
# set cpu usage limit
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

from util.dataset import create_numpy_dataset
from util.tools import univariate

from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LinearRegression

from tsfresh.feature_extraction import feature_calculators

import random
random.seed(42)
np.random.seed(42)


window_size = -1
word_length = 12
alphabet_size = 4

repeat_num = 5
dataset_num = 128

def entropy(signal, prob="standard"):
    """Computes the entropy of the signal using the Shannon Entropy.

    Description in Article:
    Regularities Unseen, Randomness Observed: Levels of Entropy Convergence
    Authors: Crutchfield J. Feldman David

    Feature computational cost: 1

    Parameters
    ----------
    signal : nd-array
        Input from which entropy is computed
    prob : string
        Probability function (kde or gaussian functions are available)

    Returns
    -------
    float
        The normalized entropy value

    """

    if prob == "standard":
        value, counts = np.unique(signal, return_counts=True)
        p = counts / counts.sum()
    elif prob == "kde":
        p = kde(signal)
    elif prob == "gauss":
        p = gaussian(signal)

    if np.sum(p) == 0:
        return 0.0

    # Handling zero probability values
    p = p[np.where(p != 0)]

    # If probability all in one value, there is no entropy
    if np.log2(len(signal)) == 1:
        return 0.0
    elif np.sum(p * np.log2(p)) / np.log2(len(signal)) == 0:
        return 0.0
    else:
        return -np.sum(p * np.log2(p)) / np.log2(len(signal))

def slope(X):

    max_pt = np.argmax(X)
    min_pt = np.argmin(X)

    max_val = X[max_pt]
    min_val = X[min_pt]

    return (max_val - min_val) / (max_pt - min_pt)

def sax_vfd_feat_func(X, func_name):
    
    # print(func_name, X.shape)
    X = np.array(X).reshape(-1,)
    
    if func_name == 'max':
        return feature_calculators.maximum(X)
    elif func_name == 'min':
        return feature_calculators.minimum(X)
    elif func_name == 'mean':
        return feature_calculators.mean(X)
    elif func_name == 'median':
        return feature_calculators.median(X)   
    elif func_name == 'var':
        return feature_calculators.variance(X)
    elif func_name == 'skew':
        return feature_calculators.skewness(X)
    elif func_name == 'kurtosis':
        return feature_calculators.kurtosis(X)
    elif func_name == 'range':
        return np.max(X) - np.mean(X)
    elif func_name == 'IQR':
        return np.percentile(X, 75) - np.percentile(X, 25)
    elif func_name == 'entropy':
        return entropy(X)
    elif func_name == 'bEn':
        return feature_calculators.binned_entropy(X, max_bins=10)
    elif func_name == 'apEn':
        return feature_calculators.approximate_entropy(X,m=3,r=0.2)
    elif func_name == 'sampEn':
        return feature_calculators.sample_entropy(X)
    elif func_name == 'slope':
        return slope(X)
    elif func_name == 'abs_en':
        return feature_calculators.abs_energy(X)
    elif func_name == 'abs_sum_of_ch':
        return feature_calculators.absolute_sum_of_changes(X)
    elif func_name == 'mean_abs_ch':
        return feature_calculators.mean_abs_change(X)
    elif func_name == 'mean_sec_deri_central':
        return feature_calculators.mean_second_derivative_central(X)

    else:

        print('no matching func!')
        assert func_name == 'max'

def saxdr_direct_feat(X):

    N, M = X.shape
    
    slope = np.array(X[:, 1:]) - np.array(X[:, :-1])

    pos_slope = np.sum(slope>0, axis=1)
    neg_slope = np.sum(slope<0, axis=1)

    # print("pos shape: ", pos_slope.shape)

    direct_feat = np.zeros(N)
    direct_feat = np.where(pos_slope > (M/2), 2, 0)
    direct_feat = np.where((pos_slope - neg_slope) == 0, 1, direct_feat)
    # print("direct feat: ", direct_feat.shape)
    return direct_feat

def tfsax_trend_feat(X):

    N, M  = X.shape
    mean_val = np.mean(X, axis=1)
    td = (X[:, -1] - mean_val) - (X[:,0] - mean_val)
    
    # K_arr = np.ones(N)
    # for i in range(N):
    #     K_arr[i] = tfsax_trend_point(X[i])

    K_arr = tfsax_trend_point_vec(X)

    tan = td / K_arr

    angle = np.arctan(tan) * 180 / np.pi

    return angle

def tfsax_trend_point_vec(X):

    N, M = X.shape
   
    neg_check = (X[:, 1:-1] - X[:, :-2])*(X[:, 2:] - X[:, 1:-1]) < 0
    zero_check = (X[:, 1:-1] - X[:, :-2])*(X[:, 2:] - X[:, 1:-1]) == 0
    diff_check = (X[:, 1:-1] - X[:, :-2])!=(X[:, 2:] - X[:, 1:-1])

    K = np.where(neg_check | (zero_check & diff_check), 1, 0)
    K = np.sum(K, axis=1)
    K = np.maximum(K,1)

    return K

def tfsax_trend_point(X):
    
    K = 0
    for i in range(1, len(X)-1):

        if (X[i] - X[i-1])*(X[i+1] - X[i]) < 0:
            K += 1
        elif ((X[i] - X[i-1])*(X[i+1] - X[i]) == 0) and ((X[i] - X[i-1]) != (X[i+1] - X[i])): 
            K += 1

    return max(K, 1)

def slope_feat(y_data):

    # N, M = X.shape
    # slope = np.zeros(N)
    # time_steps = np.arange(M).reshape(-1,1)

    # for i in range(N):
        
    #     reg = LinearRegression().fit(time_steps, X[i])
    #     slope[i] = reg.coef_[0]
    x_ind = np.arange(len(y_data[0])).reshape(1,-1)
    x_ind_mean = np.mean(x_ind, axis=1, keepdims=True)

    s_nom = np.sum((x_ind - x_ind_mean)*y_data, axis=1)
    
    s_den = np.sum((x_ind - x_ind_mean)**2, axis=1)

    
    return s_nom / s_den

def paa_fit(X,word_length):
   
    X = X.reshape(-1, X.shape[-1])
    X_split = np.array_split(X,word_length,axis=1)
    X_split = np.concatenate([np.expand_dims(np.mean(x,axis=1),axis=1) for x in X_split],axis=1)
    X_transform = X_split

    return X_transform

def paa_inference(X):

    X = X.reshape(-1, X.shape[-1])
    X_split = np.array_split(X,word_length,axis=1)
    X_split = np.concatenate([np.expand_dims(np.mean(x,axis=1),axis=1) for x in X_split],axis=1)
    
    return X_split

def esax_fit(X, word_length):

    word_length = int(word_length // 3)
    X = X.reshape(-1, X.shape[-1])

    X_split = np.array_split(X,word_length,axis=1)
    X_mean = np.concatenate([np.expand_dims(np.mean(x,axis=1),axis=1) for x in X_split],axis=1)
    X_max = np.concatenate([np.expand_dims(np.max(x,axis=1),axis=1) for x in X_split],axis=1)
    X_min = np.concatenate([np.expand_dims(np.min(x,axis=1),axis=1) for x in X_split],axis=1)
    
    return X_split

def saxdr_fit(X, word_length):

    X = X.reshape(-1, X.shape[-1])

    word_length = int(word_length // 2)
    X_split = np.array_split(X,word_length,axis=1)
 
    X_mean = np.concatenate([np.expand_dims(np.mean(x,axis=1),axis=1) for x in X_split],axis=1)
    X_dr_feat = np.concatenate([np.expand_dims(saxdr_direct_feat(x),axis=1) for x in X_split],axis=1)
    
    return X_split

def tfsax_fit(X, word_length):

    X = X.reshape(-1, X.shape[-1])

    word_length = int(word_length // 2)
    X_split = np.array_split(X,word_length,axis=1)
 
    X_mean = np.concatenate([np.expand_dims(np.mean(x,axis=1),axis=1) for x in X_split],axis=1)
    X_tf_feat = np.concatenate([np.expand_dims(tfsax_trend_feat(x),axis=1) for x in X_split],axis=1)
    
    return X_split

def onedsax_fit(X, word_length):

    X = X.reshape(-1, X.shape[-1])

    word_length = int(word_length // 2)
    X_split = np.array_split(X,word_length,axis=1)
 
    X_mean = np.concatenate([np.expand_dims(np.mean(x,axis=1),axis=1) for x in X_split],axis=1)
    X_slope = np.concatenate([np.expand_dims(slope_feat(x),axis=1) for x in X_split],axis=1)

    return X_split

def sax_vfd_fit(X_train, X_test, word_length):

    # print("training")
    X_train = X_train.reshape(-1, X_train.shape[-1])
    X_test  = X_test.reshape(-1, X_test.shape[-1])

    word_length = int(word_length // 4)

    train_num = len(X_train)
    test_num = len(X_test)
    # X_mean = np.concatenate([np.expand_dims(np.mean(x,axis=1),axis=1) for x in X_split],axis=1)
    # X_slope = np.concatenate([np.expand_dims(slope_feat(x),axis=1) for x in X_split],axis=1)

    if train_num >= 50:
        N = min(int(0.1*train_num), test_num)
    else:
        N = min(int(0.5*train_num), test_num)

    select_train_idx = np.random.choice(train_num, size=N, replace=False)
    select_test_idx  = np.random.choice(test_num,  size=N, replace=False)
    select_train = X_train[select_train_idx]
    select_test  = X_test[select_test_idx]

    X_train_split = np.array_split(select_train,word_length,axis=1)
    X_test_split  = np.array_split(select_test,word_length,axis=1)

    feat_list = ['max', 'min', 'mean', 'median', 'var', 
                 'skew', 'slope', 'range', 'IQR', 'entropy', 
                 'mean_sec_deri_central', 'mean_abs_ch', #'sampEn', # 'apEn', 
                 'abs_sum_of_ch', 'kurtosis','abs_en', 'bEn']
    for x in X_train_split:
        
        for feat_name in feat_list:
            sax_vfd_feat_func(x, func_name=feat_name)
    for x in X_test_split:
        for feat_name in feat_list:
            sax_vfd_feat_func(x, func_name=feat_name)
    return None

def sax_vfd_inference(X_train, X_test, word_length):

    # print("inference")
    X_train = X_train.reshape(-1, X_train.shape[-1])
    X_test  = X_test.reshape(-1, X_test.shape[-1])

    word_length = int(word_length // 4)
    
    X_train_split = np.array_split(X_train,word_length,axis=1)
    X_test_split  = np.array_split(X_test,word_length,axis=1)

    feat_list = ['max', 'min', 'mean', 'median', 'var', 
                 'skew', 'slope', 'range', 'IQR', 'entropy', 
                 'mean_sec_deri_central', 'mean_abs_ch', #'sampEn', 'apEn', 
                 'abs_sum_of_ch', 'kurtosis','abs_en', 'bEn']
    
    rand_ind = np.random.choice(len(feat_list), size=4, replace=False)

    for x in X_train_split:
        for ind in rand_ind:
            sax_vfd_feat_func(x, func_name=feat_list[ind])
    for x in X_test_split:
        for ind in rand_ind:
            sax_vfd_feat_func(x, func_name=feat_list[ind])
    return None

def dft_fit(X, word_length):
    X= X.reshape(-1,X.shape[-1])

    X_ffts = np.fft.fft(X,axis=1)

    reals = np.real(X_ffts)  
    imags = np.imag(X_ffts) 

    dft = np.zeros((len(X), word_length))
    dft[:, 0::2] = reals[:, 0 : word_length // 2]
    dft[:, 1::2] = imags[:, 0 : word_length // 2]

    return dft


if __name__ == "__main__":

    data_path = "data/Univariate_ts/"

    methods = ['sax','sfa', 'esax', 'sax_dr', 'tfsax', '1dsax', 'sax_vfd']

    results = pd.DataFrame()

    univariate = list(univariate)

    for i in range(len(univariate)):
        dataset = univariate[i]

        if i > dataset_num - 1:
            break

        X_train,y_train,X_test,y_test = create_numpy_dataset(name=dataset,path=data_path)

        #Normalize Labels
        label_encode = LabelEncoder()
        y_train_transformed = label_encode.fit_transform(y_train)
        y_test_transformed = label_encode.transform(y_test)

        num_train = len(X_train)

        X_train = X_train[:,0,:]
        X_test = X_test[:,0,:]
        print(f"DATASET [{i}]: {dataset}, Train: {X_train.shape}, Test: {X_test.shape}")

        train_means = np.mean(X_train,axis=1,keepdims=True)
        train_stds = np.std(X_train,axis=1,keepdims=True)
        test_means = np.mean(X_test,axis=1,keepdims=True)
        test_stds = np.std(X_test,axis=1,keepdims=True)

        train_stds[train_stds == 0] = 1
        test_stds[test_stds == 0] = 1

        X_train_transform = (X_train - train_means) / train_stds
        X_test_transform = (X_test - test_means) / test_stds

        if window_size == -1:
            dataset_win = X_train_transform.shape[1]
            num_windows_per_inst = 1
        else:
            dataset_win = window_size
            num_windows_per_inst = X_train_transform.shape[1] - window_size + 1

        X_train_split = X_train_transform[:,np.arange(dataset_win)[None,:] + np.arange(num_windows_per_inst)[:,None]]
        X_test_split = X_test_transform[:,np.arange(dataset_win)[None,:] + np.arange(num_windows_per_inst)[:,None]]

        for method in methods:
            
            train_time = 0.0
            pred_time = 0.0
            for j in range(repeat_num):

                fit_start = time.time()
                
                if method == 'sax':
                    paa = paa_fit(X_train_split,word_length)
                elif method =='sfa':
                    dft = dft_fit(X_train_split,word_length)
                elif method == 'esax':
                    paa = esax_fit(X_train_split,word_length)
                elif method == 'sax_dr':
                    paa = saxdr_fit(X_train_split,word_length)
                elif method == 'tfsax':
                    paa = tfsax_fit(X_train_split,word_length)
                elif method == '1dsax':
                    paa = onedsax_fit(X_train_split,word_length)
                elif method == 'sax_vfd':
                    paa = sax_vfd_fit(X_train_split, X_test_split, word_length)
        

                fit_end = time.time()

                pred_start = time.time()
                
                if method == 'sax':
                    paa = paa_inference(X_test_split)
                elif method =='sfa':
                    dft = dft_fit(X_test_split,word_length)
                elif method == 'esax':
                    paa = esax_fit(X_test_split,word_length)
                elif method == 'sax_dr':
                    paa = saxdr_fit(X_test_split,word_length)
                elif method == 'tfsax':
                    paa = tfsax_fit(X_test_split,word_length)
                elif method == '1dsax':
                    paa = onedsax_fit(X_test_split,word_length)
                elif method == 'sax_vfd':
                    paa = sax_vfd_inference(X_train_split, X_test_split, word_length)
                

                pred_end= time.time()

                train_time_iter = fit_end - fit_start
                pred_time_iter = pred_end - pred_start

                train_time += train_time_iter
                pred_time += pred_time_iter

            train_time = train_time / repeat_num
            pred_time = pred_time / repeat_num
            time_record = pd.DataFrame([{'dataset':dataset,'method':method,'train_time':train_time,'pred_time':pred_time}])
            results = pd.concat([results,time_record],ignore_index=True)
          
    print(results)

    result_dir = './result/Section5_1' # change as needed
    os.makedirs(result_dir, exist_ok=True)

    results.to_csv(os.path.join(result_dir, f'cumulative_runtime_results_comparison_{len(methods)}methods_{dataset_num}dataset.csv'),index=False)

    summary = results.groupby('method')[['train_time', 'pred_time']].sum()
    print(summary)