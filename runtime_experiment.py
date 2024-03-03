from curses import window
from tools import univariate

import argparse
import time
import seaborn as sns

import os
import numpy as np
import pandas as pd
from scipy.optimize import linprog
import pulp

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})

from normalization import create_normalizer
from dataset import create_numpy_dataset

from sklearn.preprocessing import LabelEncoder

from sklearn.decomposition import PCA

from paa.paa import PAA

from tools import permutations_w_constraints

window_size =-1
word_length = 4
bit_budget =8


def pca_fit(X,word_length):
    X = X.reshape(-1, X.shape[-1])
    pca = PCA(n_components=word_length,svd_solver='full')

    X_transform = pca.fit_transform(X)
    X_transform = X_transform[:word_length]

    return pca

def pca_allocation_fit(X,word_length,bit_budget):
    X = X.reshape(-1, X.shape[-1])
    pca = PCA(n_components=word_length,svd_solver='full')
    X_transform = pca.fit_transform(X)
    evcr = pca.explained_variance_ratio_
    assigned_evc = evcr[0:word_length]

    avg_allocation = bit_budget // word_length

    if 2**(avg_allocation+2) > X.shape[0]:
        max_allocation = avg_allocation+1
    else:
        max_allocation = avg_allocation+2

    candidates = list(permutations_w_constraints(word_length,bit_budget,min_value=1,max_value=max_allocation))
    candidates = np.array(candidates)

    rewards = (assigned_evc[None,:] * candidates)
    rewards = rewards - ((0.15*(candidates-1)) * rewards)

    score = np.sum(rewards,axis=1)

    max_candidate_ind = np.argmax(score)

    allocation = candidates[max_candidate_ind]
    alphabet_size = 2**allocation

    X_transform = pca.fit_transform(X)
    X_transform = X_transform[:word_length]

    return pca

def pca_allocation_randomized(X,word_length,bit_budget):
    X = X.reshape(-1, X.shape[-1])
    pca = PCA(n_components=word_length)
    X_transform = pca.fit_transform(X)
    evcr = pca.explained_variance_ratio_
    assigned_evc = evcr[0:word_length]

    avg_allocation = bit_budget // word_length

    if 2**(avg_allocation+2) > X.shape[0]:
        max_allocation = avg_allocation+1
    else:
        max_allocation = avg_allocation+2

    candidates = list(permutations_w_constraints(word_length,bit_budget,min_value=1,max_value=max_allocation))
    candidates = np.array(candidates)

    rewards = (assigned_evc[None,:] * candidates)
    rewards = rewards - ((0.15*(candidates-1)) * rewards)

    score = np.sum(rewards,axis=1)

    max_candidate_ind = np.argmax(score)

    allocation = candidates[max_candidate_ind]
    alphabet_size = 2**allocation

    # m = GEKKO()

    # def intermediate(vars,evc):
    #     reward = 0
    #     for i in range(len(vars)):
    #         reward = reward + evc[i]*vars[i] - (0.15*(evc[i] * vars[i]) * (vars[i] -1))
    #     return reward
    # vrs = [m.Var(integer=True,lb=1,ub=max_allocation) for i in range(word_length)]
    # for v in vrs:
    #     v = bit_budget // word_length
    # evc = [m.Const(assigned_evc[i]) for i in range(word_length)]

    # sum_ = m.Const(bit_budget)
    # m.Equation(sum_ == np.sum(vrs))

    # reward = m.Intermediate(intermediate(vrs,evc))

    # m.Minimize(reward)

    # m.solve()

    # print(allocation)
    # print(vrs)
    
    
    

    # obj = [-0.85*assigned_evc[i] for i in range(word_length)]
    
    # constraints_lhs = []
    # constraints_rhs = []

    # #add pairwise constraints
    # for i in range(word_length-1):
    #     constraint = np.zeros(word_length)
    #     constraint[i] = -1
    #     constraint[i+1] = 1

    #     constraints_lhs.append(constraint)
    #     constraints_rhs.append(0)

    # #add max bit and min bit constrint
    # for i in range(word_length):
    #     constraint = np.zeros(word_length)
    #     constraint[i] = -1

    #     constraints_lhs.append(constraint)
    #     constraints_rhs.append(-1)

    #     constraint = np.zeros(word_length)
    #     constraint[i] = 1
    #     constraints_lhs.append(constraint)
    #     constraints_rhs.append(max_allocation)

    # #add sum to bit_budget constraint
    # constraints_lhs.append([1]*word_length)
    # constraints_rhs.append(bit_budget)

    # opt = linprog(c=obj, A_ub=constraints_lhs, b_ub=constraints_rhs,method="highs")
    # res = opt.x

    # print(allocation)
    # print(res)

    X_transform = pca.fit_transform(X)
    X_transform = X_transform[:word_length]

    return pca

def pca_inference(X,pca):
    X = X.reshape(-1, X.shape[-1])
    X_transform = pca.transform(X)
    return X_transform

def paa_fit(X,word_length):
    paa = PAA(num_intervals=word_length)
    # X_transform = paa.transform(X)

    X = X.reshape(-1, X.shape[-1])

    X_split = np.array_split(X,word_length,axis=1)
    X_split = np.concatenate([np.expand_dims(np.mean(x,axis=1),axis=1) for x in X_split],axis=1)
    X_transform = X_split

    return X_transform

def paa_inference(X,paa):
    X = X.reshape(-1, X.shape[-1])

    X_split = np.array_split(X,word_length,axis=1)
    X_split = np.concatenate([np.expand_dims(np.mean(x,axis=1),axis=1) for x in X_split],axis=1)
    # X_transform = paa.transform(X)
    return paa

def dft_fit(X,word_length):
    X= X.reshape(-1,X.shape[-1])

    X_ffts = np.fft.rfft(X,axis=1)
    reals = np.real(X_ffts)  # float64[]
    imags = np.imag(X_ffts)  # float64[]

    dft = np.zeros((len(X), word_length))
    dft[:, 0::2] = reals[:, 0 : word_length // 2]
    dft[:, 1::2] = imags[:, 0 : word_length // 2]

    return dft


data_path = "./data/Univariate_ts/"

dset_info = pd.read_csv('summaryUnivariate.csv')
# dset = np.random.choice(len(dset_info),1)
dset_info = dset_info.sort_values(by=['numTrainCases','numTestCases'])

methods = ['PAA','DFT','PCA+Allocation','PCA+Allocation_randomized']


results = pd.DataFrame()

for i in range(len(dset_info['problem'])):
    dataset = dset_info['problem'].iloc[i]

    X_train,y_train,X_test,y_test = create_numpy_dataset(name=dataset,path=data_path)

    #Create Normalizer & Normalize Data
    # normalizer, X_train_transform = create_normalizer(normalization,X_train)
    # X_test_transform = normalizer.transform(X_test)

    #Normalize Labels
    label_encode = LabelEncoder()
    y_train_transformed = label_encode.fit_transform(y_train)
    y_test_transformed = label_encode.transform(y_test)

    X_train = X_train[:,0,:]
    X_test = X_test[:,0,:]
    print(X_train.shape)

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


        fit_start = time.time()
        if method == 'PAA':
            paa = paa_fit(X_train_split,word_length)
        elif method =='DFT':
            dft = dft_fit(X_train_split,word_length)
        elif method == 'PCA+Allocation':
            pca = pca_allocation_fit(X_train_split,word_length,bit_budget)
        elif method == 'PCA+Allocation_randomized':
            pca = pca_allocation_randomized(X_train_split,word_length,bit_budget)
            
        fit_end = time.time()

        pred_start = time.time()
        if method == 'PAA':
            paa = paa_inference(X_test_split,paa)
        elif method =='DFT':
            dft = dft_fit(X_test_split,word_length)
        elif method == 'PCA+Allocation' or method =='PCA+Allocation_randomized':
            X_transform = pca_inference(X_test_split,pca)
        pred_end= time.time()

        train_time = (fit_end - fit_start) / len(X_train_transform)
        pred_time = (pred_end - pred_start) / len(X_test_transform)

        time_record = pd.DataFrame([{'dataset':dataset,'method':method,'train_time':train_time,'pred_time':pred_time}])
        results = pd.concat([results,time_record],ignore_index=True)

results.to_csv('results/runtime_results_randomized.csv',index=False)