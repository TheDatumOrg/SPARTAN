import math
import argparse
import json
import os
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from scipy.spatial import distance
from torch import _euclidean_dist

from dataset import create_numpy_dataset
from normalization import create_normalizer
from tools import univariate

from distance_vectorized import euclidean_vectorized
from scipy.spatial import distance

from sax_classifier import SAXDictionaryClassifier
from sfa_classifier import SFADictionaryClassifier
from spartan_pca_classifier import SpartanPCAClassifier
from tslearn.piecewise import SymbolicAggregateApproximation

method_kwargs = [json.load(open(config)) for config in method_configs]

tlb_results = pd.DataFrame()

data_path = "../TSC-Benchmark/tscbench/data/Univariate_ts/"
normalization = 'zscore'

dset_info = pd.read_csv('summaryUnivariate.csv')
dset_info = dset_info.sort_values(by=['numTrainCases','numTestCases'])

method = 'sax'

alphabet_size=4
word_len=8
window_size=0

word_length = method_kwargs[0]['word_length']
for i in range(dset_info.shape[0]):
    dataset = dset_info['problem'].iloc[i]
    X_train,y_train,X_test,y_test = create_numpy_dataset(name=dataset,path=data_path)

    n_instances,n_channels,n_timepoints = X_train.shape
    nearest_even_divisor=(n_timepoints // word_length) * word_length 

    X_train = X_train[:,:,:nearest_even_divisor]
    X_test = X_test[:,:,:nearest_even_divisor]

    # normalizer, X_train_transform = create_normalizer(normalization,X_train)
    # X_test_transform = normalizer.transform(X_test)

    X_train = X_train[:,0,:]
    X_test = X_test[:,0,:]

    train_means = np.mean(X_train,axis=1,keepdims=True)
    train_stds = np.std(X_train,axis=1,keepdims=True)
    test_means = np.mean(X_test,axis=1,keepdims=True)
    test_stds = np.std(X_test,axis=1,keepdims=True)

    test_stds[test_stds == 0] = 1

    X_train_transform = (X_train - train_means) / train_stds
    X_test_transform = (X_test - test_means) / test_stds
    

    label_encode = LabelEncoder()
    y_train_transformed = label_encode.fit_transform(y_train)
    y_test_transformed = label_encode.transform(y_test)

    n_instances,n_timepoints = X_train_transform.shape

    X_all = np.concatenate([X_train_transform,X_test_transform],axis=0)
    y_all = np.concatenate([y_train,y_test],axis=0)

    euclidean_dist_mat = distance.cdist(X_all,X_all,'euclidean')
    # euclidean_dist_mat = euclidean_vectorized(X_all,X_all)

    if method == 'spartan':
        spartan = SpartanPCAClassifier(
            alphabet_size=alphabet_size,
            word_length=word_len,
            metric='sax_mindist'
        )
        spartan.fit(X_all,y_all)
        pred = spartan.predict(X_all)

        dist_mat = spartan.dist_mat
    elif method == 'sfa':

        sfa = SFADictionaryClassifier(
            alphabet_size=alphabet_size,
            word_length=word_len,
            window_size=window_size,
            lower_bounding=False,
            save_words=True,
            norm=True,
            n_jobs=-1,
            metric='sax_mindist'
        )
        sfa.fit(X_all,y_all)
        pred = sfa.predict(X_all)

        dist_mat = sfa.dist_mat
    elif method == 'sax':
        tsl_sax = SymbolicAggregateApproximation(
            n_segments=word_len,
            alphabet_size_avg=alphabet_size
        )
        sax = SAXDictionaryClassifier(
            alphabet_size=alphabet_size,
            word_length=word_len,
            window_size=window_size,
            save_words=True,
            metric='sax_mindist'
        )

        sax.fit(X_all,y_all)
        pred = sax.predict(X_all)

        dist_mat = ((X_all.shape[1] / word_len)**0.5) * sax.dist_mat
        # tsl_words = tsl_sax.fit_transform(X_all)

        # tsl_sax_dist_mat = np.zeros((tsl_words.shape[0],tsl_words.shape[0]))
        # for i in range(tsl_words.shape[0]):
        #     for j in range(tsl_words.shape[0]):

        #         tsl_sax_dist_mat[i,j] = tsl_sax.distance_sax(tsl_words[i],tsl_words[j])
        # dist_mat = tsl_sax_dist_mat

    tlbs = []
    for i,j in zip(dist_mat.ravel(),euclidean_dist_mat.ravel()):
        if i == 0 and j == 0:
            tlbs.append(1)
        else:
            tlbs.append(i / j)

    mean_tlb = np.mean(tlbs)

    record = pd.DataFrame([{'dataset':dataset,'method':method,'tlb':mean_tlb}])

    # tlb_results = pd.concat([tlb_results,record],ignore_index=True)

    filename = f'{method}_tlbs_ours.csv'
    with open(filename, 'a') as f:
        record.to_csv(f, mode='a', header=f.tell()==0,index=False)

# tlb_results.to_csv(f'{method}_tlbs.csv',index=False)
