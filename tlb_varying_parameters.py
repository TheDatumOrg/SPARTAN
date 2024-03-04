import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
# plt.rcParams.update({'font_size':22})

from tslearn.piecewise import SymbolicAggregateApproximation

from distance_vectorized import euclidean_vectorized
from dataset import create_numpy_dataset
from normalization import create_normalizer
from spartan_classifier import SPARTANClassifier
from sax_classifier import SAXDictionaryClassifier
from sfa_classifier import SFADictionaryClassifier

from sklearn.preprocessing import LabelEncoder

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data", required=False, default="../TSC-Benchmark/tscbench/data/Univariate_ts/")
parser.add_argument("-p", "--problem", required=False, default="ECG200")  # see data_loader.regression_datasets

arguments = parser.parse_args()



word_length = 11
alphabet_size = 8
window_size = 0

dset_size = 300
n_elements_per_word = 32

dset = np.random.normal(loc=0,scale=1,size=(dset_size,n_elements_per_word*word_length)) + np.random.uniform(dset_size)
y = np.random.randint(0,4,dset_size)

word_sizes = np.arange(2,9,1)
alphabet_sizes = np.arange(3,11,1)
euclidean_dist_mat = euclidean_vectorized(dset,dset)
spartan_tlb_values = np.zeros((len(word_sizes),len(alphabet_sizes)))
sax_tlb_values = np.zeros((len(word_sizes),len(alphabet_sizes)))
sfa_tlb_values = np.zeros((len(word_sizes),len(alphabet_sizes)))

dataset = arguments.problem
data_path = arguments.data
normalization = 'zscore'

X_train,y_train,X_test,y_test = create_numpy_dataset(name=dataset,path=data_path)

X_train = X_train[:,0,:]
X_test = X_test[:,0,:]

train_means = np.mean(X_train,axis=1,keepdims=True)
train_stds = np.std(X_train,axis=1,keepdims=True)
test_means = np.mean(X_test,axis=1,keepdims=True)
test_stds = np.std(X_test,axis=1,keepdims=True)

test_stds[test_stds == 0] = 1

X_train_transform = (X_train - train_means) / train_stds
X_test_transform = (X_test - test_means) / test_stds

n_instances,n_timepoints = X_train_transform.shape

X_all = np.concatenate([X_train_transform,X_test_transform],axis=0)



label_encode = LabelEncoder()
y_train_transformed = label_encode.fit_transform(y_train)
y_test_transformed = label_encode.transform(y_test)

y_all = np.concatenate([y_train_transformed,y_test_transformed],axis=0)

n_instances,n_timepoints = X_all.shape

euclidean_dist_mat = euclidean_vectorized(X_all,X_all)

tlb_results = pd.DataFrame()


for n, word_len in enumerate(word_sizes):
    for m,alphabet_size in enumerate(alphabet_sizes):
        spartan = SPARTANClassifier(
            alphabet_size=int(alphabet_size),
            word_length=word_len,
            metric='sax_mindist'
        )
        sax= SAXDictionaryClassifier(
            alphabet_size=alphabet_size,
            word_length=word_len,
            window_size=window_size,
            save_words=True,
            metric='sax_mindist'
        )
        sfa = SFADictionaryClassifier(
            alphabet_size=alphabet_size,
            word_length=word_len,
            window_size=window_size,
            lower_bounding=False,
            save_words=True,
            norm=True,
            n_jobs=1,
            metric='sax_mindist'
        )
        spartan.fit(X_all,y_all)
        pred = spartan.predict(X_all)

        # sax.fit(X_all,y_all)
        # pred = sax.predict(X_all)

        sfa.fit(X_all,y_all)
        pred = sfa.predict(X_all)
        # sax_dist_mat = ((X_all.shape[1] / word_len)**0.5) * sax.dist_mat
        spartan_dist_mat = spartan.dist_mat
        sfa_dist_mat = sfa.dist_mat
        

        tsl_sax = SymbolicAggregateApproximation(
            n_segments=word_len,
            alphabet_size_avg=alphabet_size
        )

        tsl_words = tsl_sax.fit_transform(X_all)

        tsl_sax_dist_mat = np.zeros((tsl_words.shape[0],tsl_words.shape[0]))
        for i in range(tsl_words.shape[0]):
            for j in range(tsl_words.shape[0]):

                tsl_sax_dist_mat[i,j] = tsl_sax.distance_sax(tsl_words[i],tsl_words[j])

        spartan_tlbs = []
        for i,j in zip(spartan_dist_mat.ravel(),euclidean_dist_mat.ravel()):
            if i == 0 and j == 0:
                spartan_tlbs.append(1)
            else:
                spartan_tlbs.append(i / j)

        spartan_mean_tlb = np.mean(spartan_tlbs)

        spartan_tlb_values[n,m] = spartan_mean_tlb

        sax_tlbs = []
        for i,j in zip(tsl_sax_dist_mat.ravel(),euclidean_dist_mat.ravel()):
            if i == 0 and j == 0:
                sax_tlbs.append(1)
            else:
                sax_tlbs.append(i / j)

        sax_mean_tlb = np.mean(sax_tlbs)

        sax_tlb_values[n,m] = sax_mean_tlb
        

        sfa_tlbs = []
        for i,j in zip(sfa_dist_mat.ravel(),euclidean_dist_mat.ravel()):
            if i == 0 and j == 0:
                sfa_tlbs.append(1)
            else:
                sfa_tlbs.append(i / j)

        sfa_mean_tlb = np.mean(sfa_tlbs)

        sfa_tlb_values[n,m] = sfa_mean_tlb

        sax_record = pd.DataFrame([{'a':alphabet_size,'w':word_len,'method':'sax','tlb':sax_mean_tlb}])
        sfa_record = pd.DataFrame([{'a':alphabet_size,'w':word_len,'method':'sfa','tlb':sfa_mean_tlb}])
        spartan_record = pd.DataFrame([{'a':alphabet_size,'w':word_len,'method':'spartan','tlb':spartan_mean_tlb}])

        tlb_results = pd.concat([tlb_results,sax_record,sfa_record,spartan_record],ignore_index=True)

tlb_results.to_csv(f'figures/tlb/{dataset}_tlb_results.csv',index=False)
        

# print(np.max(sfa_tlb_values))
# print(tlb_values)

fig = plt.figure(figsize=(12,4))

ax1 = fig.add_subplot(131, projection='3d')
ax2 = fig.add_subplot(132, projection='3d')
ax3 = fig.add_subplot(133,projection='3d')

width = depth = 1


x,y = np.meshgrid(alphabet_sizes,word_sizes)
bottom = np.zeros_like(x)

# print(x)
# print(y)

ax1.bar3d(x.ravel(),y.ravel(),bottom.ravel(),width,depth,spartan_tlb_values.ravel(),shade=True)
ax1.invert_xaxis()
# ax1.set_zlim(0.7)
ax1.set_zlim(0,0.7)
ax1.set_xlabel('w: word_length')
ax1.set_ylabel('a: alphabet_size')
ax1.set_zlabel('mean tlb')
ax1.set_title(f'Spartan Mean TLB {dataset}')

ax2.bar3d(x.ravel(),y.ravel(),bottom.ravel(),width,depth,sax_tlb_values.ravel(),shade=True)
ax2.invert_xaxis()
ax2.set_zlim(0,0.7)
ax2.set_xlabel('w: word_length')
ax2.set_ylabel('a: alphabet_size')
ax2.set_zlabel('mean tlb')
ax2.set_title(f'SAX Mean TLB {dataset}')

ax3.bar3d(x.ravel(),y.ravel(),bottom.ravel(),width,depth,sfa_tlb_values.ravel(),shade=True)
ax3.invert_xaxis()
ax3.set_zlim(0,0.7)
ax3.set_xlabel('w: word_length')
ax3.set_ylabel('a: alphabet_size')
ax3.set_zlabel('mean tlb')
ax3.set_title(f'SFA* Mean TLB {dataset}')

plt.savefig(f'{dataset}_tlb.png')