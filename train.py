import argparse
import json
import os
import time
import numpy as np
import pandas as pd

from dataset import create_numpy_dataset
from normalization import create_normalizer
from tools import create_directory,compute_classification_metrics

from sklearn.preprocessing import LabelEncoder

from sax.sax import SAX
from sfa.sfa_fast import SFAFast

from sax_classifier import SAXDictionaryClassifier
from esax_classifier import ESAXDictionaryClassifier
from oned_sax_classifier import OneDSAXDictionaryClassifier
from tfsax_classifier import TFSAXDictionaryClassifier
from sax_dr_classifier import SAXDRDictionaryClassifier
from sax_vfd_classifier import SAXVFDDictionaryClassifier
from sfa_classifier import SFADictionaryClassifier
from spartan_classifier import SPARTANClassifier

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data", required=False, default="../Univariate_ts/")
parser.add_argument("-p", "--problem", required=False, default="ArrowHead")  # see data_loader.regression_datasets
parser.add_argument("-c", "--classifier", required=False, default="sax")  # see regressor_tools.all_models
parser.add_argument("-g",'--config',required=False,default='./configs/sax/sax_boss.json')
parser.add_argument("-i", "--itr", required=False, default=13)
parser.add_argument("-n", "--norm", required=False, default="zscore")  # none, standard, minmax
parser.add_argument("-s","--save_model",required=False, default=None)
parser.add_argument("-r","--skip_repeat",required=False,default=True)
parser.add_argument("-m","--dataset_num",required=False,default=1, type=int)



arguments = parser.parse_args()

module = 'SymbolicRepresentationExperiments'

data_path = arguments.data
classifier_name = arguments.classifier
normalization = arguments.norm
problem = arguments.problem
itr = arguments.itr
config = arguments.config
skip_repeat = arguments.skip_repeat
data_id = arguments.dataset_num

# create output directory
output_directory = "output/classification/"
if normalization != None:
    output_directory = "output/classification/"
output_directory = output_directory + classifier_name + '/' + problem + '/itr_' + str(itr) + '/'
create_directory(output_directory)



print("=======================================================================")
print("[{}] Starting Classification Experiment".format(module))
print("=======================================================================")
print("[{}] Data path: {}".format(module, data_path))
print("[{}] Output Dir: {}".format(module, output_directory))
print("[{}] Iteration: {}".format(module, itr))
print("[{}] Problem: {} | {}".format(module, data_id, problem))
print("[{}] Classifier: {}".format(module, classifier_name))
print("[{}] Config: {}".format(module,config))
print("[{}] Normalisation: {}".format(module, normalization))

#Call Datasets
print("[{}] Loading data".format(module))
X_train,y_train,X_test,y_test = create_numpy_dataset(name=problem,path=data_path)

#Create Normalizer & Normalize Data
print("[{}] X_train: {}".format(module, X_train.shape))
print("[{}] X_test: {}".format(module, X_test.shape))
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

#Normalize Labels
label_encode = LabelEncoder()
y_train_transformed = label_encode.fit_transform(y_train)
y_test_transformed = label_encode.transform(y_test)


#Load Model Config
if config is not None:
    model_kwargs = json.load(open(config))
    print("[{}] Model Args: {}".format(module,model_kwargs))


if classifier_name in ['sax', 'sax_vseg']:
    if config is None:
        clf = SAXDictionaryClassifier(save_words = True)
    else:
        clf = SAXDictionaryClassifier(**model_kwargs)
elif classifier_name =='esax':
    clf = ESAXDictionaryClassifier(**model_kwargs)
elif classifier_name =='1dsax':
    clf = OneDSAXDictionaryClassifier(**model_kwargs)
elif classifier_name =='tfsax':
    clf = TFSAXDictionaryClassifier(**model_kwargs)
elif classifier_name =='sax_dr':
    clf = SAXDRDictionaryClassifier(**model_kwargs)
elif classifier_name =='sax_vfd':
    clf = SAXVFDDictionaryClassifier(**model_kwargs)
elif classifier_name == 'sfa':
    if config is None:
        clf = SFADictionaryClassifier(save_words=True)
    else:
        clf = SFADictionaryClassifier(**model_kwargs)
elif classifier_name == 'spartan':
    clf = SPARTANClassifier(**model_kwargs)
comp_start = time.time()

fit_start = time.time()
clf.fit(X_train_transform,y_train_transformed)
fit_end = time.time()

pred_start = time.time()
model_pred = clf.predict(X_test_transform)
pred_end = time.time()

comp_end = time.time()

print(f'Fit time: {fit_end - fit_start}')
print(f'Pred time: {pred_end - pred_start}')
# model_eval = (model_pred == y_test_transformed).sum() / len(model_pred)
# print(model_eval)

if model_kwargs['metric'] != 'matching_symbol':
    results = compute_classification_metrics(y_test_transformed,model_pred)

    

    model_params = pd.DataFrame([model_kwargs])
    model_params['runtime'] = comp_end - comp_start

    results = pd.concat([results,model_params],ignore_index=False,axis=1)

    print(results)

    filename = output_directory + 'classification_results.csv'
    with open(filename, 'a') as f:
        results.to_csv(f, mode='a', header=f.tell()==0,index=False)

else:

    results_1 = compute_classification_metrics(y_test_transformed,model_pred[0])
    results_2 = compute_classification_metrics(y_test_transformed,model_pred[1])
    print("hamming: \n", results_1)
    print("symbol: \n", results_2)

    model_kwargs['metric'] = 'matching'
    model_params = pd.DataFrame([model_kwargs])
    results_1 = pd.concat([results_1,model_params],ignore_index=False,axis=1)
    
    
    model_kwargs['metric'] = 'symbol'
    model_params = pd.DataFrame([model_kwargs])
    results_2 = pd.concat([results_2,model_params],ignore_index=False,axis=1)

    filename = output_directory + 'classification_results.csv'
    with open(filename, 'a') as f:
        results_1.to_csv(f, mode='a', header=f.tell()==0,index=False)

    # filename_2 = output_directory + 'classification_symbol_results.csv'
    with open(filename, 'a') as f:
        results_2.to_csv(f, mode='a', header=f.tell()==0,index=False)



