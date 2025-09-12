import argparse
import json
import os
import sys
import time
import numpy as np
import pandas as pd

from .util.tools import create_directory,compute_classification_metrics
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

from TSB_Symbolic.onennclassifier.sax_classifier import SAXDictionaryClassifier
from TSB_Symbolic.onennclassifier.sfa_classifier import SFADictionaryClassifier
from TSB_Symbolic.onennclassifier.spartan_classifier import SPARTANClassifier

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data", required=False, default="data/cbf_gen/")
parser.add_argument("-p", "--problem", required=False, default="CBF")  # see data_loader.regression_datasets
parser.add_argument("-c", "--classifier", required=False, default="sax")  # see regressor_tools.all_models
parser.add_argument("-g",'--config',required=False,default=None)
parser.add_argument("-i", "--itr", required=False, default=1)
parser.add_argument("-n", "--norm", required=False, default="zscore")  # none, standard, minmax
parser.add_argument("-s","--save_model",required=False, default=None)
parser.add_argument("-r","--skip_repeat",required=False,default=True)
parser.add_argument("-m","--dataset_num",required=False,default=1, type=int)
parser.add_argument("-w","--store_words",default=None)
parser.add_argument("-t","--downsample",default=1.0, type=float)
parser.add_argument("-e","--eval_task",default='varying_num', type=str, choices=['varying_num', 'varying_len'])
parser.add_argument("-j","--ts_num",default=100, type=int)
parser.add_argument("-k","--ts_len",default=128, type=int)


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
downsample_rate = arguments.downsample
exp_type = arguments.eval_task
ts_num = arguments.ts_num
ts_len = arguments.ts_len

# create output directory
output_directory = "output/scaling/"
output_directory = output_directory + classifier_name + '/' + problem + '/itr_' + str(itr) + '/'
create_directory(output_directory)


print("=======================================================================")
print("[{}] Starting Scaling Experiment".format(module))
print("=======================================================================")
print("[{}] Data path: {}".format(module, data_path))
print("[{}] Output Dir: {}".format(module, output_directory))
print("[{}] Iteration: {}".format(module, itr))
print("[{}] Problem: {} | {}".format(module, data_id, problem))
print("[{}] Classifier: {}".format(module, classifier_name))
print("[{}] Config: {}".format(module,config))
print("[{}] Normalization: {}".format(module, normalization))

#Call Datasets
print("[{}] Loading data".format(module))
os.makedirs("result/Section5_6/", exist_ok=True)

if exp_type == 'varying_num':
    
    X_train = np.load(os.path.join(data_path, f'cbf_train_X_1M.npy'))
    y_train = np.load(os.path.join(data_path, f'cbf_train_Y_1M.npy')).astype(int)
    X_test = np.load(os.path.join(data_path, f'cbf_test_X.npy'))
    y_test = np.load(os.path.join(data_path, f'cbf_test_Y.npy'))

    assert X_train.shape == (1e6, 128) and X_test.shape == (900, 128)
    assert y_train.shape == (1e6, )    and y_test.shape == (900, )

    X_train, y_train = X_train[:ts_num], y_train[:ts_num]
    
    if ts_num // 3 == 0:
        assert np.sum(y_train) == int(ts_num/3)*3
    if ts_num // 3 == 1:
        assert np.sum(y_train) == int(ts_num/3)*3
    if ts_num // 3 == 2:
        assert np.sum(y_train) == int(ts_num/3)*3 + 1

    import matplotlib.pyplot as plt
    id1, id2, id3 = np.random.choice(ts_num, 3)
    plt.plot(X_train[id1], label=f'ID{id1}-Class{y_train[id1]}')
    plt.plot(X_train[id2], label=f'ID{id2}-Class{y_train[id2]}')
    plt.plot(X_train[id3], label=f'ID{id3}-Class{y_train[id3]}')
    plt.legend()
    plt.savefig(f"result/Section5_6/showcbf_varyingnum_{ts_num}_ID{id1}_{id2}_{id3}.png", dpi=300)
    plt.show()

else:

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

    if rep_num <= 5:
        id1, id2, id3 = np.random.choice(900, 3)
        plt.plot(X_train[id1], label=f'ID{id1}-Class{y_train[id1]}')
        plt.plot(X_train[id2], label=f'ID{id2}-Class{y_train[id2]}')
        plt.plot(X_train[id3], label=f'ID{id3}-Class{y_train[id3]}')
        plt.legend()
        plt.savefig(f"result/Section5_6/showcbf_varylen_train_{ts_len}_ID{id1}_{id2}_{id3}.png", dpi=300)
        plt.show()
        plt.close()

        plt.plot(X_test[id1], label=f'ID{id1}-Class{y_test[id1]}')
        plt.plot(X_test[id2], label=f'ID{id2}-Class{y_test[id2]}')
        plt.plot(X_test[id3], label=f'ID{id3}-Class{y_test[id3]}')
        plt.legend()
        plt.savefig(f"result/Section5_6/showcbf_varylen_test_{ts_len}_ID{id1}_{id2}_{id3}.png", dpi=300)
        plt.show()
        plt.close()

#Create Normalizer & Normalize Data
print("[{}] X_train: {}".format(module, X_train.shape))
print("[{}] X_test: {}".format(module, X_test.shape))


train_means = np.mean(X_train,axis=1,keepdims=True)
train_stds = np.std(X_train,axis=1,keepdims=True)
test_means = np.mean(X_test,axis=1,keepdims=True)
test_stds = np.std(X_test,axis=1,keepdims=True)

train_stds[np.abs(train_stds) < 1e-6] = 1
test_stds[np.abs(test_stds) < 1e-6] = 1

X_train_transform = (X_train - train_means) / train_stds
X_test_transform = (X_test - test_means) / test_stds

#Normalize Labels
label_encode = LabelEncoder()
y_train_transformed = label_encode.fit_transform(y_train)
y_test_transformed = label_encode.transform(y_test)

#Load Model Config
if config is not None:
    model_kwargs = json.load(open(config))

if classifier_name == 'sax':
    if config is None:
        clf = SAXDictionaryClassifier(save_words = True)
    else:
        clf = SAXDictionaryClassifier(**model_kwargs)
elif classifier_name == 'sfa':
    if config is None:
        clf = SFADictionaryClassifier(save_words=True)
    else:
        clf = SFADictionaryClassifier(**model_kwargs)
elif classifier_name == 'spartan':
    model_kwargs['downsample'] = downsample_rate
    clf = SPARTANClassifier(**model_kwargs)

print("[{}] Model Args: {}".format(module,model_kwargs))

if classifier_name == 'spartan' and clf.downsample < 1.0:

    repeat_num = 5
else:
    repeat_num = 1

avg_runtime = 0.0
avg_results = pd.DataFrame(columns=['acc','precision','recall','f1'])
for itr in range(repeat_num):

    # initialize the symbolic model
    if classifier_name == 'sax':
        if config is None:
            clf = SAXDictionaryClassifier(save_words = True)
        else:
            clf = SAXDictionaryClassifier(**model_kwargs)
    elif classifier_name == 'sfa':
        if config is None:
            clf = SFADictionaryClassifier(save_words=True)
        else:
            clf = SFADictionaryClassifier(**model_kwargs)
    elif classifier_name == 'spartan':
        # model_kwargs['downsample'] = downsample_rate
        clf = SPARTANClassifier(**model_kwargs)


    comp_start = time.time()

    fit_start = time.time()
    clf.fit(X_train_transform,y_train_transformed)
    fit_end = time.time()

    pred_start = time.time()
    model_pred = clf.predict(X_test_transform)
    pred_end = time.time()

    comp_end = time.time()

    avg_runtime += comp_end - comp_start
    
    results = compute_classification_metrics(y_test_transformed,model_pred)
    avg_results = pd.concat([avg_results, results], ignore_index=True)

        
print(f'Fit time: {(fit_end - fit_start):.4f}s')
print(f'Pred time: {(pred_end - pred_start):.4f}s')

print(avg_results)
avg_results = avg_results.mean().to_frame().T
model_params = pd.DataFrame([model_kwargs])
model_params['runtime'] = avg_runtime / repeat_num

final_results = pd.concat([avg_results,model_params],ignore_index=False,axis=1)

print(final_results)

filename = output_directory + 'classification_results.csv'
with open(filename, 'a') as f:
    final_results.to_csv(f, mode='a', header=f.tell()==0,index=False)


