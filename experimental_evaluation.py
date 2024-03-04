from tools import univariate

import argparse

import os
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data", required=False, default="../TSC-Benchmark/tscbench/data/Univariate_ts/")
parser.add_argument("-c", "--classifier", required=False, default="spartan")  # see regressor_tools.all_models
parser.add_argument("-g",'--config',required=False,default='./configs/sfa/sfa_single_word.json')
parser.add_argument("-i", "--itr", required=False, default=3)
parser.add_argument("-n", "--norm", required=False, default="zscore")  # none, standard, minmax
parser.add_argument("-s","--save_model",required=False, default=None)


arguments = parser.parse_args()

data_path = arguments.data
classifier_name = arguments.classifier
normalization = arguments.norm
itr = arguments.itr
config = arguments.config

dset_info = pd.read_csv('summaryUnivariate.csv')
dset_info = dset_info.sort_values(by=['numTrainCases','numTestCases'])

print(os.cpu_count())

# skip_dataset = ['ArrowHead','Beef','BeetleFly','Meat','OliveOil','Lightning2']
for i in range(dset_info.shape[0]):
    dataset = dset_info['problem'].iloc[i]
    call_string = 'time python3 train.py --data {} --classifier {} --norm {} --problem {} --itr {} --config {}'.format(data_path,classifier_name,normalization,dataset,itr,config)

    os.system(call_string)