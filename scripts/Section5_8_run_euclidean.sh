#!/bin/bash

data_path='data/Univariate_ts' # change as needed
test_num=0 # full UCR archive with 128 datasets
dataset_list='dataset_list_full' # full TSB-UAD archive

# classification
python main.py --eval_task classification --classifier euclidean --data $data_path -i euc_1nn -t $test_num

# clustering
python main.py --eval_task clustering --classifier euclidean --data $data_path -k kmeans -b merge -r single -i euc_kmeans -t $test_num

# anomaly detection
python main.py --eval_task anomaly --classifier euclidean --data 'data/TSB-UAD-Public' -dl $dataset_list -i euc_win100 -t $test_num