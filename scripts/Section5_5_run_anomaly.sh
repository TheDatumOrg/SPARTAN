#!/bin/bash

data_path='data/TSB-UAD-Public' # change as needed
test_num=0
dataset_list='dataset_list_full'


python main.py --eval_task anomaly --classifier spartan --data $data_path --config benchmark/configs/spartan_anomaly.json -i a16_w16_win100 -t $test_num -dl $dataset_list

python main.py --eval_task anomaly --classifier sax --data $data_path --config benchmark/configs/sax_anomaly.json -i a16_w16_win100 -t $test_num -dl $dataset_list

python main.py --eval_task anomaly --classifier sfa --data $data_path --config benchmark/configs/sax_anomaly.json -i a16_w16_win100 -t $test_num -dl $dataset_list

