#!/bin/bash

data_path='data/Univariate_ts' # change as needed
test_num=128 # skip two large datasets due to OOM issue

# single pattern
python main.py --eval_task clustering --classifier spartan --data $data_path --config benchmark/configs/clustering/spartan_single_pattern.json -k kmedoids -b merge -r single -i a4_w8 -t $test_num

python main.py --eval_task clustering --classifier sax --data $data_path --config benchmark/configs/clustering/sax_single_pattern.json -k kmedoids -b merge -r single -i a4_w8 -t $test_num

python main.py --eval_task clustering --classifier sfa --data $data_path --config benchmark/configs/clustering/sfa_single_pattern.json -k kmedoids -b merge -r single -i a4_w8 -t $test_num


# bag of pattern
python main.py --eval_task clustering --classifier spartan --data $data_path --config benchmark/configs/clustering/spartan_bag_of_patterns.json -k symb_kmeans -b merge -r bop -i a4_w4_win0.05 -t $test_num

python main.py --eval_task clustering --classifier sax --data $data_path --config benchmark/configs/clustering/sax_bag_of_patterns.json -k symb_kmeans -b merge -r bop -i a4_w4_win0.05 -t $test_num

python main.py --eval_task clustering --classifier sfa --data $data_path --config benchmark/configs/clustering/sfa_bag_of_patterns.json -k symb_kmeans -b merge -r bop -i a4_w4_win0.05 -t $test_num

