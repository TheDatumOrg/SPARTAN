#!/bin/bash

data_path='data/Univariate_ts' # change as needed
test_num=128 # full UCR archive with 128 datasets

# single pattern
python main.py --eval_task classification --classifier sax --data $data_path --config benchmark/configs/sax_single_pattern.json -i a4_w8 -t $test_num

python main.py --eval_task classification --classifier sfa --data $data_path --config benchmark/configs/sfa_single_pattern.json -i a4_w8 -t $test_num

python main.py --eval_task classification --classifier spartan --data $data_path --config benchmark/configs/spartan_single_pattern.json -i a4_w8 -t $test_num

# bag of patterns
python main.py --eval_task classification --classifier sax --data $data_path --config benchmark/configs/sax_bag_of_patterns.json -i a4_w4_win0.05 -t $test_num

python main.py --eval_task classification --classifier sfa --data $data_path --config benchmark/configs/sfa_bag_of_patterns.json -i a4_w4_win0.05 -t $test_num

python main.py --eval_task classification --classifier spartan --data $data_path --config benchmark/configs/spartan_bag_of_patterns.json -i a4_w4_win0.05 -t $test_num




