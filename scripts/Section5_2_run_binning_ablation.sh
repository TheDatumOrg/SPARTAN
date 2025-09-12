#!/bin/bash

data_path='data/Univariate_ts' # change as needed
test_num=128 # full UCR archive with 128 datasets

# single pattern

python main.py --eval_task classification --classifier sfa --data $data_path --config benchmark/configs/ablation_binning/sfa_single_pattern_equiwidth.json -i a4_w8_equiwidth -t $test_num &

python main.py --eval_task classification --classifier spartan --data $data_path --config benchmark/configs/ablation_binning/spartan_single_pattern_equiwidth.json -i a4_w8_equiwidth -t $test_num &

wait


