#!/bin/bash

data_path='data/Univariate_ts' # change as needed
test_num=128 # full UCR archive with 128 datasets

# benchmark baseline methods with **constrained** bit-budget
python main.py --eval_task classification --classifier sax --data $data_path --config benchmark/configs/saxvariant_single_pattern_constrained.json -i a4_w12 -t $test_num

python main.py --eval_task classification --classifier esax --data $data_path --config benchmark/configs/saxvariant_single_pattern_constrained.json -i a4_w12 -t $test_num

python main.py --eval_task classification --classifier sax_dr --data $data_path --config benchmark/configs/saxvariant_single_pattern_constrained.json -i a4_w12 -t $test_num

python main.py --eval_task classification --classifier tfsax --data $data_path --config benchmark/configs/saxvariant_single_pattern_constrained.json -i a4_w12 -t $test_num

python main.py --eval_task classification --classifier 1dsax --data $data_path --config benchmark/configs/saxvariant_single_pattern_constrained.json -i a4_w12 -t $test_num

python main.py --eval_task classification --classifier sax_vfd --data $data_path --config benchmark/configs/saxvariant_single_pattern_constrained.json -i a4_w12 -t $test_num

python main.py --eval_task classification --classifier sfa --data $data_path --config benchmark/configs/sfa_single_pattern_constrained.json -i a4_w12 -t $test_num



# benchmark baseline methods with **unconstrained** bit-budget
python main.py --eval_task classification --classifier esax --data $data_path --config benchmark/configs/esax_single_pattern_unconstrained.json -i a4_w36 -t $test_num

python main.py --eval_task classification --classifier sax_dr --data $data_path --config benchmark/configs/sax_dr_single_pattern_unconstrained.json -i a4_w24 -t $test_num

python main.py --eval_task classification --classifier tfsax --data $data_path --config benchmark/configs/tfsax_single_pattern_unconstrained.json -i a4_w24 -t $test_num

python main.py --eval_task classification --classifier 1dsax --data $data_path --config benchmark/configs/1dsax_single_pattern_unconstrained.json -i a4_w24 -t $test_num

python main.py --eval_task classification --classifier sax_vfd --data $data_path --config benchmark/configs/1dsax_single_pattern_unconstrained.json -i a4_w48 -t $test_num



