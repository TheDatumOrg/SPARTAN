#!/bin/bash

data_path='data/Univariate_ts' # change as needed
test_num=128 # full UCR archive with 128 datasets

# test spartan family
python main.py --eval_task classification --classifier spartan --data $data_path --config benchmark/configs/spartan_scaling/spartan_full_a4w8.json -i a4_w8_full -t $test_num

python main.py --eval_task classification --classifier spartan --data $data_path --config benchmark/configs/spartan_scaling/spartan_rand_a4w8.json -i a4_w8_rand -t $test_num

python main.py --eval_task classification --classifier spartan --data $data_path --config benchmark/configs/spartan_scaling/spartan_sample_a4w8.json -i a4_w8_sample -t $test_num --downsample 0.2




