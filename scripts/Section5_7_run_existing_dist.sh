#!/bin/bash

data_path='data/Univariate_ts' # change as needed
test_num=128 # full UCR archive with 128 datasets

# running on existing distance
python main.py --eval_task classification --classifier sax --data $data_path --config benchmark/configs/ablation_dist/sax_mindist.json -i a4_w8_exist_dist -t $test_num

python main.py --eval_task classification --classifier sfa --data $data_path --config benchmark/configs/ablation_dist/sfa_mindist.json -i a4_w8_exist_dist -t $test_num

python main.py --eval_task classification --classifier spartan --data $data_path --config benchmark/configs/ablation_dist/spartan_mindist.json -i a4_w8_exist_dist -t $test_num

python main.py --eval_task classification --classifier sax_dr --data $data_path --config benchmark/configs/ablation_dist/sax_dr_mindist.json -i a4_w8_exist_dist -t $test_num

python main.py --eval_task classification --classifier sax_dr --data $data_path --config benchmark/configs/ablation_dist/sax_dr_symbolicl1.json -i a4_w8 -t $test_num






