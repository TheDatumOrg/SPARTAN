#!/bin/bash

data_path='data/Univariate_ts' # change as needed
test_num=128 # full UCR archive with 128 datasets

# alpha=4, omega=2
python main.py --eval_task classification --classifier sax --data $data_path --config benchmark/configs/ablation_param/sax_a4w2.json -i a4_w2 -t $test_num

python main.py --eval_task classification --classifier sfa --data $data_path --config benchmark/configs/ablation_param/sfa_a4w2.json -i a4_w2 -t $test_num

python main.py --eval_task classification --classifier spartan --data $data_path --config benchmark/configs/ablation_param/spartan_a4w2.json -i a4_w2 -t $test_num

# alpha=4, omega=4
python main.py --eval_task classification --classifier sax --data $data_path --config benchmark/configs/ablation_param/sax_a4w4.json -i a4_w4 -t $test_num

python main.py --eval_task classification --classifier sfa --data $data_path --config benchmark/configs/ablation_param/sfa_a4w4.json -i a4_w4 -t $test_num

python main.py --eval_task classification --classifier spartan --data $data_path --config benchmark/configs/ablation_param/spartan_a4w4.json -i a4_w4 -t $test_num

# alpha=4, omega=12
python main.py --eval_task classification --classifier sax --data $data_path --config benchmark/configs/ablation_param/sax_a4w12.json -i a4_w12 -t $test_num

python main.py --eval_task classification --classifier sfa --data $data_path --config benchmark/configs/ablation_param/sfa_a4w12.json -i a4_w12 -t $test_num

python main.py --eval_task classification --classifier spartan --data $data_path --config benchmark/configs/ablation_param/spartan_a4w12.json -i a4_w12 -t $test_num








