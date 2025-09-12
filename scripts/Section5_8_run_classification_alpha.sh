#!/bin/bash

data_path='data/Univariate_ts' # change as needed
test_num=128 # full UCR archive with 128 datasets

# alpha=4, omega=4
python main.py --eval_task classification --classifier sax --data $data_path --config benchmark/configs/ablation_param/sax_a4w4.json -i a4_w4 -t $test_num

python main.py --eval_task classification --classifier sfa --data $data_path --config benchmark/configs/ablation_param/sfa_a4w4.json -i a4_w4 -t $test_num

python main.py --eval_task classification --classifier spartan --data $data_path --config benchmark/configs/ablation_param/spartan_a4w4.json -i a4_w4 -t $test_num

# alpha=6, omega=4
python main.py --eval_task classification --classifier sax --data $data_path --config benchmark/configs/ablation_param/sax_a6w4.json -i a6_w4 -t $test_num

python main.py --eval_task classification --classifier sfa --data $data_path --config benchmark/configs/ablation_param/sfa_a6w4.json -i a6_w4 -t $test_num

python main.py --eval_task classification --classifier spartan --data $data_path --config benchmark/configs/ablation_param/spartan_a6w4.json -i a6_w4 -t $test_num

# alpha=8, omega=4
python main.py --eval_task classification --classifier sax --data $data_path --config benchmark/configs/ablation_param/sax_a8w4.json -i a8_w4 -t $test_num

python main.py --eval_task classification --classifier sfa --data $data_path --config benchmark/configs/ablation_param/sfa_a8w4.json -i a8_w4 -t $test_num

python main.py --eval_task classification --classifier spartan --data $data_path --config benchmark/configs/ablation_param/spartan_a8w4.json -i a8_w4 -t $test_num

# alpha=10, omega=4
python main.py --eval_task classification --classifier sax --data $data_path --config benchmark/configs/ablation_param/sax_a10w4.json -i a10_w4 -t $test_num

python main.py --eval_task classification --classifier sfa --data $data_path --config benchmark/configs/ablation_param/sfa_a10w4.json -i a10_w4 -t $test_num

python main.py --eval_task classification --classifier spartan --data $data_path --config benchmark/configs/ablation_param/spartan_a10w4.json -i a10_w4 -t $test_num







