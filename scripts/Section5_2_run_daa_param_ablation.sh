#!/bin/bash

data_path='data/Univariate_ts' # change as needed
test_num=128 # full UCR archive with 128 datasets

# single pattern
python main.py --eval_task classification --classifier spartan --data $data_path --config benchmark/configs/spartan_daa_param/spartan_a4w8_lamda0.15.json -i a4_w8_lambda0.15 -t $test_num

python main.py --eval_task classification --classifier spartan --data $data_path --config benchmark/configs/spartan_daa_param/spartan_a4w8_lamda0.2.json -i a4_w8_lambda0.2 -t $test_num

python main.py --eval_task classification --classifier spartan --data $data_path --config benchmark/configs/spartan_daa_param/spartan_a4w8_lamda0.3.json -i a4_w8_lambda0.3 -t $test_num

python main.py --eval_task classification --classifier spartan --data $data_path --config benchmark/configs/spartan_daa_param/spartan_a4w8_lamda0.4.json -i a4_w8_lambda0.4 -t $test_num

python main.py --eval_task classification --classifier spartan --data $data_path --config benchmark/configs/spartan_daa_param/spartan_a4w8_lamda0.5.json -i a4_w8_lambda0.5 -t $test_num

python main.py --eval_task classification --classifier spartan --data $data_path --config benchmark/configs/spartan_daa_param/spartan_a4w8_lamda0.8.json -i a4_w8_lambda0.8 -t $test_num

python main.py --eval_task classification --classifier spartan --data $data_path --config benchmark/configs/spartan_daa_param/spartan_a4w8_lamda1.0.json -i a4_w8_lambda1.0 -t $test_num

# bag of patterns

python main.py --eval_task classification --classifier spartan --data $data_path --config benchmark/configs/spartan_daa_param/spartan_a4w4win0.05_lamda0.15.json -i a4_w4_win0.05_lamda0.15 -t $test_num

python main.py --eval_task classification --classifier spartan --data $data_path --config benchmark/configs/spartan_daa_param/spartan_a4w4win0.05_lamda0.2.json -i a4_w4_win0.05_lamda0.2 -t $test_num

python main.py --eval_task classification --classifier spartan --data $data_path --config benchmark/configs/spartan_daa_param/spartan_a4w4win0.05_lamda0.3.json -i a4_w4_win0.05_lamda0.3 -t $test_num

python main.py --eval_task classification --classifier spartan --data $data_path --config benchmark/configs/spartan_daa_param/spartan_a4w4win0.05_lamda0.4.json -i a4_w4_win0.05_lamda0.4 -t $test_num

python main.py --eval_task classification --classifier spartan --data $data_path --config benchmark/configs/spartan_daa_param/spartan_a4w4win0.05_lamda0.5.json -i a4_w4_win0.05_lamda0.5 -t $test_num

python main.py --eval_task classification --classifier spartan --data $data_path --config benchmark/configs/spartan_daa_param/spartan_a4w4win0.05_lamda0.8.json -i a4_w4_win0.05_lamda0.8 -t $test_num

python main.py --eval_task classification --classifier spartan --data $data_path --config benchmark/configs/spartan_daa_param/spartan_a4w4win0.05_lamda1.0.json -i a4_w4_win0.05_lamda1.0 -t $test_num




