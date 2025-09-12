#!/bin/bash

data_path='data/Univariate_ts' # change as needed
test_num=128 # full UCR archive with 128 datasets

# alpha=4, omega=2
python main.py --eval_task clustering --classifier spartan --data $data_path --config benchmark/configs/ablation_param/spartan_a4w2.json -k kmedoids -b merge -r single -i a4_w2 -t $test_num

python main.py --eval_task clustering --classifier sax --data $data_path --config benchmark/configs/ablation_param/sax_a4w2.json -k kmedoids -b merge -r single -i a4_w2 -t $test_num

python main.py --eval_task clustering --classifier sfa --data $data_path --config benchmark/configs/ablation_param/sfa_a4w2.json -k kmedoids -b merge -r single -i a4_w2 -t $test_num

# alpha=4, omega=4
python main.py --eval_task clustering --classifier spartan --data $data_path --config benchmark/configs/ablation_param/spartan_a4w4.json -k kmedoids -b merge -r single -i a4_w4 -t $test_num

python main.py --eval_task clustering --classifier sax --data $data_path --config benchmark/configs/ablation_param/sax_a4w4.json -k kmedoids -b merge -r single -i a4_w4 -t $test_num

python main.py --eval_task clustering --classifier sfa --data $data_path --config benchmark/configs/ablation_param/sfa_a4w4.json -k kmedoids -b merge -r single -i a4_w4 -t $test_num

# alpha=4, omega=6
python main.py --eval_task clustering --classifier spartan --data $data_path --config benchmark/configs/ablation_param/spartan_a4w6.json -k kmedoids -b merge -r single -i a4_w6 -t $test_num

python main.py --eval_task clustering --classifier sax --data $data_path --config benchmark/configs/ablation_param/sax_a4w6.json -k kmedoids -b merge -r single -i a4_w6 -t $test_num

python main.py --eval_task clustering --classifier sfa --data $data_path --config benchmark/configs/ablation_param/sfa_a4w6.json -k kmedoids -b merge -r single -i a4_w6 -t $test_num

# alpha=4, omega=8
python main.py --eval_task clustering --classifier spartan --data $data_path --config benchmark/configs/ablation_param/spartan_a4w8.json -k kmedoids -b merge -r single -i a4_w8 -t $test_num

python main.py --eval_task clustering --classifier sax --data $data_path --config benchmark/configs/ablation_param/sax_a4w8.json -k kmedoids -b merge -r single -i a4_w8 -t $test_num

python main.py --eval_task clustering --classifier sfa --data $data_path --config benchmark/configs/ablation_param/sfa_a4w8.json -k kmedoids -b merge -r single -i a4_w8 -t $test_num







