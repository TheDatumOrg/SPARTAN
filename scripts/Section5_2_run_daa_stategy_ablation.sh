#!/bin/bash

data_path='data/Univariate_ts' # change as needed
test_num=128 # full UCR archive with 128 datasets

# different alphabet allocation strategies (single pattern)

python main.py --eval_task classification --classifier spartan --data $data_path --config benchmark/configs/spartan_daa_strategy/spartan_a4w8_naiveC.json -i a4_w8_naiveC -t $test_num

python main.py --eval_task classification --classifier spartan --data $data_path --config benchmark/configs/spartan_daa_strategy/spartan_a4w8_naiveDAA.json -i a4_w8_naiveDAA -t $test_num

python main.py --eval_task classification --classifier spartan --data $data_path --config benchmark/configs/spartan_daa_strategy/spartan_a4w8_woDAA.json -i a4_w8_woDAA -t $test_num

# different alphabet allocation strategies (bag of pattern)

python main.py --eval_task classification --classifier spartan --data $data_path --config benchmark/configs/spartan_daa_strategy/spartan_a4w4win0.05_naiveC.json -i a4_w4_win0.05_naiveC -t $test_num

python main.py --eval_task classification --classifier spartan --data $data_path --config benchmark/configs/spartan_daa_strategy/spartan_a4w4win0.05_naiveDAA.json -i a4_w4_win0.05_naiveDAA -t $test_num

python main.py --eval_task classification --classifier spartan --data $data_path --config benchmark/configs/spartan_daa_strategy/spartan_a4w4win0.05_woDAA.json -i a4_w4_win0.05_woDAA -t $test_num






