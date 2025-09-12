#!/bin/bash

data_path='data/Univariate_ts' # change as needed
test_num=128 # full UCR archive with 128 datasets
MinWordLen=4
MaxWordLen=10
MinAlphabetSize=4
MaxAlphabetSize=10

# single pattern
python main.py --eval_task tlb --data $data_path --alpha_max $MaxAlphabetSize --alpha_min $MinAlphabetSize --wordlen_max $MaxWordLen --wordlen_min $MinWordLen -t $test_num



