#!/bin/bash

data_dir="data/cbf_gen"

# SFA scaling with varying length
python -m scripts.Section5_6_scaling --classifier sfa --config benchmark/configs/sfa_single_pattern.json -d $data_dir --eval_task varying_len --itr varylen_128000 --ts_len 128000

python -m scripts.Section5_6_scaling --classifier sfa --config benchmark/configs/sfa_single_pattern.json -d $data_dir --eval_task varying_len --itr varylen_64000 --ts_len 64000

python -m scripts.Section5_6_scaling --classifier sfa --config benchmark/configs/sfa_single_pattern.json -d $data_dir --eval_task varying_len --itr varylen_12800 --ts_len 12800

python -m scripts.Section5_6_scaling --classifier sfa --config benchmark/configs/sfa_single_pattern.json -d $data_dir --eval_task varying_len --itr varylen_6400 --ts_len 6400

python -m scripts.Section5_6_scaling --classifier sfa --config benchmark/configs/sfa_single_pattern.json -d $data_dir --eval_task varying_len --itr varylen_1280 --ts_len 1280

python -m scripts.Section5_6_scaling --classifier sfa --config benchmark/configs/sfa_single_pattern.json -d $data_dir --eval_task varying_len --itr varylen_640 --ts_len 640

python -m scripts.Section5_6_scaling --classifier sfa --config benchmark/configs/sfa_single_pattern.json -d $data_dir --eval_task varying_len --itr varylen_128 --ts_len 128

# SFA scaling with varying number
python -m scripts.Section5_6_scaling --classifier sfa --config benchmark/configs/sfa_single_pattern.json -d $data_dir --eval_task varying_num --itr varynum_1000 --ts_num 1000

python -m scripts.Section5_6_scaling --classifier sfa --config benchmark/configs/sfa_single_pattern.json -d $data_dir --eval_task varying_num --itr varynum_5000 --ts_num 5000

python -m scripts.Section5_6_scaling --classifier sfa --config benchmark/configs/sfa_single_pattern.json -d $data_dir --eval_task varying_num --itr varynum_10000 --ts_num 10000

python -m scripts.Section5_6_scaling --classifier sfa --config benchmark/configs/sfa_single_pattern.json -d $data_dir --eval_task varying_num --itr varynum_50000 --ts_num 50000

python -m scripts.Section5_6_scaling --classifier sfa --config benchmark/configs/sfa_single_pattern.json -d $data_dir --eval_task varying_num --itr varynum_100000 --ts_num 100000

python -m scripts.Section5_6_scaling --classifier sfa --config benchmark/configs/sfa_single_pattern.json -d $data_dir --eval_task varying_num --itr varynum_500000 --ts_num 500000

python -m scripts.Section5_6_scaling --classifier sfa --config benchmark/configs/sfa_single_pattern.json -d $data_dir --eval_task varying_num --itr varynum_1000000 --ts_num 1000000


# SAX scaling with varying length
python -m scripts.Section5_6_scaling --classifier sax --config benchmark/configs/sax_single_pattern.json -d $data_dir --eval_task varying_len --itr varylen_128000 --ts_len 128000

python -m scripts.Section5_6_scaling --classifier sax --config benchmark/configs/sax_single_pattern.json -d $data_dir --eval_task varying_len --itr varylen_64000 --ts_len 64000

python -m scripts.Section5_6_scaling --classifier sax --config benchmark/configs/sax_single_pattern.json -d $data_dir --eval_task varying_len --itr varylen_12800 --ts_len 12800

python -m scripts.Section5_6_scaling --classifier sax --config benchmark/configs/sax_single_pattern.json -d $data_dir --eval_task varying_len --itr varylen_6400 --ts_len 6400

python -m scripts.Section5_6_scaling --classifier sax --config benchmark/configs/sax_single_pattern.json -d $data_dir --eval_task varying_len --itr varylen_1280 --ts_len 1280

python -m scripts.Section5_6_scaling --classifier sax --config benchmark/configs/sax_single_pattern.json -d $data_dir --eval_task varying_len --itr varylen_640 --ts_len 640

python -m scripts.Section5_6_scaling --classifier sax --config benchmark/configs/sax_single_pattern.json -d $data_dir --eval_task varying_len --itr varylen_128 --ts_len 128

# SAX scaling with varying number
python -m scripts.Section5_6_scaling --classifier sax --config benchmark/configs/sax_single_pattern.json -d $data_dir --eval_task varying_num --itr varynum_1000 --ts_num 1000

python -m scripts.Section5_6_scaling --classifier sax --config benchmark/configs/sax_single_pattern.json -d $data_dir --eval_task varying_num --itr varynum_5000 --ts_num 5000

python -m scripts.Section5_6_scaling --classifier sax --config benchmark/configs/sax_single_pattern.json -d $data_dir --eval_task varying_num --itr varynum_10000 --ts_num 10000

python -m scripts.Section5_6_scaling --classifier sax --config benchmark/configs/sax_single_pattern.json -d $data_dir --eval_task varying_num --itr varynum_50000 --ts_num 50000

python -m scripts.Section5_6_scaling --classifier sax --config benchmark/configs/sax_single_pattern.json -d $data_dir --eval_task varying_num --itr varynum_100000 --ts_num 100000

python -m scripts.Section5_6_scaling --classifier sax --config benchmark/configs/sax_single_pattern.json -d $data_dir --eval_task varying_num --itr varynum_500000 --ts_num 500000

python -m scripts.Section5_6_scaling --classifier sax --config benchmark/configs/sax_single_pattern.json -d $data_dir --eval_task varying_num --itr varynum_1000000 --ts_num 1000000



# SPARTAN scaling with varying length
python -m scripts.Section5_6_scaling --classifier spartan --config benchmark/configs/spartan_scaling/spartan_full_a4w8.json -d $data_dir --eval_task varying_len --itr varylen_128000 --ts_len 128000

python -m scripts.Section5_6_scaling --classifier spartan --config benchmark/configs/spartan_scaling/spartan_full_a4w8.json -d $data_dir --eval_task varying_len --itr varylen_64000 --ts_len 64000

python -m scripts.Section5_6_scaling --classifier spartan --config benchmark/configs/spartan_scaling/spartan_full_a4w8.json -d $data_dir --eval_task varying_len --itr varylen_12800 --ts_len 12800

python -m scripts.Section5_6_scaling --classifier spartan --config benchmark/configs/spartan_scaling/spartan_full_a4w8.json -d $data_dir --eval_task varying_len --itr varylen_6400 --ts_len 6400

python -m scripts.Section5_6_scaling --classifier spartan --config benchmark/configs/spartan_scaling/spartan_full_a4w8.json -d $data_dir --eval_task varying_len --itr varylen_1280 --ts_len 1280

python -m scripts.Section5_6_scaling --classifier spartan --config benchmark/configs/spartan_scaling/spartan_full_a4w8.json -d $data_dir --eval_task varying_len --itr varylen_640 --ts_len 640

python -m scripts.Section5_6_scaling --classifier spartan --config benchmark/configs/spartan_scaling/spartan_full_a4w8.json -d $data_dir --eval_task varying_len --itr varylen_128 --ts_len 128

# SPARTAN scaling with varying number
python -m scripts.Section5_6_scaling --classifier spartan --config benchmark/configs/spartan_scaling/spartan_full_a4w8.json -d $data_dir --eval_task varying_num --itr varynum_1000 --ts_num 1000

python -m scripts.Section5_6_scaling --classifier spartan --config benchmark/configs/spartan_scaling/spartan_full_a4w8.json -d $data_dir --eval_task varying_num --itr varynum_5000 --ts_num 5000

python -m scripts.Section5_6_scaling --classifier spartan --config benchmark/configs/spartan_scaling/spartan_full_a4w8.json -d $data_dir --eval_task varying_num --itr varynum_10000 --ts_num 10000

python -m scripts.Section5_6_scaling --classifier spartan --config benchmark/configs/spartan_scaling/spartan_full_a4w8.json -d $data_dir --eval_task varying_num --itr varynum_50000 --ts_num 50000

python -m scripts.Section5_6_scaling --classifier spartan --config benchmark/configs/spartan_scaling/spartan_full_a4w8.json -d $data_dir --eval_task varying_num --itr varynum_100000 --ts_num 100000

python -m scripts.Section5_6_scaling --classifier spartan --config benchmark/configs/spartan_scaling/spartan_full_a4w8.json -d $data_dir --eval_task varying_num --itr varynum_500000 --ts_num 500000

python -m scripts.Section5_6_scaling --classifier spartan --config benchmark/configs/spartan_scaling/spartan_full_a4w8.json -d $data_dir --eval_task varying_num --itr varynum_1000000 --ts_num 1000000


# SPARTAN-R scaling with varying number
python -m scripts.Section5_6_scaling --classifier spartan --config benchmark/configs/spartan_scaling/spartan_rand_a4w8.json -d $data_dir --eval_task varying_num --itr varynum_1000_R --ts_num 1000 --downsample 0.05

python -m scripts.Section5_6_scaling --classifier spartan --config benchmark/configs/spartan_scaling/spartan_rand_a4w8.json -d $data_dir --eval_task varying_num --itr varynum_5000_R --ts_num 5000 --downsample 0.05

python -m scripts.Section5_6_scaling --classifier spartan --config benchmark/configs/spartan_scaling/spartan_rand_a4w8.json -d $data_dir --eval_task varying_num --itr varynum_10000_R --ts_num 10000 --downsample 0.05

python -m scripts.Section5_6_scaling --classifier spartan --config benchmark/configs/spartan_scaling/spartan_rand_a4w8.json -d $data_dir --eval_task varying_num --itr varynum_50000_R --ts_num 50000 --downsample 0.05

python -m scripts.Section5_6_scaling --classifier spartan --config benchmark/configs/spartan_scaling/spartan_rand_a4w8.json -d $data_dir --eval_task varying_num --itr varynum_100000_R --ts_num 100000 --downsample 0.05

python -m scripts.Section5_6_scaling --classifier spartan --config benchmark/configs/spartan_scaling/spartan_rand_a4w8.json -d $data_dir --eval_task varying_num --itr varynum_500000_R --ts_num 500000 --downsample 0.05

python -m scripts.Section5_6_scaling --classifier spartan --config benchmark/configs/spartan_scaling/spartan_rand_a4w8.json -d $data_dir --eval_task varying_num --itr varynum_1000000_R --ts_num 1000000 --downsample 0.05

# SPARTAN-R scaling with varying length
python -m scripts.Section5_6_scaling --classifier spartan --config benchmark/configs/spartan_scaling/spartan_rand_a4w8.json -d $data_dir --eval_task varying_len --itr varylen_128000_R --ts_len 128000 --downsample 0.05

python -m scripts.Section5_6_scaling --classifier spartan --config benchmark/configs/spartan_scaling/spartan_rand_a4w8.json -d $data_dir --eval_task varying_len --itr varylen_64000_R --ts_len 64000 --downsample 0.05

python -m scripts.Section5_6_scaling --classifier spartan --config benchmark/configs/spartan_scaling/spartan_rand_a4w8.json -d $data_dir --eval_task varying_len --itr varylen_12800_R --ts_len 12800 --downsample 0.05

python -m scripts.Section5_6_scaling --classifier spartan --config benchmark/configs/spartan_scaling/spartan_rand_a4w8.json -d $data_dir --eval_task varying_len --itr varylen_6400_R --ts_len 6400 --downsample 0.05

python -m scripts.Section5_6_scaling --classifier spartan --config benchmark/configs/spartan_scaling/spartan_rand_a4w8.json -d $data_dir --eval_task varying_len --itr varylen_1280_R --ts_len 1280 --downsample 0.05

python -m scripts.Section5_6_scaling --classifier spartan --config benchmark/configs/spartan_scaling/spartan_rand_a4w8.json -d $data_dir --eval_task varying_len --itr varylen_640_R --ts_len 640 --downsample 0.05

python -m scripts.Section5_6_scaling --classifier spartan --config benchmark/configs/spartan_scaling/spartan_rand_a4w8.json -d $data_dir --eval_task varying_len --itr varylen_128_R --ts_len 128 --downsample 0.05

# SPARTAN-S scaling with varying number
python -m scripts.Section5_6_scaling --classifier spartan --config benchmark/configs/spartan_scaling/spartan_sample_a4w8.json -d $data_dir --eval_task varying_num --itr varynum_1000_S --ts_num 1000 --downsample 0.05

python -m scripts.Section5_6_scaling --classifier spartan --config benchmark/configs/spartan_scaling/spartan_sample_a4w8.json -d $data_dir --eval_task varying_num --itr varynum_5000_S --ts_num 5000 --downsample 0.05

python -m scripts.Section5_6_scaling --classifier spartan --config benchmark/configs/spartan_scaling/spartan_sample_a4w8.json -d $data_dir --eval_task varying_num --itr varynum_10000_S --ts_num 10000 --downsample 0.05

python -m scripts.Section5_6_scaling --classifier spartan --config benchmark/configs/spartan_scaling/spartan_sample_a4w8.json -d $data_dir --eval_task varying_num --itr varynum_50000_S --ts_num 50000 --downsample 0.05

python -m scripts.Section5_6_scaling --classifier spartan --config benchmark/configs/spartan_scaling/spartan_sample_a4w8.json -d $data_dir --eval_task varying_num --itr varynum_100000_S --ts_num 100000 --downsample 0.05

python -m scripts.Section5_6_scaling --classifier spartan --config benchmark/configs/spartan_scaling/spartan_sample_a4w8.json -d $data_dir --eval_task varying_num --itr varynum_500000_S --ts_num 500000 --downsample 0.05

python -m scripts.Section5_6_scaling --classifier spartan --config benchmark/configs/spartan_scaling/spartan_sample_a4w8.json -d $data_dir --eval_task varying_num --itr varynum_1000000_S --ts_num 1000000 --downsample 0.05


# SPARTAN-S scaling with varying length
python -m scripts.Section5_6_scaling --classifier spartan --config benchmark/configs/spartan_scaling/spartan_sample_a4w8.json -d $data_dir --eval_task varying_len --itr varylen_128000_S --ts_len 128000 --downsample 0.05

python -m scripts.Section5_6_scaling --classifier spartan --config benchmark/configs/spartan_scaling/spartan_sample_a4w8.json -d $data_dir --eval_task varying_len --itr varylen_64000_S --ts_len 64000 --downsample 0.05

python -m scripts.Section5_6_scaling --classifier spartan --config benchmark/configs/spartan_scaling/spartan_sample_a4w8.json -d $data_dir --eval_task varying_len --itr varylen_12800_S --ts_len 12800 --downsample 0.05

python -m scripts.Section5_6_scaling --classifier spartan --config benchmark/configs/spartan_scaling/spartan_sample_a4w8.json -d $data_dir --eval_task varying_len --itr varylen_6400_S --ts_len 6400 --downsample 0.05

python -m scripts.Section5_6_scaling --classifier spartan --config benchmark/configs/spartan_scaling/spartan_sample_a4w8.json -d $data_dir --eval_task varying_len --itr varylen_1280_S --ts_len 1280 --downsample 0.05

python -m scripts.Section5_6_scaling --classifier spartan --config benchmark/configs/spartan_scaling/spartan_sample_a4w8.json -d $data_dir --eval_task varying_len --itr varylen_640_S --ts_len 640 --downsample 0.05

python -m scripts.Section5_6_scaling --classifier spartan --config benchmark/configs/spartan_scaling/spartan_sample_a4w8.json -d $data_dir --eval_task varying_len --itr varylen_128_S --ts_len 128 --downsample 0.05