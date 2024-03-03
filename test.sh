python3 train.py --data ../TSC-Benchmark/tscbench/data/Univariate_ts/ --classifier sax --config ./configs/sax/sax_symbol.json --problem ArrowHead

python3 train.py --data ../TSC-Benchmark/tscbench/data/Univariate_ts/ --classifier sfa --config ./configs/sfa/sfa_symbol.json --problem ArrowHead

python3 train.py --data ../TSC-Benchmark/tscbench/data/Univariate_ts/ --classifier spartan --config ./configs/spartan/spartan_symbol.json --problem ArrowHead

python3 train.py --data ../TSC-Benchmark/tscbench/data/Univariate_ts/ --classifier spartan --config ./configs/spartan/spartan_budget_symbol.json --problem ArrowHead

python3 train.py --data ../TSC-Benchmark/tscbench/data/Univariate_ts/ --classifier spartan --config ./configs/spartan_pca/spartan_pca_a4_w4_win12_hist.json --problem ArrowHead
