from pathlib import Path
from typing import List, Optional, Union, Tuple
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from util.helpers import compare_methods_scatter

if __name__ == "__main__":

    result_dir = "output/clustering" # change as needed
    output_dir = "result/Section5_4" # change as needed
    sp_dist = "symbolic_l1" # symbolic_l1 as default
    bop_dist = "hist_euclidean"

    methods = ['sax', 'spartan']
    itrs = ['a4_w8', 'a4_w8']
    display_names = [method.upper() for method in methods]
    dist_measures = sp_dist

    _ = compare_methods_scatter(
        methods=methods,
        itrs=itrs,
        display_names=display_names,
        dist_measures=dist_measures,
        result_dir=result_dir,
        output_dir=output_dir,
        save_filename=f"1NN_comparison_plot_{display_names[0]}_{display_names[1]}_{itrs[-1]}_tab.png",
        metric="ri",  
        result_csv_name="clustering_results.csv"
    )

    methods = ['sfa', 'spartan']
    itrs = ['a4_w8', 'a4_w8']
    display_names = [method.upper() for method in methods]
    dist_measures = sp_dist

    _ = compare_methods_scatter(
        methods=methods,
        itrs=itrs,
        display_names=display_names,
        dist_measures=dist_measures,
        result_dir=result_dir,
        output_dir=output_dir,
        save_filename=f"1NN_comparison_plot_{display_names[0]}_{display_names[1]}_{itrs[-1]}_tab.png",
        metric="ri",  
        result_csv_name="clustering_results.csv"
    )

    methods = ['sax', 'spartan']
    itrs = ['a4_w4_win0.05', 'a4_w4_win0.05']
    display_names = [method.upper() for method in methods]
    dist_measures = bop_dist

    _ = compare_methods_scatter(
        methods=methods,
        itrs=itrs,
        display_names=display_names,
        dist_measures=dist_measures,
        result_dir=result_dir,
        output_dir=output_dir,
        save_filename=f"1NN_comparison_plot_{display_names[0]}_{display_names[1]}_{itrs[-1]}_tab.png",
        metric="ri",  
        result_csv_name="clustering_results.csv"
    )

    methods = ['sfa', 'spartan']
    itrs = ['a4_w4_win0.05', 'a4_w4_win0.05']
    display_names = [method.upper() for method in methods]
    dist_measures = bop_dist

    _ = compare_methods_scatter(
        methods=methods,
        itrs=itrs,
        display_names=display_names,
        dist_measures=dist_measures,
        result_dir=result_dir,
        output_dir=output_dir,
        save_filename=f"1NN_comparison_plot_{display_names[0]}_{display_names[1]}_{itrs[-1]}_tab.png",
        metric="ri", 
        result_csv_name="clustering_results.csv"
    )